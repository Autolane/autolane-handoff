#!/bin/bash
# launch_sft.sh — Create GCE VM and run Cosmos Reason 2 SFT training
#
# Creates a spot (preemptible) a2-highgpu-4g instance with 4x A100-40GB,
# sets up the cosmos-rl framework, downloads training data from GCS,
# and runs SFT training.
#
# Usage:
#   # Create VM and run 2B fast iteration (default)
#   ./scripts/launch_sft.sh
#
#   # Run 8B final training
#   ./scripts/launch_sft.sh --model 8b
#
#   # SSH into existing VM
#   ./scripts/launch_sft.sh --ssh
#
#   # Delete VM when done
#   ./scripts/launch_sft.sh --delete

set -euo pipefail

# Defaults
GCP_PROJECT="${GCP_PROJECT_ID:-autolane-handoff-20260221}"
GCP_ZONE="${GCP_ZONE:-us-central1-a}"
GCS_BUCKET="gs://autolane-handoff-datagen"
INSTANCE_NAME="autolane-sft-training"
MACHINE_TYPE="a2-highgpu-4g"  # 4x A100-40GB
DISK_SIZE="500"               # GB
MODEL="2b"
ACTION="create"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --ssh) ACTION="ssh"; shift ;;
        --delete) ACTION="delete"; shift ;;
        --status) ACTION="status"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

case "$ACTION" in
    ssh)
        echo "Connecting to ${INSTANCE_NAME}..."
        gcloud compute ssh "$INSTANCE_NAME" \
            --project="$GCP_PROJECT" \
            --zone="$GCP_ZONE"
        exit 0
        ;;
    delete)
        echo "Deleting ${INSTANCE_NAME}..."
        gcloud compute instances delete "$INSTANCE_NAME" \
            --project="$GCP_PROJECT" \
            --zone="$GCP_ZONE" \
            --quiet
        echo "Instance deleted."
        exit 0
        ;;
    status)
        gcloud compute instances describe "$INSTANCE_NAME" \
            --project="$GCP_PROJECT" \
            --zone="$GCP_ZONE" \
            --format="table(name,status,machineType.basename(),scheduling.provisioningModel)" 2>/dev/null || echo "Instance not found"
        exit 0
        ;;
esac

echo "=============================================="
echo "  Autolane Handoff — SFT Training Launch"
echo "=============================================="
echo "  Project:   ${GCP_PROJECT}"
echo "  Zone:      ${GCP_ZONE}"
echo "  Instance:  ${INSTANCE_NAME}"
echo "  Machine:   ${MACHINE_TYPE} (4x A100-40GB, spot)"
echo "  Model:     Cosmos-Reason2-$(echo "$MODEL" | tr '[:lower:]' '[:upper:]')"
echo "  Disk:      ${DISK_SIZE}GB SSD"
echo "=============================================="
echo ""

# Check if instance already exists
if gcloud compute instances describe "$INSTANCE_NAME" --project="$GCP_PROJECT" --zone="$GCP_ZONE" &>/dev/null; then
    echo "Instance ${INSTANCE_NAME} already exists. Use --ssh to connect or --delete to remove."
    exit 1
fi

# Check HF_TOKEN
if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN required for gated model download."
    echo "  export HF_TOKEN=hf_xxx"
    exit 1
fi

echo "Creating spot instance..."
gcloud compute instances create "$INSTANCE_NAME" \
    --project="$GCP_PROJECT" \
    --zone="$GCP_ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --boot-disk-size="${DISK_SIZE}GB" \
    --boot-disk-type=pd-ssd \
    --image-family=pytorch-2-7-cu128-ubuntu-2404-nvidia-570 \
    --image-project=deeplearning-platform-release \
    --scopes=cloud-platform \
    --metadata="install-nvidia-driver=True" \
    --maintenance-policy=TERMINATE

echo ""
echo "Waiting for instance to be ready..."
sleep 30

# Wait for SSH to be available
for i in $(seq 1 12); do
    if gcloud compute ssh "$INSTANCE_NAME" --project="$GCP_PROJECT" --zone="$GCP_ZONE" --command="echo ready" &>/dev/null; then
        echo "Instance is ready."
        break
    fi
    echo "  Waiting for SSH... (attempt $i/12)"
    sleep 10
done

echo ""
echo "Uploading training scripts..."

# Upload project files to the VM
gcloud compute scp --recurse \
    training/custom_sft.py \
    training/sft_config.toml \
    training/sft_config_2b.toml \
    scripts/prepare_sft_data.sh \
    "$INSTANCE_NAME":~/ \
    --project="$GCP_PROJECT" \
    --zone="$GCP_ZONE"

echo ""
echo "Running setup on the VM..."

# Run setup script on the VM
gcloud compute ssh "$INSTANCE_NAME" \
    --project="$GCP_PROJECT" \
    --zone="$GCP_ZONE" \
    --command="bash -s" -- <<'SETUP_EOF'
set -euo pipefail

echo "=== Setting up SFT training environment ==="

# Check GPUs
echo ""
echo "--- GPU Status ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPUs available: ${GPU_COUNT}"

# Install system dependencies
echo ""
echo "--- Installing system dependencies ---"
sudo apt-get update -qq
sudo apt-get install -y -qq redis-server > /dev/null 2>&1
sudo systemctl start redis-server
echo "Redis: $(redis-cli ping)"

# Install uv
echo ""
echo "--- Installing uv ---"
curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null
export PATH="$HOME/.local/bin:$PATH"
echo "uv version: $(uv --version)"

# Clone cosmos-reason2
echo ""
echo "--- Cloning cosmos-reason2 ---"
if [ ! -d "$HOME/cosmos-reason2" ]; then
    git clone --depth 1 https://github.com/nvidia-cosmos/cosmos-reason2.git "$HOME/cosmos-reason2"
fi

# Install cosmos-rl environment
echo ""
echo "--- Installing cosmos-rl (this may take a few minutes) ---"
cd "$HOME/cosmos-reason2/examples/cosmos_rl"
uv sync 2>&1 | tail -5

# Download training data from GCS
echo ""
echo "--- Downloading training data from GCS ---"
DATA_DIR="/data/autolane-handoff"
sudo mkdir -p "$DATA_DIR"
sudo chown $(whoami) "$DATA_DIR"
mkdir -p "$DATA_DIR/training"
mkdir -p "$DATA_DIR/data/annotations"

gsutil -m cp \
    gs://autolane-handoff-datagen/data/training/llava_train.json \
    gs://autolane-handoff-datagen/data/training/llava_val.json \
    "$DATA_DIR/training/"

echo "Train entries: $(python3 -c "import json; print(len(json.load(open('$DATA_DIR/training/llava_train.json'))))")"
echo "Val entries: $(python3 -c "import json; print(len(json.load(open('$DATA_DIR/training/llava_val.json'))))")"

echo ""
echo "--- Downloading annotated videos from GCS ---"
gsutil -m rsync -r \
    gs://autolane-handoff-datagen/data/annotations/ \
    "$DATA_DIR/data/annotations/" 2>&1 | tail -3

VIDEO_COUNT=$(find "$DATA_DIR/data/annotations" -name "annotated.mp4" | wc -l)
echo "Downloaded ${VIDEO_COUNT} annotated videos"

# Copy training configs
echo ""
echo "--- Setting up training configs ---"
cp ~/custom_sft.py "$HOME/cosmos-reason2/examples/cosmos_rl/scripts/"
cp ~/sft_config.toml ~/sft_config_2b.toml "$HOME/cosmos-reason2/examples/cosmos_rl/configs/"

echo ""
echo "=== Setup complete ==="
echo "Ready to train. Run:"
echo "  cd ~/cosmos-reason2/examples/cosmos_rl"
echo '  source .venv/bin/activate'
echo "  cosmos-rl --config configs/sft_config_2b.toml --log-dir outputs/ scripts/custom_sft.py"
SETUP_EOF

echo ""
echo "=============================================="
echo "  Setup complete! Launching training..."
echo "=============================================="
echo ""

# Determine config based on model
if [ "$MODEL" = "2b" ]; then
    CONFIG="configs/sft_config_2b.toml"
    LOG_DIR="outputs/autolane_sft_2b"
else
    CONFIG="configs/sft_config.toml"
    LOG_DIR="outputs/autolane_sft_8b"
fi

# Launch training via SSH
MODEL_UPPER=$(echo "$MODEL" | tr '[:lower:]' '[:upper:]')
echo "Starting ${MODEL_UPPER} SFT training..."
echo "Config: ${CONFIG}"
echo ""

gcloud compute ssh "$INSTANCE_NAME" \
    --project="$GCP_PROJECT" \
    --zone="$GCP_ZONE" \
    --command="bash -s" -- <<TRAIN_EOF
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
export HF_TOKEN="${HF_TOKEN}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

cd ~/cosmos-reason2/examples/cosmos_rl
source .venv/bin/activate

echo "Launching cosmos-rl SFT training..."
echo "  Config: ${CONFIG}"
echo "  Log dir: ${LOG_DIR}"
echo ""

# Run training (nohup so it survives SSH disconnect)
nohup cosmos-rl \
    --config "${CONFIG}" \
    --log-dir "${LOG_DIR}" \
    scripts/custom_sft.py \
    > ~/sft_training.log 2>&1 &

TRAIN_PID=\$!
echo "Training started with PID \${TRAIN_PID}"
echo "Training log: ~/sft_training.log"
echo ""
echo "Monitor with:"
echo "  gcloud compute ssh ${INSTANCE_NAME} --zone=${GCP_ZONE} --command='tail -f ~/sft_training.log'"
echo ""
echo "Or SSH in:"
echo "  ./scripts/launch_sft.sh --ssh"

# Wait briefly to check for immediate errors
sleep 15
if ! kill -0 \${TRAIN_PID} 2>/dev/null; then
    echo ""
    echo "WARNING: Training process may have exited early. Check the log:"
    tail -30 ~/sft_training.log
    exit 1
fi

echo "Training is running. Tail of log:"
tail -5 ~/sft_training.log
TRAIN_EOF

echo ""
echo "=============================================="
echo "  Training launched on ${INSTANCE_NAME}"
echo "=============================================="
echo ""
echo "Monitor:"
echo "  ./scripts/launch_sft.sh --status"
echo "  gcloud compute ssh ${INSTANCE_NAME} --zone=${GCP_ZONE} --project=${GCP_PROJECT} --command='tail -f ~/sft_training.log'"
echo ""
echo "SSH:"
echo "  ./scripts/launch_sft.sh --ssh"
echo ""
echo "Cleanup when done:"
echo "  ./scripts/launch_sft.sh --delete"
