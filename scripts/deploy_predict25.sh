#!/bin/bash
# deploy_predict25.sh — Deploy Cosmos Predict 2.5-14B on GCP Compute Engine
#
# ⚠️  DEPRECATED: This script creates a raw Compute Engine VM that requires
# manual SSH, SCP, and cleanup. Use scripts/launch_datagen.sh instead, which
# runs datagen as a Vertex AI Custom Job with automatic provisioning, GCS I/O,
# and auto-termination.
#
#   ./scripts/launch_datagen.sh --help
#
# Usage (legacy):
#   ./scripts/deploy_predict25.sh [--zone us-central1-a] [--machine-type a2-highgpu-1g]
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - GCP project set: gcloud config set project autolane-handoff-20260221
#   - Hugging Face token with read access (for model download)

set -euo pipefail

echo "⚠️  DEPRECATED: Use scripts/launch_datagen.sh instead (Vertex AI Custom Jobs)."
echo "   This script creates a raw Compute Engine VM requiring manual SSH/SCP/cleanup."
echo ""

# --- Configuration ---
PROJECT_ID="${GCP_PROJECT_ID:-autolane-handoff-20260221}"
ZONE="${GCP_ZONE:-us-central1-a}"
MACHINE_TYPE="${GCP_MACHINE_TYPE:-a2-highgpu-1g}"  # 1x A100 40GB
INSTANCE_NAME="predict25-14b"
BOOT_DISK_SIZE="200GB"
ACCELERATOR_TYPE="nvidia-tesla-a100"
ACCELERATOR_COUNT=1
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --zone) ZONE="$2"; shift 2 ;;
        --machine-type) MACHINE_TYPE="$2"; shift 2 ;;
        --project) PROJECT_ID="$2"; shift 2 ;;
        --instance-name) INSTANCE_NAME="$2"; shift 2 ;;
        --accelerator-count) ACCELERATOR_COUNT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Autolane Handoff — Cosmos Predict 2.5-14B Deployment ==="
echo ""
echo "  Project:      $PROJECT_ID"
echo "  Zone:         $ZONE"
echo "  Machine:      $MACHINE_TYPE"
echo "  GPU:          ${ACCELERATOR_COUNT}x $ACCELERATOR_TYPE"
echo "  Instance:     $INSTANCE_NAME"
echo ""

# --- Step 1: Create GCP Compute Engine VM ---
echo "[1/4] Creating GCP Compute Engine VM with A100 GPU..."

gcloud compute instances create "$INSTANCE_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=$ACCELERATOR_TYPE,count=$ACCELERATOR_COUNT" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True" \
    --scopes=cloud-platform

echo "  VM created: $INSTANCE_NAME"
echo ""

# --- Step 2: Wait for VM to be ready ---
echo "[2/4] Waiting for VM to be ready..."
sleep 30

# Wait for SSH to become available
for i in $(seq 1 12); do
    if gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" \
        --command="echo 'VM ready'" 2>/dev/null; then
        break
    fi
    echo "  Waiting for SSH... (attempt $i/12)"
    sleep 15
done

echo ""

# --- Step 3: Install dependencies on the VM ---
echo "[3/4] Installing dependencies on VM..."

gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT_ID" --command="
set -euo pipefail

echo '--- Installing system dependencies ---'
sudo apt-get update -qq
sudo apt-get install -y -qq git git-lfs curl ffmpeg libx11-dev tree wget

echo '--- Installing uv package manager ---'
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=\"\$HOME/.local/bin:\$PATH\"

echo '--- Cloning Cosmos Predict 2.5 repository ---'
git lfs install
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git ~/cosmos-predict2.5
cd ~/cosmos-predict2.5
git lfs pull

echo '--- Setting up Python environment ---'
uv python install
uv sync --extra=cu128

echo '--- Installing Hugging Face CLI ---'
uv tool install -U 'huggingface_hub[cli]'

echo '--- Installing diffusers for alternative pipeline ---'
source .venv/bin/activate
pip install diffusers torch torchvision Pillow

echo '--- Dependencies installed ---'
nvidia-smi
python -c 'import torch; print(f\"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}\")'
"

echo ""

# --- Step 4: Download model weights ---
echo "[4/4] Downloading Cosmos Predict 2.5-14B model..."
echo ""
echo "  IMPORTANT: You need to authenticate with Hugging Face."
echo "  SSH into the VM and run:"
echo ""
echo "    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
echo "    huggingface-cli login"
echo ""
echo "  Then accept the NVIDIA Open Model License at:"
echo "    https://huggingface.co/nvidia/Cosmos-Predict2.5-14B"
echo ""
echo "  Model weights will download automatically on first inference run."
echo "  Or pre-download with:"
echo ""
echo "    cd ~/cosmos-predict2.5"
echo "    source .venv/bin/activate"
echo "    python -c \"from huggingface_hub import snapshot_download; snapshot_download('nvidia/Cosmos-Predict2.5-14B')\""
echo ""

echo "=== Deployment complete ==="
echo ""
echo "Next steps:"
echo "  1. SSH into the VM:  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "  2. Authenticate with Hugging Face:  huggingface-cli login"
echo "  3. Copy generation scripts:  gcloud compute scp datagen/*.py $INSTANCE_NAME:~/autolane-handoff/datagen/ --zone=$ZONE"
echo "  4. Copy site photos:  gcloud compute scp datagen/site_photos/*.jpg $INSTANCE_NAME:~/autolane-handoff/datagen/site_photos/ --zone=$ZONE"
echo "  5. Run Image2World:  python datagen/generate_image2world.py"
echo "  6. Run Text2World:  python datagen/generate_text2world.py"
echo ""
echo "To stop the VM (save costs):"
echo "  gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
echo ""
echo "To delete the VM:"
echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
