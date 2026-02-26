#!/bin/bash
# deploy_reason2.sh — Deploy Cosmos Reason 2-8B to Vertex AI
#
# Uses Google's prebuilt vLLM container on Vertex AI (same pattern as
# av-ride-hail-inference). No manual VM management needed — Vertex AI
# handles provisioning, scaling, and health checks.
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - GCP_PROJECT_ID set (default: autolane-handoff-20260221)
#   - HF_TOKEN set (for gated nvidia/Cosmos-Reason2-8B)
#   - A100 GPU quota in the target region
#   - pip install google-cloud-aiplatform pydantic-settings structlog
#
# Usage:
#   export GCP_PROJECT_ID=autolane-handoff-20260221
#   export HF_TOKEN=hf_xxx
#   ./scripts/deploy_reason2.sh

set -e

# Check required environment variables
if [ -z "$GCP_PROJECT_ID" ]; then
    echo "Error: GCP_PROJECT_ID environment variable is required"
    echo "  export GCP_PROJECT_ID=autolane-handoff-20260221"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable is not set."
    echo "This is required for gated models like nvidia/Cosmos-Reason2-8B"
    echo ""
    echo "To get a token:"
    echo "  1. Accept the license at https://huggingface.co/nvidia/Cosmos-Reason2-8B"
    echo "  2. Get a token at https://huggingface.co/settings/tokens"
    echo "  3. Run: export HF_TOKEN=your_token"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Defaults
GCP_REGION="${GCP_REGION:-us-central1}"
ENDPOINT_NAME="${ENDPOINT_NAME:-autolane-handoff-cosmos-reason2-8b}"
MODEL_NAME="${MODEL_NAME:-cosmos-reason2-8b-handoff}"
MIN_REPLICAS="${MIN_REPLICAS:-1}"
MAX_REPLICAS="${MAX_REPLICAS:-1}"
MACHINE_TYPE="${MACHINE_TYPE:-a2-highgpu-1g}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-NVIDIA_TESLA_A100}"

echo "=============================================="
echo "  Autolane Handoff — Deploy Cosmos Reason 2-8B"
echo "=============================================="
echo "  Project:      $GCP_PROJECT_ID"
echo "  Region:       $GCP_REGION"
echo "  Endpoint:     $ENDPOINT_NAME"
echo "  Machine:      $MACHINE_TYPE with $ACCELERATOR_TYPE"
echo "  Replicas:     $MIN_REPLICAS - $MAX_REPLICAS"
echo "  Container:    pytorch-vllm-serve:v0.14.0"
echo "=============================================="
echo ""

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable aiplatform.googleapis.com --project="$GCP_PROJECT_ID"
gcloud services enable storage.googleapis.com --project="$GCP_PROJECT_ID"

# Check GPU quota
echo ""
echo "Checking A100 GPU quota in $GCP_REGION..."
QUOTA=$(gcloud compute regions describe "$GCP_REGION" \
    --project="$GCP_PROJECT_ID" \
    --format="value(quotas[name=NVIDIA_A100_GPUS].limit)" 2>/dev/null || echo "0")

if [ "$QUOTA" = "0" ] || [ -z "$QUOTA" ]; then
    echo "Warning: No A100 GPU quota found in $GCP_REGION"
    echo "Request quota at: https://console.cloud.google.com/iam-admin/quotas"
    echo ""
fi

# Run deployment via Python (Vertex AI SDK)
echo ""
echo "Starting Vertex AI deployment..."
echo "This will take 15-30 minutes for model download and initialization."
echo ""

poetry run python -m inference.deploy \
    --action deploy \
    --endpoint-name "$ENDPOINT_NAME" \
    --model-name "$MODEL_NAME" \
    --machine-type "$MACHINE_TYPE" \
    --accelerator-type "$ACCELERATOR_TYPE" \
    --min-replicas "$MIN_REPLICAS" \
    --max-replicas "$MAX_REPLICAS"

echo ""
echo "=============================================="
echo "  Deployment initiated"
echo "=============================================="
echo ""
echo "  Monitor progress:"
echo "    https://console.cloud.google.com/vertex-ai/online-prediction/endpoints?project=$GCP_PROJECT_ID"
echo ""
echo "  After deployment, add the endpoint ID to .env:"
echo "    VERTEX_ENDPOINT_ID=<endpoint-id>"
echo ""
echo "=============================================="
