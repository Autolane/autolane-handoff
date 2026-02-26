#!/bin/bash
# deploy_gdino.sh — Deploy Grounding DINO for Autolane Handoff Stage 1 detection
# Run on GCP GPU instance (autolane-handoff-20260221)
#
# Usage:
#   bash scripts/deploy_gdino.sh              # Default: SwinT variant
#   bash scripts/deploy_gdino.sh --swinb      # SwinB variant (more accurate, slower)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEIGHTS_DIR="$PROJECT_ROOT/weights"

# Model variant selection
VARIANT="swint"
if [[ "${1:-}" == "--swinb" ]]; then
    VARIANT="swinb"
fi

echo "=== Autolane Handoff — Grounding DINO Deployment ==="
echo "Variant: $VARIANT"
echo "Project root: $PROJECT_ROOT"
echo ""

# --- 1. Install dependencies ---
echo "[1/4] Installing Python dependencies..."
pip install --quiet groundingdino-py opencv-python-headless Pillow numpy torch torchvision
echo "  Done."

# --- 2. Download model weights ---
echo "[2/4] Downloading model weights..."
mkdir -p "$WEIGHTS_DIR"

if [[ "$VARIANT" == "swint" ]]; then
    WEIGHTS_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    WEIGHTS_FILE="groundingdino_swint_ogc.pth"
elif [[ "$VARIANT" == "swinb" ]]; then
    WEIGHTS_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
    WEIGHTS_FILE="groundingdino_swinb_cogcoor.pth"
fi

if [[ -f "$WEIGHTS_DIR/$WEIGHTS_FILE" ]]; then
    echo "  Weights already present: $WEIGHTS_DIR/$WEIGHTS_FILE"
else
    echo "  Downloading $WEIGHTS_FILE ..."
    wget -q --show-progress -O "$WEIGHTS_DIR/$WEIGHTS_FILE" "$WEIGHTS_URL"
    echo "  Downloaded: $WEIGHTS_DIR/$WEIGHTS_FILE"
fi

# --- 3. Verify installation ---
echo "[3/4] Verifying Grounding DINO installation..."
python -c "
from groundingdino.util.inference import load_model
print('  groundingdino package imported successfully')
" 2>&1

echo "  Weights file: $(ls -lh "$WEIGHTS_DIR/$WEIGHTS_FILE" | awk '{print $5, $NF}')"

# --- 4. Quick GPU check ---
echo "[4/4] Checking GPU availability..."
python -c "
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_count = torch.cuda.device_count()
    print(f'  GPU available: {gpu_count}x {gpu_name}')
else:
    print('  No GPU detected — will run on CPU (slower)')
"

echo ""
echo "=== Grounding DINO ($VARIANT) ready ==="
echo ""
echo "Run detection:"
echo "  python -m detection.grounding_dino --input <frames_dir> --output results.json"
echo ""
echo "Render overlays:"
echo "  python -m detection.overlay --detections results.json --output annotated/"
