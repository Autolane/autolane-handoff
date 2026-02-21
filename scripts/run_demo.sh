#!/bin/bash
# run_demo.sh — End-to-end demo runner
# Runs the full pipeline on a sample video

set -euo pipefail

VIDEO=${1:-"samples/curbside_demo.mp4"}

echo "=== Autolane Handoff — Demo ==="
echo "Input: $VIDEO"
echo ""

# Stage 1: Detection
echo "[Stage 1] Running Grounding DINO detection..."
python detection/grounding_dino.py --video "$VIDEO" --output /tmp/detections.json

# Stage 2: Reasoning
echo "[Stage 2] Running Cosmos Reason 2 reasoning..."
python inference/pipeline.py --video "$VIDEO" --detections /tmp/detections.json

echo ""
echo "=== Demo complete ==="
