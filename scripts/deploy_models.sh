#!/bin/bash
# deploy_models.sh — Deploy all three models for Autolane Handoff
# Run on GPU instance (8xA100 recommended)

set -euo pipefail

echo "=== Autolane Handoff — Model Deployment ==="
echo ""

# --- Cosmos Reason 2-8B (via Vertex AI) ---
echo "[1/3] Deploying Cosmos Reason 2-8B to Vertex AI..."
echo "  Uses prebuilt vLLM container: pytorch-vllm-serve:v0.14.0"
echo "  Machine: a2-highgpu-1g (1x A100 40GB)"
echo "  Run: ./scripts/deploy_reason2.sh"
bash "$(dirname "$0")/deploy_reason2.sh"

# --- Grounding DINO ---
echo "[2/3] Deploying Grounding DINO..."
bash "$(dirname "$0")/deploy_gdino.sh"

# --- Cosmos Predict 2.5-14B (for datagen only) ---
echo "[3/3] Deploying Cosmos Predict 2.5-14B..."
echo "  See scripts/deploy_predict25.sh for dedicated GCP VM setup."
echo "  Only needed during data generation phase (Phase 2)."
echo "  Run: ./scripts/deploy_predict25.sh"

echo ""
echo "=== All models deployed ==="
