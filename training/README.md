# Post-Training — Cosmos Reason 2 SFT

Fine-tune Cosmos Reason 2 on curbside handoff domain data using the ITS recipe from the Cosmos Cookbook.

## Overview

| Config | Model | GPUs | Time/Epoch | Use |
|---|---|---|---|---|
| `sft_config_2b.toml` | Cosmos-Reason2-2B | 4x A100-80GB | ~20 min | Fast iteration |
| `sft_config.toml` | Cosmos-Reason2-8B | 8x A100-80GB | ~1-2 hours | Final training |

## Prerequisites

```bash
# 1. Clone cosmos-reason2 repo (provides cosmos-rl framework)
git clone https://github.com/nvidia-cosmos/cosmos-reason2.git
cd cosmos-reason2/examples/cosmos_rl

# 2. Install cosmos-rl + dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate

# 3. Install Redis (required by cosmos-rl)
conda install -c conda-forge redis-server
# OR: sudo apt-get install redis-server

# 4. Authenticate with Hugging Face (model download)
huggingface-cli login

# 5. Download training data from GCS
cd /path/to/autolane-handoff
./scripts/prepare_sft_data.sh /data/autolane-handoff
```

## Training

```bash
# Copy training scripts to cosmos-rl scripts dir
cp training/custom_sft.py cosmos-reason2/examples/cosmos_rl/scripts/
cp training/sft_config*.toml cosmos-reason2/examples/cosmos_rl/configs/

# Fast iteration with 2B model
cd cosmos-reason2/examples/cosmos_rl
cosmos-rl \
    --config configs/sft_config_2b.toml \
    --log-dir outputs/autolane_sft_2b \
    scripts/custom_sft.py

# Final training with 8B model
cosmos-rl \
    --config configs/sft_config.toml \
    --log-dir outputs/autolane_sft_8b \
    scripts/custom_sft.py
```

## Evaluation

Compare post-trained model against zero-shot baseline:

```bash
# Against Vertex AI endpoints
python -m training.evaluate \
    --val-data data/training/llava_val.json \
    --zero-shot-url https://us-central1-aiplatform.googleapis.com/v1/projects/autolane-handoff-20260221/locations/us-central1/endpoints/ZERO_SHOT_ENDPOINT_ID \
    --post-trained-url https://us-central1-aiplatform.googleapis.com/v1/projects/autolane-handoff-20260221/locations/us-central1/endpoints/POST_TRAINED_ENDPOINT_ID \
    --media-path /data/autolane-handoff

# Against local vLLM endpoints
python -m training.evaluate \
    --val-data data/training/llava_val.json \
    --zero-shot-url http://localhost:8001/v1 \
    --post-trained-url http://localhost:8002/v1 \
    --media-path /data/autolane-handoff \
    --local
```

Results are saved to `evaluation/results/`:
- `eval_results.json` — Per-sample detailed results
- `eval_metrics.json` — Aggregate metrics
- `comparison.md` — Markdown comparison table for demo

## Files

| File | Purpose |
|---|---|
| `custom_sft.py` | SFT training script (cosmos-rl + Llava format) |
| `sft_config.toml` | 8B model training config |
| `sft_config_2b.toml` | 2B model training config (fast iteration) |
| `evaluate.py` | Zero-shot vs post-trained comparison |

## Training Data

- **636 train** / **70 val** QA pairs across 353 clips
- Each clip produces 2 QA pairs: safety assessment + handoff planning
- Llava format with `<think>/<answer>` structured outputs
- Videos are Grounding DINO-annotated clips (bounding box overlays)
- See `data/README.md` for the full annotation pipeline
