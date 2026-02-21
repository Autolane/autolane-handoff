# Post-Training — Cosmos Reason 2 SFT

Fine-tune Cosmos Reason 2 on curbside handoff domain data using the ITS recipe from the Cosmos Cookbook.

## Config

- `sft_config.toml` — Training configuration adapted from ITS recipe
- Start with 2B model for fast iteration (~20 min/epoch)
- Final training on 8B model (~1-2 hours)

## Usage

```bash
# Train
python custom_sft.py --config sft_config.toml

# Evaluate post-trained vs zero-shot
python evaluate.py --config eval_config.yaml
```
