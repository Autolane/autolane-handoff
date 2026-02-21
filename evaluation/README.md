# Evaluation

Compare zero-shot vs post-trained Cosmos Reason 2 on curbside handoff tasks.

## Metrics

- Safety Assessment Accuracy
- Signage OCR Accuracy
- Handoff Plan Quality (human-rated)
- Agent Detection Recall (via Grounding DINO)

## Usage

```bash
python compare.py --zero-shot results/zero_shot_results.json --post-trained results/post_trained_results.json
```
