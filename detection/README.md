# Detection Stage — Grounding DINO

Open-vocabulary object detection for all agents in curbside scenes.

## Detection Classes

```
Tesla Model Y . Tesla Model 3 . Waymo Jaguar I-PACE . delivery robot . humanoid robot . pedestrian . parking sign . loading zone sign . obstruction
```

## Usage

```bash
# Run detection on video clips
python grounding_dino.py --input ../data/synthetic_clips/ --output ../data/annotated/

# Render detection overlays onto frames
python overlay.py --detections results.json --frames frames/ --output annotated_frames/
```
