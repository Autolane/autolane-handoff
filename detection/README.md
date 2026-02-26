# Detection Stage -- Grounding DINO

Open-vocabulary object detection for all agents in curbside scenes (Stage 1 of the Autolane Handoff pipeline).

## Setup

```bash
# Deploy Grounding DINO (installs deps + downloads weights)
bash scripts/deploy_gdino.sh

# Or for the more accurate SwinB variant:
bash scripts/deploy_gdino.sh --swinb
```

## Detection Classes

```
Tesla Model Y . Tesla Model 3 . Waymo Jaguar I-PACE . delivery robot . humanoid robot . pedestrian . parking sign . loading zone sign . obstruction
```

Class groupings:
- **Vehicles:** Tesla Model Y, Tesla Model 3, Waymo Jaguar I-PACE
- **Robots:** delivery robot, humanoid robot
- **Humans:** pedestrian
- **Signage:** parking sign, loading zone sign
- **Hazards:** obstruction

## Usage

```bash
# Run detection on a directory of frames
python -m detection.grounding_dino --input data/frames/ --output data/results.json

# Run detection on a single image
python -m detection.grounding_dino --input frame.jpg --output results.json

# Adjust thresholds
python -m detection.grounding_dino --input data/frames/ --output results.json \
    --box-threshold 0.4 --text-threshold 0.3

# Render BBox overlays onto frames (for Stage 2 Cosmos Reason 2 input)
python -m detection.overlay --detections data/results.json --output data/annotated/
```

## Python API

```python
from detection.grounding_dino import GroundingDINODetector, save_results

# Initialize and load model
detector = GroundingDINODetector(variant="swint")
detector.load()

# Detect objects in a single image
result = detector.detect_image("frame.jpg")
for det in result.detections:
    print(f"[{det.confidence:.0%}] {det.label} @ ({det.x_min:.0f},{det.y_min:.0f})-({det.x_max:.0f},{det.y_max:.0f})")

# Detect across a directory of frames
results = detector.detect_video_frames("data/frames/")
save_results(results, "data/results.json")

# Render overlays
from detection.overlay import render_batch
from detection.grounding_dino import load_results

results = load_results("data/results.json")
render_batch(results, "data/annotated/")
```

## Files

- `grounding_dino.py` -- Detection inference wrapper (load model, run detection, serialize results)
- `overlay.py` -- BBox overlay renderer (burn annotations into frames for Stage 2)
- `prompts.py` -- Detection prompt config and class groupings
- `__init__.py` -- Package init

## Model Variants

| Variant | Backbone | Speed | Accuracy | Weights |
|---------|----------|-------|----------|---------|
| `swint` (default) | Swin-T | Faster | Good | 694 MB |
| `swinb` | Swin-B | Slower | Better | 938 MB |

## Output Format

Detection results JSON:
```json
[
    {
        "frame_path": "data/frames/frame_001.jpg",
        "width": 1920,
        "height": 1080,
        "detections": [
            {
                "label": "Tesla Model Y",
                "confidence": 0.8234,
                "x_min": 150.0,
                "y_min": 300.0,
                "x_max": 650.0,
                "y_max": 700.0
            }
        ]
    }
]
```
