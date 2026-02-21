# 🚀 Autolane Handoff

**Multi-agent autonomous delivery coordination powered by NVIDIA Cosmos**

Autolane Handoff is a two-stage AI system that watches curbside video feeds and performs real-time reasoning about multi-agent delivery handoffs. It combines [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) for pixel-accurate detection with [NVIDIA Cosmos Reason 2](https://huggingface.co/nvidia/Cosmos-Reason2-8B) (post-trained on synthetic curbside data) for physical reasoning and coordination planning.

Built by [Autolane](https://autolane.ai) — an NVIDIA Inception company building orchestration infrastructure for autonomous last-mile logistics.

> **🏆 Entry for the [NVIDIA Cosmos Cookoff](https://luma.com/nvidia-cosmos-cookoff) (Jan 29 – Mar 5, 2026)**

---

## The Problem

Autonomous delivery is here — Tesla vehicles, Waymo robotaxis, delivery bots, and humanoid robots are all operating at curbside zones. But coordinating handoffs between these heterogeneous agents is unsolved. When a Tesla Model Y arrives with a delivery, how does a robot know when it's safe to approach? What if the loading zone is restricted? What if pedestrians are in the path?

Today, this requires human operators monitoring cameras. **Autolane Handoff automates this reasoning.**

## The Cosmos Flywheel

This project demonstrates NVIDIA's full Cosmos ecosystem working as a closed loop:

```
Real photos (Stanford Shopping Center pilot site)
  → Cosmos Predict 2.5 (Image2World + Text2World)
    → Synthetic curbside delivery videos
      → Grounding DINO annotations
        → Cosmos Reason 2 post-training (SFT)
          → Production handoff reasoning
```

Three Cosmos models. One pipeline. A flywheel that scales to any new deployment site in hours.

## Architecture

| Stage | Model | Role |
|---|---|---|
| Data Generation | Cosmos Predict 2.5-14B | Synthetic training video generation from site photos |
| Detection | Grounding DINO | Open-vocabulary agent detection + bounding boxes |
| Reasoning | Cosmos Reason 2-8B (post-trained) | Safety, OCR, compliance, handoff planning, alerts |

**Pipeline flow:** Video → FFmpeg chunking → Frame extraction → Grounding DINO → Detection overlay → Cosmos Reason 2 → Structured reasoning output

**Detected agents:** Tesla Model Y Juniper, Tesla Model 3, Waymo Jaguar I-PACE, delivery robots, humanoid robots, pedestrians, signage

## Quick Start

### Prerequisites

- NVIDIA GPU with ≥ 40GB VRAM (A100 80GB recommended, or GCP Vertex AI)
- Python 3.10+
- CUDA 12.x
- Node.js 20+ (for dashboard)

### Setup

```bash
# Clone
git clone https://github.com/autolane/autolane-handoff.git
cd autolane-handoff

# Install Python dependencies
pip install -r requirements.txt

# Deploy models
./scripts/deploy_models.sh

# Run inference on a sample video
python inference/pipeline.py --video samples/curbside_demo.mp4

# Start the dashboard
cd dashboard && npm install && npm run dev
```

### Post-Training (reproduce our results)

```bash
# Generate synthetic data (requires Cosmos Predict 2.5)
python datagen/generate_image2world.py --photos datagen/site_photos/
python datagen/generate_text2world.py --prompts datagen/prompts/text2world_base.json

# Annotate with Grounding DINO
python detection/grounding_dino.py --input data/synthetic_clips/ --output data/annotated/

# Bootstrap QA pairs
python data/bootstrap_annotations.py --input data/annotated/ --output data/train.json

# Run SFT
python training/custom_sft.py --config training/sft_config.toml
```

## Evaluation

| Metric | Zero-Shot | Post-Trained |
|---|---|---|
| Safety Assessment Accuracy | TBD | TBD |
| Signage OCR Accuracy | TBD | TBD |
| Handoff Plan Quality | TBD | TBD |
| Agent Detection Recall | TBD | TBD |

## Project Structure

```
autolane-handoff/
├── README.md              # This file
├── CLAUDE.md              # AI assistant context
├── LICENSE
├── requirements.txt
├── docs/                  # Architecture documentation
├── datagen/               # Cosmos Predict 2.5 synthetic data generation
│   ├── prompts/           # Base prompts (Image2World + Text2World)
│   └── site_photos/       # Stanford Shopping Center reference photos
├── data/                  # Dataset preparation + annotation
├── detection/             # Grounding DINO detection stage
├── training/              # SFT post-training (ITS recipe)
├── inference/             # Two-stage inference pipeline
├── dashboard/             # Next.js web dashboard
├── scripts/               # Setup + deployment scripts
└── evaluation/            # Evaluation + comparison results
```

## About Autolane

[Autolane](https://autolane.ai) builds orchestration infrastructure for autonomous last-mile logistics. We coordinate heterogeneous autonomous systems — Tesla vehicles, Unitree G1 humanoid robots, and delivery bots — at retail locations including Simon Property Group's Stanford Shopping Center and Barton Creek Mall.

Autolane is an [NVIDIA Inception](https://www.nvidia.com/en-us/startups/) company.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Acknowledgments

- [NVIDIA Cosmos](https://github.com/nvidia-cosmos) — Cosmos Reason 2, Cosmos Predict 2.5
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) — Open-vocabulary detection
- [NVIDIA AI Blueprint for VSS](https://build.nvidia.com/nvidia/video-search-and-summarization) — Architectural inspiration
- [Cosmos Cookbook ITS Recipe](https://github.com/nvidia-cosmos/cosmos-cookbook) — Post-training methodology
