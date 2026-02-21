# CLAUDE.md — Autolane Handoff

## Project Overview

**Autolane Handoff** is a competition entry for the NVIDIA Cosmos Cookoff (Jan 29 – Mar 5, 2026). It's a two-stage AI system that watches curbside video feeds and reasons about multi-agent autonomous delivery handoffs using Grounding DINO + NVIDIA Cosmos Reason 2.

This project directly advances Autolane's core platform — the handoff reasoning system and synthetic data pipeline are production-bound components for Autolane Portal.

## Architecture

Three-model Cosmos Flywheel:

1. **Cosmos Predict 2.5** — Generates synthetic curbside delivery training videos from real Stanford Shopping Center photos (Image2World) and text prompts (Text2World)
2. **Grounding DINO** — Open-vocabulary object detection for pixel-accurate bounding boxes on all agents (Tesla Model Y Juniper, Tesla Model 3, Waymo Jaguar I-PACE, delivery bots, humanoid robots, pedestrians, signage)
3. **Cosmos Reason 2-8B** (post-trained) — Physical reasoning over annotated frames: safety assessment, signage OCR, zone compliance, multi-agent handoff sequencing, alert generation

Pipeline: Video → FFmpeg chunking → Frame extraction → Grounding DINO detection → Overlay rendering → Cosmos Reason 2 reasoning → Structured JSON output

## Tech Stack

- **GCP Project:** `autolane-handoff-20260221`
- **GPU Compute:** GCP Vertex AI (A100 instances)
- **Synthetic Data Gen:** Cosmos Predict 2.5-14B (Image2World + Text2World)
- **Detection:** Grounding DINO (open-vocabulary, zero-shot)
- **Post-Training:** Cosmos-RL SFT framework (ITS recipe), Llava-format datasets
- **Inference:** vLLM serving Cosmos Reason 2-8B
- **Backend:** Python FastAPI
- **Dashboard:** Next.js + React (Radix UI components)
- **Video Processing:** FFmpeg + Python

## Directory Structure

```
autolane-handoff/
├── CLAUDE.md              # This file
├── README.md              # Public-facing project overview + setup
├── LICENSE                # Apache 2.0
├── docs/                  # Architecture docs
├── datagen/               # Cosmos Predict 2.5 synthetic data generation
│   ├── prompts/           # Base prompts (Image2World + Text2World)
│   └── site_photos/       # Stanford Shopping Center reference photos
├── data/                  # Dataset preparation + annotation tools
├── detection/             # Grounding DINO detection stage
├── training/              # SFT post-training scripts + configs
├── inference/             # Two-stage inference pipeline + FastAPI server
├── dashboard/             # Next.js web UI
├── scripts/               # Setup and deployment scripts
└── evaluation/            # Evaluation scripts + results
```

## Key Vehicle Models

All prompts and detection classes reference specific vehicles:
- **Tesla Model Y Juniper** — White, Autolane fleet vehicle
- **Tesla Model 3** — Silver/black, Autolane fleet vehicle
- **Waymo Jaguar I-PACE** — White with roof-mounted lidar sensor array

## Development Conventions

- Python code uses type hints, docstrings, and follows PEP 8
- Config files use TOML (training) and YAML (evaluation)
- All Python dependencies in `requirements.txt` at project root
- Dashboard follows Next.js App Router conventions
- Use `vi` for quick terminal edits
- Commit messages: conventional commits format (`feat:`, `fix:`, `docs:`, etc.)

## Data Format

Training data follows Llava format for Cosmos SFT (per ITS recipe):

```json
{
    "id": "curbside_handoff_001",
    "video": "path/to/video.mp4",
    "conversations": [
        {"from": "human", "value": "<video>\n[prompt]"},
        {"from": "gpt", "value": "<think>\n[reasoning]\n</think>\n<answer>\n[structured output]\n</answer>"}
    ]
}
```

## Inference Prompting

Stage 2 (Cosmos Reason 2) uses a two-pass approach:
1. **Pass 1 — Safety + Compliance:** Spatial relationships, signage OCR, pedestrian conflicts, obstruction detection, safety score (0-100)
2. **Pass 2 — Handoff Planning:** Agent sequencing, trajectory coordinates, estimated handoff time, abort conditions

Both passes receive annotated frames (Grounding DINO BBoxes overlaid) and reason in `<think>` / `<answer>` tags.

## Important Context

- Autolane is an NVIDIA Inception company with active pilots at Simon Property Group locations (Stanford Shopping Center, Barton Creek Mall)
- Competition is judged by NVIDIA, Hugging Face, Nexar, Datature, and Nebius
- Submission deadline: March 5, 2026
- Submit via GitHub issue on nvidia-cosmos/cosmos-cookbook
- Demo video must be < 3 minutes
- Full project plan: see `docs/project-plan.md`
