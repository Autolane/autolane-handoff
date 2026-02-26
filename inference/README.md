# Inference Pipeline

Two-stage inference: Grounding DINO detection -> Cosmos Reason 2 reasoning.

## Components

- `config.py` -- Settings via pydantic-settings (loads `.env`)
- `deploy.py` -- Vertex AI deployment (model upload + endpoint creation)
- `client.py` -- Inference client using Vertex AI `rawPredict` API
- `prompts.py` -- System prompt and per-pass prompts for Cosmos Reason 2
- `pipeline.py` -- Orchestrates the full two-stage pipeline (TODO)
- `video_processor.py` -- FFmpeg chunking + frame extraction (TODO)
- `output_parser.py` -- Parse structured CoT outputs into JSON (TODO)
- `server.py` -- FastAPI server exposing the pipeline as an API (TODO)

## Deployment

Cosmos Reason 2-8B is deployed to GCP Vertex AI using Google's prebuilt
vLLM container (`pytorch-vllm-serve:v0.14.0`).

```bash
# 1. Set environment variables
export GCP_PROJECT_ID=autolane-handoff-20260221
export HF_TOKEN=hf_xxx

# 2. Deploy to Vertex AI (takes 15-30 min)
./scripts/deploy_reason2.sh

# 3. After deployment, add endpoint ID to .env
echo "VERTEX_ENDPOINT_ID=<id>" >> .env
```

## Inference

```python
import asyncio
from inference.client import InferenceClient

client = InferenceClient()

# Full two-pass analysis (safety + handoff)
result = asyncio.run(client.analyze_video("/path/to/annotated_video.mp4"))

# Or run passes individually
safety = asyncio.run(client.assess_safety("/path/to/annotated_video.mp4"))
handoff = asyncio.run(client.plan_handoff("/path/to/annotated_video.mp4"))
```

## API Endpoints (via FastAPI server)

- `POST /analyze` -- Submit a video for analysis
- `GET /stream/{job_id}` -- SSE stream for real-time results
