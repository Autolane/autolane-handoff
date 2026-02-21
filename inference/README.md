# Inference Pipeline

Two-stage inference: Grounding DINO detection → Cosmos Reason 2 reasoning.

## Components

- `pipeline.py` — Orchestrates the full two-stage pipeline
- `video_processor.py` — FFmpeg chunking + frame extraction
- `prompts.py` — System and per-pass prompts for Cosmos Reason 2
- `output_parser.py` — Parse structured CoT outputs into JSON
- `server.py` — FastAPI server exposing the pipeline as an API

## Usage

```bash
# Run pipeline on a single video
python pipeline.py --video path/to/video.mp4

# Start API server
uvicorn server:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /analyze` — Submit a video for analysis
- `GET /stream/{job_id}` — SSE stream for real-time results
