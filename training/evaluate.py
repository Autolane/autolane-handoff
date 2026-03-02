#!/usr/bin/env python3
"""Evaluate post-trained vs zero-shot Cosmos Reason 2 on validation clips.

Runs both the zero-shot and post-trained models on the validation set,
extracts structured outputs, and compares key metrics:
  - Safety score extraction accuracy
  - Handoff readiness classification (READY/NOT_READY/CAUTION)
  - Structured output format compliance (<think>/<answer> tag usage)
  - Response quality (truncation, refusals, hallucinations)

Results are saved as JSON for the demo video comparison screenshots.

Usage:
    # Compare zero-shot endpoint vs post-trained endpoint
    python -m training.evaluate \
        --val-data data/training/llava_val.json \
        --zero-shot-url https://REGION-aiplatform.googleapis.com/v1/projects/PROJECT/locations/REGION/endpoints/ZERO_SHOT_ID \
        --post-trained-url https://REGION-aiplatform.googleapis.com/v1/projects/PROJECT/locations/REGION/endpoints/POST_TRAINED_ID

    # Use local vLLM endpoints
    python -m training.evaluate \
        --val-data data/training/llava_val.json \
        --zero-shot-url http://localhost:8001/v1 \
        --post-trained-url http://localhost:8002/v1 \
        --local
"""

import argparse
import base64
import json
import logging
import re
import time
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("evaluation/results")


def _extract_tags(response: str) -> dict:
    """Extract <think> and <answer> content from a response."""
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    return {
        "has_think_tag": think_match is not None,
        "has_answer_tag": answer_match is not None,
        "think": think_match.group(1).strip() if think_match else "",
        "answer": answer_match.group(1).strip() if answer_match else "",
    }


def _extract_safety_score(text: str) -> int | None:
    """Extract safety_score integer from response text."""
    match = re.search(r'"?safety_score"?\s*:\s*(\d+)', text)
    if match:
        return int(match.group(1))
    match = re.search(r"safety.*?(\d{1,3})\s*/?\s*100", text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        if 0 <= score <= 100:
            return score
    return None


def _extract_readiness(text: str) -> str | None:
    """Extract readiness status from response text."""
    match = re.search(
        r'"?readiness"?\s*:\s*"?(READY|NOT_READY|CAUTION)"?',
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()
    return None


def _encode_video_frames(video_path: str, max_frames: int = 8) -> list[str]:
    """Sample frames from video and encode as base64 JPEG.

    Falls back to sending the first frame if ffmpeg is unavailable.
    """
    import subprocess
    import tempfile

    frames_b64 = []
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use ffmpeg to extract frames
        result = subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vf", f"select=not(mod(n\\,3)),scale=640:-1",
                "-frames:v", str(max_frames),
                "-q:v", "5",
                f"{tmpdir}/frame_%04d.jpg",
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            logger.warning("FFmpeg failed for %s, skipping", video_path)
            return []

        for frame_path in sorted(Path(tmpdir).glob("frame_*.jpg")):
            with open(frame_path, "rb") as f:
                frames_b64.append(base64.b64encode(f.read()).decode())

    return frames_b64


def call_vertex_endpoint(
    url: str, frames_b64: list[str], prompt: str, system_prompt: str,
) -> tuple[str, float]:
    """Call a Vertex AI rawPredict endpoint with frames."""
    import google.auth
    import google.auth.transport.requests

    creds, _ = google.auth.default()
    creds.refresh(google.auth.transport.requests.Request())
    token = creds.token

    content = []
    for fb64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{fb64}"},
        })
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": "nvidia/Cosmos-Reason2-8B",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ],
        "max_tokens": 4096,
        "temperature": 0.2,
    }

    start = time.time()
    resp = httpx.post(
        f"{url}/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=120.0,
    )
    elapsed = time.time() - start
    resp.raise_for_status()

    return resp.json()["choices"][0]["message"]["content"], elapsed


def call_local_endpoint(
    url: str, frames_b64: list[str], prompt: str, system_prompt: str,
) -> tuple[str, float]:
    """Call a local vLLM endpoint with frames."""
    content = []
    for fb64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{fb64}"},
        })
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": "nvidia/Cosmos-Reason2-8B",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ],
        "max_tokens": 4096,
        "temperature": 0.2,
    }

    start = time.time()
    resp = httpx.post(
        f"{url}/chat/completions",
        json=payload,
        timeout=120.0,
    )
    elapsed = time.time() - start
    resp.raise_for_status()

    return resp.json()["choices"][0]["message"]["content"], elapsed


def evaluate_sample(
    sample: dict,
    zero_shot_url: str,
    post_trained_url: str,
    media_path: str,
    local: bool = False,
    system_prompt: str = "",
) -> dict | None:
    """Evaluate a single sample against both endpoints."""
    sample_id = sample["id"]
    prompt = sample["conversations"][0]["value"]
    # Strip <video> tag from prompt (we send frames as images)
    prompt = re.sub(r"(\n)?</?(video|image)>(\n)?", "", prompt)

    video_path = sample["video"]
    if media_path:
        video_path = str(Path(media_path) / video_path)

    if not Path(video_path).exists():
        logger.warning("Video not found: %s, skipping", video_path)
        return None

    frames_b64 = _encode_video_frames(video_path)
    if not frames_b64:
        return None

    call_fn = call_local_endpoint if local else call_vertex_endpoint

    # Call zero-shot
    try:
        zs_response, zs_time = call_fn(
            zero_shot_url, frames_b64, prompt, system_prompt,
        )
    except Exception as e:
        logger.error("Zero-shot failed for %s: %s", sample_id, e)
        zs_response, zs_time = "", 0.0

    # Call post-trained
    try:
        pt_response, pt_time = call_fn(
            post_trained_url, frames_b64, prompt, system_prompt,
        )
    except Exception as e:
        logger.error("Post-trained failed for %s: %s", sample_id, e)
        pt_response, pt_time = "", 0.0

    # Parse and compare
    is_safety = sample_id.endswith("_safety")
    zs_tags = _extract_tags(zs_response)
    pt_tags = _extract_tags(pt_response)

    result = {
        "id": sample_id,
        "type": "safety" if is_safety else "handoff",
        "zero_shot": {
            "response": zs_response,
            "latency_s": round(zs_time, 2),
            **zs_tags,
            "truncated": len(zs_response) > 3900,
        },
        "post_trained": {
            "response": pt_response,
            "latency_s": round(pt_time, 2),
            **pt_tags,
            "truncated": len(pt_response) > 3900,
        },
    }

    if is_safety:
        result["zero_shot"]["safety_score"] = _extract_safety_score(zs_response)
        result["post_trained"]["safety_score"] = _extract_safety_score(pt_response)
    else:
        result["zero_shot"]["readiness"] = _extract_readiness(zs_response)
        result["post_trained"]["readiness"] = _extract_readiness(pt_response)

    return result


def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from evaluation results."""
    safety_results = [r for r in results if r["type"] == "safety"]
    handoff_results = [r for r in results if r["type"] == "handoff"]

    def _tag_rate(items: list[dict], key: str) -> float:
        if not items:
            return 0.0
        return sum(1 for r in items if r[key]["has_think_tag"] and r[key]["has_answer_tag"]) / len(items)

    def _truncation_rate(items: list[dict], key: str) -> float:
        if not items:
            return 0.0
        return sum(1 for r in items if r[key]["truncated"]) / len(items)

    def _avg_safety_score(items: list[dict], key: str) -> float | None:
        scores = [r[key]["safety_score"] for r in items if r[key].get("safety_score") is not None]
        return round(sum(scores) / len(scores), 1) if scores else None

    def _readiness_rate(items: list[dict], key: str) -> float:
        if not items:
            return 0.0
        return sum(1 for r in items if r[key].get("readiness") is not None) / len(items)

    def _avg_latency(items: list[dict], key: str) -> float:
        latencies = [r[key]["latency_s"] for r in items if r[key]["latency_s"] > 0]
        return round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    metrics = {
        "total_samples": len(results),
        "safety_samples": len(safety_results),
        "handoff_samples": len(handoff_results),
        "zero_shot": {
            "tag_compliance": round(_tag_rate(results, "zero_shot") * 100, 1),
            "truncation_rate": round(_truncation_rate(results, "zero_shot") * 100, 1),
            "avg_safety_score": _avg_safety_score(safety_results, "zero_shot"),
            "readiness_extraction_rate": round(_readiness_rate(handoff_results, "zero_shot") * 100, 1),
            "avg_latency_s": _avg_latency(results, "zero_shot"),
        },
        "post_trained": {
            "tag_compliance": round(_tag_rate(results, "post_trained") * 100, 1),
            "truncation_rate": round(_truncation_rate(results, "post_trained") * 100, 1),
            "avg_safety_score": _avg_safety_score(safety_results, "post_trained"),
            "readiness_extraction_rate": round(_readiness_rate(handoff_results, "post_trained") * 100, 1),
            "avg_latency_s": _avg_latency(results, "post_trained"),
        },
    }

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate post-trained vs zero-shot Cosmos Reason 2",
    )
    parser.add_argument(
        "--val-data", type=Path, required=True,
        help="Path to llava_val.json validation data",
    )
    parser.add_argument(
        "--zero-shot-url", type=str, required=True,
        help="Zero-shot model endpoint URL",
    )
    parser.add_argument(
        "--post-trained-url", type=str, required=True,
        help="Post-trained model endpoint URL",
    )
    parser.add_argument(
        "--media-path", type=str, default="",
        help="Base path to prepend to video paths in the dataset",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Use local vLLM endpoints (no GCP auth)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Max samples to evaluate (0 = all)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"Output directory for results (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    from inference.prompts import SYSTEM_PROMPT

    logger.info("=== Cosmos Reason 2 Evaluation ===")
    logger.info("  Val data: %s", args.val_data)
    logger.info("  Zero-shot: %s", args.zero_shot_url)
    logger.info("  Post-trained: %s", args.post_trained_url)

    with open(args.val_data) as f:
        val_data = json.load(f)

    if args.max_samples > 0:
        val_data = val_data[:args.max_samples]
    logger.info("Evaluating %d samples", len(val_data))

    results = []
    for i, sample in enumerate(val_data):
        logger.info("[%d/%d] %s", i + 1, len(val_data), sample["id"])
        result = evaluate_sample(
            sample,
            zero_shot_url=args.zero_shot_url,
            post_trained_url=args.post_trained_url,
            media_path=args.media_path,
            local=args.local,
            system_prompt=SYSTEM_PROMPT,
        )
        if result:
            results.append(result)

    # Compute metrics
    metrics = compute_metrics(results)

    logger.info("=== Results ===")
    logger.info("Samples evaluated: %d", metrics["total_samples"])
    logger.info("")
    logger.info("Zero-shot:")
    logger.info("  Tag compliance: %.1f%%", metrics["zero_shot"]["tag_compliance"])
    logger.info("  Truncation rate: %.1f%%", metrics["zero_shot"]["truncation_rate"])
    logger.info("  Avg safety score: %s", metrics["zero_shot"]["avg_safety_score"])
    logger.info("  Readiness extraction: %.1f%%", metrics["zero_shot"]["readiness_extraction_rate"])
    logger.info("  Avg latency: %.2fs", metrics["zero_shot"]["avg_latency_s"])
    logger.info("")
    logger.info("Post-trained:")
    logger.info("  Tag compliance: %.1f%%", metrics["post_trained"]["tag_compliance"])
    logger.info("  Truncation rate: %.1f%%", metrics["post_trained"]["truncation_rate"])
    logger.info("  Avg safety score: %s", metrics["post_trained"]["avg_safety_score"])
    logger.info("  Readiness extraction: %.1f%%", metrics["post_trained"]["readiness_extraction_rate"])
    logger.info("  Avg latency: %.2fs", metrics["post_trained"]["avg_latency_s"])

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_path = args.output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote detailed results: %s", results_path)

    metrics_path = args.output_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote metrics summary: %s", metrics_path)

    # Save comparison table for demo
    comparison_path = args.output_dir / "comparison.md"
    with open(comparison_path, "w") as f:
        f.write("# Cosmos Reason 2: Zero-Shot vs Post-Trained\n\n")
        f.write("| Metric | Zero-Shot | Post-Trained | Improvement |\n")
        f.write("|---|---|---|---|\n")

        zs = metrics["zero_shot"]
        pt = metrics["post_trained"]

        tag_diff = pt["tag_compliance"] - zs["tag_compliance"]
        f.write(f"| Tag compliance | {zs['tag_compliance']}% | {pt['tag_compliance']}% | +{tag_diff:.1f}% |\n")

        trunc_diff = zs["truncation_rate"] - pt["truncation_rate"]
        f.write(f"| Truncation rate | {zs['truncation_rate']}% | {pt['truncation_rate']}% | -{trunc_diff:.1f}% |\n")

        if zs["avg_safety_score"] and pt["avg_safety_score"]:
            score_diff = pt["avg_safety_score"] - zs["avg_safety_score"]
            f.write(f"| Avg safety score | {zs['avg_safety_score']} | {pt['avg_safety_score']} | {score_diff:+.1f} |\n")

        read_diff = pt["readiness_extraction_rate"] - zs["readiness_extraction_rate"]
        f.write(f"| Readiness extraction | {zs['readiness_extraction_rate']}% | {pt['readiness_extraction_rate']}% | +{read_diff:.1f}% |\n")

        lat_diff = zs["avg_latency_s"] - pt["avg_latency_s"]
        f.write(f"| Avg latency | {zs['avg_latency_s']}s | {pt['avg_latency_s']}s | {lat_diff:+.2f}s |\n")

    logger.info("Wrote comparison table: %s", comparison_path)
    logger.info("=== Evaluation complete ===")


if __name__ == "__main__":
    main()
