#!/usr/bin/env python3
"""Bootstrap QA annotations using Cosmos Reason 2-8B zero-shot inference.

Downloads annotated frames from GCS (Grounding DINO overlays), reassembles
them into annotated videos, runs two-pass inference (Safety + Handoff) via
the deployed Vertex AI endpoint, and saves draft QA pairs for human review.

Pipeline step 1 of 3 (see data/README.md).

Usage:
    # Dry run — list clips and check which are already done
    python -m data.bootstrap_annotations --dry-run

    # Process all clips
    python -m data.bootstrap_annotations

    # Process only image2world, limit to 10 clips
    python -m data.bootstrap_annotations --mode image2world --max-clips 10

    # Run as batch 1 of 4 (for parallelism via separate processes)
    python -m data.bootstrap_annotations --batch-index 0 --total-batches 4

    # Download results from GCS
    python -m data.bootstrap_annotations --download
"""

import argparse
import asyncio
import base64
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
from google.auth import default as gcp_default
from google.auth.transport.requests import Request
from google.cloud import storage
from pydantic_settings import BaseSettings, SettingsConfigDict

from inference.prompts import (
    HANDOFF_PROMPT,
    SAFETY_PROMPT,
    SYSTEM_PROMPT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_GCP_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Local working directories
LOCAL_WORK = Path("/tmp/bootstrap")
LOCAL_FRAMES = LOCAL_WORK / "frames"
LOCAL_VIDEOS = LOCAL_WORK / "videos"

# Frame rate for annotated videos (must match training setup)
ANNOTATED_FPS = 4

# Max frames to send per inference request (context window budget).
# At ~900 tokens/image, 8 frames ≈ 7200 image tokens + prompt ≈ 8500 total,
# well within the 16384 max_model_len.
MAX_INFERENCE_FRAMES = 8


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class BootstrapConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # GCP
    gcp_project_id: str = "autolane-handoff-20260221"
    gcp_region: str = "us-central1"

    # Vertex AI endpoint
    vertex_endpoint_id: str = ""
    model_name: str = "nvidia/Cosmos-Reason2-8B"

    # GCS
    datagen_gcs_bucket: str = "autolane-handoff-datagen"

    # Inference
    temperature: float = 0.2
    max_tokens: int = 4096


# GCS prefixes
GCS_DETECTION_PREFIX = "data/detection"
GCS_ANNOTATIONS_PREFIX = "data/annotations"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_think_answer(raw: str) -> tuple[str, str]:
    """Extract <think> and <answer> blocks from model output."""
    think = ""
    answer = ""
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if think_match:
        think = think_match.group(1).strip()
    answer_match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    return think, answer


def assess_quality(safety_raw: str, handoff_raw: str) -> dict:
    """Flag quality issues in the bootstrap responses."""
    s_think, s_answer = parse_think_answer(safety_raw)
    h_think, h_answer = parse_think_answer(handoff_raw)

    safety_score_found = bool(
        re.search(r"safety_score.*?(\d+)", s_answer, re.IGNORECASE)
    )
    readiness_found = bool(
        re.search(r"readiness.*?(READY|NOT_READY|CAUTION)", h_answer, re.IGNORECASE)
    )

    needs_review = (
        not s_think or not s_answer
        or not h_think or not h_answer
        or not safety_score_found
        or not readiness_found
    )

    return {
        "safety_has_think": bool(s_think),
        "safety_has_answer": bool(s_answer),
        "safety_score_found": safety_score_found,
        "handoff_has_think": bool(h_think),
        "handoff_has_answer": bool(h_answer),
        "readiness_found": readiness_found,
        "needs_review": needs_review,
    }


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def gcs_blob_exists(
    client: storage.Client, bucket_name: str, gcs_path: str,
) -> bool:
    bucket = client.bucket(bucket_name)
    return bucket.blob(gcs_path).exists()


def gcs_upload_string(
    client: storage.Client, bucket_name: str, gcs_path: str,
    content: str, content_type: str = "application/json",
) -> None:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(content, content_type=content_type)


def gcs_upload_file(
    client: storage.Client, bucket_name: str, gcs_path: str,
    local_path: Path,
) -> None:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))


# ---------------------------------------------------------------------------
# Bootstrap annotator
# ---------------------------------------------------------------------------

class BootstrapAnnotator:
    """Generates zero-shot QA pairs using Cosmos Reason 2-8B."""

    def __init__(
        self,
        config: BootstrapConfig,
        request_timeout: float = 300.0,
    ):
        self.config = config
        self.request_timeout = request_timeout
        self.gcs = storage.Client()
        self._token: str | None = None
        self._token_expiry: float = 0

    # ---- Auth ----

    def _refresh_token(self) -> str:
        """Get a fresh OAuth2 token, cached for 50 minutes."""
        now = time.time()
        if self._token and now < self._token_expiry:
            return self._token
        credentials, _ = gcp_default(scopes=_GCP_SCOPES)
        credentials.refresh(Request())
        self._token = credentials.token
        self._token_expiry = now + 3000
        return self._token

    def _endpoint_url(self) -> str:
        return (
            f"https://{self.config.gcp_region}-aiplatform.googleapis.com/v1/"
            f"projects/{self.config.gcp_project_id}/"
            f"locations/{self.config.gcp_region}/"
            f"endpoints/{self.config.vertex_endpoint_id}:rawPredict"
        )

    # ---- GCS discovery ----

    def list_clips(self, mode: str) -> list[str]:
        """List clip folder names from GCS detection output."""
        bucket = self.gcs.bucket(self.config.datagen_gcs_bucket)
        prefix = f"{GCS_DETECTION_PREFIX}/{mode}/"
        iterator = bucket.list_blobs(prefix=prefix, delimiter="/")
        list(iterator)  # consume to populate prefixes
        clips = []
        for folder in iterator.prefixes:
            clip_name = folder.rstrip("/").split("/")[-1]
            clips.append(clip_name)
        return sorted(clips)

    def clip_already_processed(self, clip_name: str, mode: str) -> bool:
        gcs_path = f"{GCS_ANNOTATIONS_PREFIX}/{mode}/{clip_name}/annotation.json"
        return gcs_blob_exists(self.gcs, self.config.datagen_gcs_bucket, gcs_path)

    # ---- Download annotated frames ----

    def download_annotated_frames(self, clip_name: str, mode: str) -> list[Path]:
        """Download annotated frames for a clip from GCS."""
        local_dir = LOCAL_FRAMES / mode / clip_name
        local_dir.mkdir(parents=True, exist_ok=True)

        bucket = self.gcs.bucket(self.config.datagen_gcs_bucket)
        prefix = f"{GCS_DETECTION_PREFIX}/{mode}/{clip_name}/"
        blobs = list(bucket.list_blobs(prefix=prefix))

        frames = []
        for blob in blobs:
            name = blob.name.split("/")[-1]
            if name.startswith("annotated_") and name.endswith(".jpg"):
                local_path = local_dir / name
                try:
                    blob.download_to_filename(str(local_path))
                    frames.append(local_path)
                except Exception:
                    logger.exception("Failed to download %s", blob.name)
                    local_path.unlink(missing_ok=True)

        return sorted(frames)

    # ---- Video reassembly ----

    def reassemble_video(
        self, frames: list[Path], clip_name: str, mode: str,
    ) -> Path | None:
        """Reassemble annotated frames into an MP4 video using FFmpeg."""
        out_dir = LOCAL_VIDEOS / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{clip_name}.mp4"

        # FFmpeg concat demuxer file list
        list_file = out_dir / f"{clip_name}_list.txt"
        with open(list_file, "w") as f:
            for frame in frames:
                f.write(f"file '{frame}'\n")
                f.write(f"duration {1.0 / ANNOTATED_FPS}\n")

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", str(ANNOTATED_FPS),
            str(out_path),
        ]

        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        list_file.unlink(missing_ok=True)

        if result.returncode != 0:
            logger.error("FFmpeg failed for %s: %s", clip_name, result.stderr)
            return None

        return out_path

    # ---- Inference ----

    @staticmethod
    def _sample_frames(frames: list[Path], max_frames: int) -> list[Path]:
        """Evenly sample frames to fit within context window budget."""
        if len(frames) <= max_frames:
            return frames
        step = len(frames) / max_frames
        return [frames[int(i * step)] for i in range(max_frames)]

    @staticmethod
    def _encode_frames_b64(
        frames: list[Path], max_width: int = 640, jpeg_quality: int = 50,
    ) -> list[str]:
        """Resize and base64-encode frames to fit rawPredict 1.5MB limit.

        Vertex AI rawPredict has a 1.5MB request size limit. Original
        annotated frames are ~280KB each at 1280x704. Resizing to 640px
        wide and compressing to quality=50 gives ~30-50KB per frame,
        allowing 8 frames comfortably under the limit.
        """
        from io import BytesIO
        from PIL import Image

        encoded = []
        for frame in frames:
            img = Image.open(frame)
            # Resize if wider than max_width, preserving aspect ratio
            if img.width > max_width:
                ratio = max_width / img.width
                new_size = (max_width, int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality)
            encoded.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        return encoded

    def _build_messages(
        self, frame_b64s: list[str], prompt: str,
    ) -> list[dict]:
        """Build vLLM chat messages with base64-encoded image frames."""
        content: list[dict] = []
        for b64 in frame_b64s:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        content.append({"type": "text", "text": prompt})

        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": content,
            },
        ]

    async def _call_endpoint(
        self, frame_b64s: list[str], prompt: str,
    ) -> str:
        """Send a single inference request to the Vertex AI endpoint."""
        token = await asyncio.get_event_loop().run_in_executor(
            None, self._refresh_token,
        )

        payload = {
            "model": self.config.model_name,
            "messages": self._build_messages(frame_b64s, prompt),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            resp = await client.post(
                self._endpoint_url(), json=payload, headers=headers,
            )
            if resp.status_code != 200:
                logger.error(
                    "Endpoint returned %d: %s", resp.status_code, resp.text[:500],
                )
                resp.raise_for_status()

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError(f"Model returned no choices: {data}")
        return choices[0].get("message", {}).get("content", "")

    # ---- Per-clip processing ----

    async def process_clip(self, clip_name: str, mode: str) -> dict | None:
        """Process one clip end-to-end: download → reassemble → infer → upload.

        Returns annotation dict or None on failure.
        """
        try:
            # 1. Download annotated frames
            frames = await asyncio.get_event_loop().run_in_executor(
                None, self.download_annotated_frames, clip_name, mode,
            )
            if not frames:
                logger.warning("  No annotated frames for %s", clip_name)
                return None

            # 2. Reassemble into annotated video (for SFT training use)
            video_path = await asyncio.get_event_loop().run_in_executor(
                None, self.reassemble_video, frames, clip_name, mode,
            )
            if not video_path:
                return None

            # 3. Sample frames for inference and base64-encode
            sampled = self._sample_frames(frames, MAX_INFERENCE_FRAMES)
            frame_b64s = self._encode_frames_b64(sampled)

            video_mb = video_path.stat().st_size / (1024 * 1024)
            logger.info(
                "  Prepared: %d frames (%d sampled), %.1f MB video",
                len(frames), len(sampled), video_mb,
            )

            # 4. Pass 1 — Safety + Compliance
            t0 = time.time()
            safety_raw = await self._call_endpoint(frame_b64s, SAFETY_PROMPT)
            safety_ms = (time.time() - t0) * 1000
            logger.info("  Safety pass: %.1fs", safety_ms / 1000)

            # 5. Pass 2 — Handoff Planning
            t0 = time.time()
            handoff_raw = await self._call_endpoint(frame_b64s, HANDOFF_PROMPT)
            handoff_ms = (time.time() - t0) * 1000
            logger.info("  Handoff pass: %.1fs", handoff_ms / 1000)

            # 6. Parse and assess quality
            s_think, s_answer = parse_think_answer(safety_raw)
            h_think, h_answer = parse_think_answer(handoff_raw)
            quality = assess_quality(safety_raw, handoff_raw)

            # 7. Upload annotated video to GCS (for SFT training)
            video_gcs = f"{GCS_ANNOTATIONS_PREFIX}/{mode}/{clip_name}/annotated.mp4"
            await asyncio.get_event_loop().run_in_executor(
                None, gcs_upload_file,
                self.gcs, self.config.datagen_gcs_bucket, video_gcs, video_path,
            )

            # 8. Build annotation record
            annotation = {
                "clip_name": clip_name,
                "mode": mode,
                "annotated_video_gcs": video_gcs,
                "num_frames": len(frames),
                "inference_frames": len(sampled),
                "safety": {
                    "raw_response": safety_raw,
                    "think": s_think,
                    "answer": s_answer,
                    "processing_time_ms": round(safety_ms),
                },
                "handoff": {
                    "raw_response": handoff_raw,
                    "think": h_think,
                    "answer": h_answer,
                    "processing_time_ms": round(handoff_ms),
                },
                "quality": quality,
            }

            # 9. Upload annotation JSON to GCS (this marks the clip as done)
            ann_gcs = f"{GCS_ANNOTATIONS_PREFIX}/{mode}/{clip_name}/annotation.json"
            await asyncio.get_event_loop().run_in_executor(
                None, gcs_upload_string,
                self.gcs, self.config.datagen_gcs_bucket, ann_gcs,
                json.dumps(annotation, indent=2),
            )

            # 10. Clean up local files
            for frame in frames:
                frame.unlink(missing_ok=True)
            video_path.unlink(missing_ok=True)

            return annotation

        except Exception:
            logger.exception("Failed to process %s/%s", mode, clip_name)
            return None

    # ---- Main orchestration ----

    async def run(
        self,
        mode: str = "both",
        max_clips: int = 0,
        batch_index: int = 0,
        total_batches: int = 1,
        dry_run: bool = False,
    ) -> list[dict]:
        """Run bootstrap annotation on clips.

        Args:
            mode: image2world | text2world | both
            max_clips: Limit to N clips (0 = all)
            batch_index: Batch index for parallel slicing (0-based)
            total_batches: Total number of parallel batches
            dry_run: List clips without processing

        Returns:
            List of annotation dicts.
        """
        modes = []
        if mode in ("image2world", "both"):
            modes.append("image2world")
        if mode in ("text2world", "both"):
            modes.append("text2world")

        # Discover all clips
        all_clips: list[tuple[str, str]] = []
        for m in modes:
            clips = self.list_clips(m)
            logger.info("Found %d clips for %s", len(clips), m)
            for c in clips:
                all_clips.append((c, m))

        # Apply batch slicing
        if total_batches > 1:
            batch_size = len(all_clips) // total_batches
            remainder = len(all_clips) % total_batches
            start = batch_index * batch_size + min(batch_index, remainder)
            end = start + batch_size + (1 if batch_index < remainder else 0)
            all_clips = all_clips[start:end]
            logger.info(
                "Batch %d/%d: %d clips",
                batch_index + 1, total_batches, len(all_clips),
            )

        if max_clips > 0:
            all_clips = all_clips[:max_clips]

        logger.info("Total clips to process: %d", len(all_clips))

        if dry_run:
            for clip_name, m in all_clips:
                done = self.clip_already_processed(clip_name, m)
                status = " (done)" if done else ""
                print(f"  {m}/{clip_name}{status}")
            done_count = sum(
                1 for c, m in all_clips if self.clip_already_processed(c, m)
            )
            print(f"\n  {done_count}/{len(all_clips)} already processed")
            return []

        # Filter out already-processed clips
        pending: list[tuple[str, str]] = []
        skipped = 0
        for clip_name, m in all_clips:
            if self.clip_already_processed(clip_name, m):
                skipped += 1
            else:
                pending.append((clip_name, m))

        if skipped:
            logger.info("Skipping %d already-processed clips", skipped)
        logger.info("Processing %d clips sequentially", len(pending))

        # Process clips one at a time (endpoint handles one video at a time)
        results: list[dict] = []
        failed = 0

        for idx, (clip_name, m) in enumerate(pending, 1):
            logger.info("[%d/%d] %s/%s", idx, len(pending), m, clip_name)

            annotation = await self.process_clip(clip_name, m)
            if annotation:
                results.append(annotation)
                flag = " [NEEDS REVIEW]" if annotation["quality"]["needs_review"] else ""
                logger.info(
                    "[%d/%d] Done: %s%s", idx, len(pending), clip_name, flag,
                )
            else:
                failed += 1
                logger.warning("[%d/%d] FAILED: %s", idx, len(pending), clip_name)

        # Summary
        total = len(results) + failed + skipped
        needs_review = sum(1 for r in results if r["quality"]["needs_review"])
        logger.info("")
        logger.info("=== Bootstrap Summary ===")
        logger.info("  Processed:  %d clips", len(results))
        logger.info("  Skipped:    %d clips (already in GCS)", skipped)
        if failed:
            logger.warning("  Failed:     %d clips", failed)
        logger.info("  Total:      %d clips", total)
        logger.info("  QA pairs:   %d (2 per clip)", len(results) * 2)
        logger.info("  Need review: %d / %d", needs_review, len(results))

        return results


# ---------------------------------------------------------------------------
# Download action
# ---------------------------------------------------------------------------

def download_results(gcs_bucket: str) -> None:
    """Download annotation results from GCS to data/annotations/."""
    local_dir = Path("data/annotations")
    local_dir.mkdir(parents=True, exist_ok=True)

    for mode in ("image2world", "text2world"):
        mode_dir = local_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading %s annotations...", mode)
        result = subprocess.run(
            [
                "gsutil", "-m", "rsync", "-r",
                f"gs://{gcs_bucket}/{GCS_ANNOTATIONS_PREFIX}/{mode}/",
                str(mode_dir) + "/",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            if "No URLs matched" in result.stderr:
                logger.info("  (none found)")
            else:
                logger.error("  Error: %s", result.stderr)

    logger.info("Done. Results in %s", local_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap QA annotations using Cosmos Reason 2-8B",
    )
    parser.add_argument(
        "--mode", default="both",
        choices=["image2world", "text2world", "both"],
        help="Which clip modes to process (default: both)",
    )
    parser.add_argument(
        "--max-clips", type=int, default=0,
        help="Limit to N clips, 0 = all (default: 0)",
    )
    parser.add_argument(
        "--batch-index", type=int, default=0,
        help="Batch index for parallel slicing, 0-based (default: 0)",
    )
    parser.add_argument(
        "--total-batches", type=int, default=1,
        help="Total number of parallel batches (default: 1)",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Override sampling temperature",
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="Per-request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List clips without processing",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download results from GCS instead of running inference",
    )
    args = parser.parse_args()

    config = BootstrapConfig()

    if args.download:
        download_results(config.datagen_gcs_bucket)
        return

    if not config.vertex_endpoint_id:
        logger.error(
            "VERTEX_ENDPOINT_ID not set. Deploy model first or check .env"
        )
        sys.exit(1)

    if args.temperature is not None:
        config.temperature = args.temperature

    logger.info("=== Autolane Bootstrap Annotations ===")
    logger.info("  Endpoint: %s", config.vertex_endpoint_id)
    logger.info("  Model: %s", config.model_name)
    logger.info("  Bucket: %s", config.datagen_gcs_bucket)
    logger.info("  Temperature: %.2f", config.temperature)

    annotator = BootstrapAnnotator(
        config=config,
        request_timeout=args.timeout,
    )

    results = asyncio.run(
        annotator.run(
            mode=args.mode,
            max_clips=args.max_clips,
            batch_index=args.batch_index,
            total_batches=args.total_batches,
            dry_run=args.dry_run,
        )
    )

    if results:
        output_path = Path("data/annotations/bootstrap_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved %d results to %s", len(results), output_path)


if __name__ == "__main__":
    main()
