"""
run_vertex.py — Vertex AI Custom Job entrypoint for Grounding DINO detection.

Orchestrates:
1. Download synthetic clips from GCS
2. Extract frames from each clip (FFmpeg, 4 FPS)
3. Run Grounding DINO detection on all frames
4. Render BBox overlay annotations onto frames
5. Upload detection results + annotated frames to GCS (incrementally)

Supports batch splitting for parallel Vertex AI jobs and incremental
uploads for spot VM preemption resilience.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

from google.cloud import storage
from pydantic_settings import BaseSettings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Local working dirs inside the container
LOCAL_CLIPS = Path("/tmp/clips")
LOCAL_FRAMES = Path("/tmp/frames")
LOCAL_RESULTS = Path("/tmp/results")
LOCAL_ANNOTATED = Path("/tmp/annotated")

# Weights baked into Docker image
WEIGHTS_PATH = Path("/weights/groundingdino_swint_ogc.pth")

# Frame extraction rate
EXTRACT_FPS = 4


class DetectionConfig(BaseSettings):
    model_config = {"env_prefix": "DETECTION_"}

    gcs_bucket: str = "autolane-handoff-datagen"
    gcs_clips_prefix: str = "data/synthetic_clips"
    gcs_output_prefix: str = "data/detection"

    # Modes to process — image2world, text2world, or both
    mode: str = "both"

    # Confidence thresholds
    box_threshold: float = 0.35
    text_threshold: float = 0.25

    # Batch parallelism
    batch_index: int = 0
    total_batches: int = 1


def gcs_download_prefix(
    client: storage.Client, bucket_name: str, prefix: str, local_dir: Path,
    extension: str = "",
) -> list[Path]:
    """Download blobs under a GCS prefix. Returns list of local paths."""
    local_dir.mkdir(parents=True, exist_ok=True)
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    downloaded = []
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        relative = blob.name[len(prefix):].lstrip("/")
        if not relative:
            continue
        if extension and not relative.endswith(extension):
            continue
        local_path = local_dir / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            blob.download_to_filename(str(local_path))
            downloaded.append(local_path)
        except Exception:
            logger.exception("Failed to download %s", blob.name)
            local_path.unlink(missing_ok=True)
    return downloaded


def gcs_upload_file(
    client: storage.Client, bucket_name: str, local_path: Path, gcs_path: str
) -> None:
    """Upload a single file to GCS."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))


def gcs_blob_exists(
    client: storage.Client, bucket_name: str, gcs_path: str
) -> bool:
    """Check if a specific blob exists in GCS."""
    bucket = client.bucket(bucket_name)
    return bucket.blob(gcs_path).exists()


def extract_frames(clip_path: Path, output_dir: Path, fps: int = EXTRACT_FPS) -> list[Path]:
    """Extract frames from a video clip using FFmpeg.

    Returns list of extracted frame paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_stem = clip_path.stem

    # Output pattern: <clip_stem>_frame_%04d.jpg
    pattern = str(output_dir / f"{clip_stem}_frame_%04d.jpg")

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(clip_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",  # high quality JPEG
        pattern,
    ]

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("FFmpeg failed for %s: %s", clip_path.name, result.stderr)
        return []

    # Collect extracted frames
    frames = sorted(output_dir.glob(f"{clip_stem}_frame_*.jpg"))
    return frames


def load_detector(
    box_threshold: float,
    text_threshold: float,
) -> "GroundingDINODetector":
    """Instantiate and load the Grounding DINO detector once."""
    from detection.grounding_dino import GroundingDINODetector

    detector = GroundingDINODetector(
        variant="swint",
        weights_path=str(WEIGHTS_PATH),
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    detector.load()
    return detector


def run_detection_on_frames(
    detector: "GroundingDINODetector",
    frames: list[Path],
) -> list[dict]:
    """Run Grounding DINO on a list of frame paths.

    Returns list of FrameDetections as dicts.
    """
    from detection.prompts import DETECTION_PROMPT

    results = []
    for frame_path in frames:
        frame_det = detector.detect_image(
            str(frame_path),
            prompt=DETECTION_PROMPT,
        )
        results.append(frame_det.to_dict())

    return results


def render_overlays(
    frames: list[Path],
    detection_results: list[dict],
    output_dir: Path,
) -> list[Path]:
    """Render BBox overlays onto frames. Returns annotated frame paths."""
    from detection.grounding_dino import Detection
    from detection.overlay import render_frame_detections
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_paths = []

    for frame_path, result_dict in zip(frames, detection_results, strict=True):
        detections = [Detection(**d) for d in result_dict["detections"]]

        frame = cv2.imread(str(frame_path))
        if frame is None:
            logger.warning("Could not read frame: %s", frame_path)
            continue

        annotated = render_frame_detections(frame, detections)
        out_path = output_dir / f"annotated_{frame_path.name}"
        cv2.imwrite(str(out_path), annotated)
        annotated_paths.append(out_path)

    return annotated_paths


def process_clip(
    client: storage.Client,
    config: DetectionConfig,
    detector: "GroundingDINODetector",
    clip_path: Path,
    mode: str,
) -> bool:
    """Process a single clip: extract frames, detect, render, upload.

    Returns True if successful.
    """
    clip_name = clip_path.stem
    gcs_results_path = f"{config.gcs_output_prefix}/{mode}/{clip_name}/detections.json"

    # Skip if already processed
    if gcs_blob_exists(client, config.gcs_bucket, gcs_results_path):
        logger.info("  Skipping (already in GCS): %s", clip_name)
        return True

    # 1. Extract frames
    clip_frames_dir = LOCAL_FRAMES / clip_name
    frames = extract_frames(clip_path, clip_frames_dir)
    if not frames:
        logger.error("  No frames extracted from %s", clip_name)
        return False

    logger.info("  Extracted %d frames from %s", len(frames), clip_name)

    # 2. Run detection
    detection_results = run_detection_on_frames(detector, frames)
    total_dets = sum(len(r["detections"]) for r in detection_results)
    logger.info("  %d detections across %d frames", total_dets, len(frames))

    # 3. Render overlays
    clip_annotated_dir = LOCAL_ANNOTATED / clip_name
    annotated_paths = render_overlays(frames, detection_results, clip_annotated_dir)
    logger.info("  Rendered %d annotated frames", len(annotated_paths))

    # 4. Upload results to GCS incrementally
    # Upload detection JSON
    results_local = LOCAL_RESULTS / mode / f"{clip_name}.json"
    results_local.parent.mkdir(parents=True, exist_ok=True)
    with open(results_local, "w") as f:
        json.dump(detection_results, f, indent=2)
    gcs_upload_file(client, config.gcs_bucket, results_local, gcs_results_path)

    # Upload annotated frames
    for ann_path in annotated_paths:
        gcs_ann_path = f"{config.gcs_output_prefix}/{mode}/{clip_name}/{ann_path.name}"
        gcs_upload_file(client, config.gcs_bucket, ann_path, gcs_ann_path)

    # Clean up local files to save disk
    for f in frames:
        f.unlink(missing_ok=True)
    for f in annotated_paths:
        f.unlink(missing_ok=True)
    results_local.unlink(missing_ok=True)

    return True


def main() -> None:
    config = DetectionConfig()
    logger.info("=== Autolane Detection — Vertex AI Job ===")
    logger.info("  Mode: %s", config.mode)
    logger.info("  GCS bucket: %s", config.gcs_bucket)
    logger.info("  Box threshold: %.2f", config.box_threshold)
    logger.info("  Text threshold: %.2f", config.text_threshold)
    if config.total_batches > 1:
        logger.info("  Batch: %d/%d", config.batch_index + 1, config.total_batches)

    client = storage.Client()

    # Determine which modes to process
    modes = []
    if config.mode in ("image2world", "both"):
        modes.append("image2world")
    if config.mode in ("text2world", "both"):
        modes.append("text2world")

    all_clips: list[tuple[Path, str]] = []  # (local_path, mode)

    for mode in modes:
        # Download clips from GCS
        prefix = f"{config.gcs_clips_prefix}/{mode}"
        logger.info("[1/3] Downloading %s clips from GCS...", mode)
        clip_paths = gcs_download_prefix(
            client, config.gcs_bucket, prefix,
            LOCAL_CLIPS / mode, extension=".mp4",
        )
        logger.info("  Downloaded %d clips", len(clip_paths))
        for cp in sorted(clip_paths):
            all_clips.append((cp, mode))

    # Apply batch slicing
    if config.total_batches > 1:
        batch_size = len(all_clips) // config.total_batches
        remainder = len(all_clips) % config.total_batches
        start = config.batch_index * batch_size + min(config.batch_index, remainder)
        end = start + batch_size + (1 if config.batch_index < remainder else 0)
        all_clips = all_clips[start:end]
        logger.info(
            "Batch %d/%d: processing %d clips",
            config.batch_index + 1, config.total_batches, len(all_clips),
        )

    logger.info("[2/3] Processing %d clips...", len(all_clips))

    # Load model once, then process all clips
    detector = load_detector(config.box_threshold, config.text_threshold)

    generated = 0
    skipped = 0
    failed = 0

    for idx, (clip_path, mode) in enumerate(all_clips, 1):
        clip_name = clip_path.stem
        gcs_results_path = f"{config.gcs_output_prefix}/{mode}/{clip_name}/detections.json"

        # Skip if already processed
        if gcs_blob_exists(client, config.gcs_bucket, gcs_results_path):
            logger.info("[%d/%d] Skipping (already in GCS): %s", idx, len(all_clips), clip_name)
            skipped += 1
            continue

        logger.info("[%d/%d] Processing: %s (%s)", idx, len(all_clips), clip_name, mode)

        try:
            success = process_clip(client, config, detector, clip_path, mode)
            if success:
                generated += 1
                logger.info("[%d/%d] Done + uploaded: %s", idx, len(all_clips), clip_name)
            else:
                failed += 1
        except Exception:
            logger.exception("[%d/%d] Error processing %s", idx, len(all_clips), clip_name)
            failed += 1

    logger.info("")
    logger.info("[3/3] === Detection Summary ===")
    logger.info("  Processed: %d clips", generated)
    if skipped:
        logger.info("  Skipped (already in GCS): %d clips", skipped)
    if failed:
        logger.warning("  Failed: %d clips", failed)
    logger.info("  Total: %d clips", generated + skipped + failed)

    if generated == 0 and skipped == 0:
        logger.error("No clips processed successfully")
        sys.exit(1)

    logger.info("=== Detection job complete ===")


if __name__ == "__main__":
    main()
