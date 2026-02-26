"""
run_vertex.py — Vertex AI Custom Job entrypoint for Cosmos Predict 2.5 datagen.

Orchestrates:
1. Download site photos + prompts from GCS to /tmp/
2. Restore HF model cache from GCS (if available)
3. Generate NVIDIA-format JSON param files from our prompts
4. Run cosmos-predict2.5 inference via their native CLI
5. Upload output clips to GCS
6. Save model cache to GCS (first run only)

Uses NVIDIA's native cosmos-predict2.5 CLI rather than HuggingFace diffusers.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from google.cloud import storage

from datagen.config import DatagenConfig, DatagenMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Local working dirs inside the container
LOCAL_SITE_PHOTOS = Path("/tmp/site_photos")
LOCAL_PROMPTS = Path("/tmp/prompts")
LOCAL_OUTPUT = Path("/tmp/output")
LOCAL_PARAMS = Path("/tmp/params")
LOCAL_HF_CACHE = Path(os.environ.get("HF_HOME", "/tmp/hf_cache"))

# NVIDIA's inference script location (cloned in Docker build)
NVIDIA_INFERENCE_SCRIPT = Path("/workspace/examples/inference.py")


def gcs_download_prefix(
    client: storage.Client, bucket_name: str, prefix: str, local_dir: Path
) -> int:
    """Download all blobs under a GCS prefix to a local directory."""
    local_dir.mkdir(parents=True, exist_ok=True)
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    downloaded = 0
    for blob in blobs:
        # Skip "directory" markers
        if blob.name.endswith("/"):
            continue
        relative = blob.name[len(prefix) :].lstrip("/")
        if not relative:
            continue
        local_path = local_dir / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        downloaded += 1
    return downloaded


def gcs_upload_dir(
    client: storage.Client, bucket_name: str, local_dir: Path, gcs_prefix: str
) -> int:
    """Upload all files in a local directory to a GCS prefix."""
    bucket = client.bucket(bucket_name)
    uploaded = 0
    for local_path in local_dir.rglob("*"):
        if local_path.is_dir():
            continue
        relative = local_path.relative_to(local_dir)
        blob_name = f"{gcs_prefix}/{relative}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        uploaded += 1
    return uploaded


def gcs_prefix_exists(
    client: storage.Client, bucket_name: str, prefix: str
) -> bool:
    """Check if any blobs exist under a GCS prefix."""
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix, max_results=1))
    return len(blobs) > 0


def restore_model_cache(client: storage.Client, config: DatagenConfig) -> bool:
    """Restore HF model cache from GCS if available."""
    if not config.use_model_cache:
        logger.info("Model cache disabled, skipping restore")
        return False

    cache_prefix = config.gcs_model_cache_prefix
    if not gcs_prefix_exists(client, config.gcs_bucket, cache_prefix):
        logger.info("No model cache found in GCS at gs://%s/%s", config.gcs_bucket, cache_prefix)
        return False

    logger.info("Restoring model cache from gs://%s/%s ...", config.gcs_bucket, cache_prefix)
    count = gcs_download_prefix(client, config.gcs_bucket, cache_prefix, LOCAL_HF_CACHE)
    logger.info("Restored %d cache files", count)
    return True


def save_model_cache(client: storage.Client, config: DatagenConfig) -> None:
    """Save HF model cache to GCS for subsequent runs."""
    if not config.use_model_cache:
        return

    cache_prefix = config.gcs_model_cache_prefix
    if gcs_prefix_exists(client, config.gcs_bucket, cache_prefix):
        logger.info("Model cache already exists in GCS, skipping save")
        return

    logger.info("Saving model cache to gs://%s/%s ...", config.gcs_bucket, cache_prefix)
    count = gcs_upload_dir(client, config.gcs_bucket, LOCAL_HF_CACHE, cache_prefix)
    logger.info("Saved %d cache files to GCS", count)


def generate_param_files(config: DatagenConfig) -> list[Path]:
    """Generate NVIDIA-format JSON param files from our prompt definitions.

    NVIDIA's CLI expects JSON files with:
        {"inference_type": "image2world", "name": "clip_name", "prompt": "...", "input_path": "photo.jpg"}
    or for text2world:
        {"inference_type": "text2world", "name": "clip_name", "prompt": "..."}
    """
    LOCAL_PARAMS.mkdir(parents=True, exist_ok=True)
    param_files: list[Path] = []

    if config.mode == DatagenMode.image2world:
        prompts_file = LOCAL_PROMPTS / "image2world_base.json"
        with open(prompts_file) as f:
            prompts = json.load(f)

        # Filter by prompt IDs if specified
        if config.prompt_ids:
            ids = config.prompt_ids.split(",")
            prompts = [p for p in prompts if p["id"] in ids]

        # Collect site photos
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        photos = sorted(
            p for p in LOCAL_SITE_PHOTOS.iterdir()
            if p.suffix.lower() in extensions
        )
        logger.info("Found %d photos, %d prompts -> %d param files",
                     len(photos), len(prompts), len(photos) * len(prompts))

        for photo in photos:
            for prompt_data in prompts:
                name = f"{photo.stem}_{prompt_data['id']}"
                param = {
                    "inference_type": "image2world",
                    "name": name,
                    "prompt": prompt_data["prompt"],
                    "input_path": str(photo),
                    "num_steps": config.num_inference_steps,
                    "num_output_frames": config.num_output_frames,
                    "seed": config.seed,
                }
                param_file = LOCAL_PARAMS / f"{name}.json"
                with open(param_file, "w") as f:
                    json.dump(param, f, indent=2)
                param_files.append(param_file)

    else:  # text2world
        prompts_file = LOCAL_PROMPTS / "text2world_base.json"
        with open(prompts_file) as f:
            prompts = json.load(f)

        if config.prompt_ids:
            ids = config.prompt_ids.split(",")
            prompts = [p for p in prompts if p["id"] in ids]

        logger.info("Found %d prompts, %d seeds -> %d param files",
                     len(prompts), config.num_seeds, len(prompts) * config.num_seeds)

        for prompt_data in prompts:
            for seed_idx in range(config.num_seeds):
                seed = config.seed + seed_idx
                suffix = f"_s{seed}" if config.num_seeds > 1 else ""
                name = f"{prompt_data['id']}{suffix}"
                param = {
                    "inference_type": "text2world",
                    "name": name,
                    "prompt": prompt_data["prompt"],
                    "num_steps": config.num_inference_steps,
                    "num_output_frames": config.num_output_frames,
                    "seed": seed,
                }
                param_file = LOCAL_PARAMS / f"{name}.json"
                with open(param_file, "w") as f:
                    json.dump(param, f, indent=2)
                param_files.append(param_file)

    return param_files


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


def run_generation(config: DatagenConfig, client: storage.Client) -> None:
    """Generate param files and run NVIDIA's inference CLI one at a time.

    NVIDIA's inference.py only processes a single -i param file per invocation.
    Passing multiple -i flags causes only the last one to be used. We loop over
    param files and invoke the CLI once per clip.

    Each clip is uploaded to GCS immediately after generation so that progress
    survives spot VM preemption. Clips already in GCS are skipped on restart.
    """
    LOCAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    output_gcs_prefix = f"{config.gcs_output_prefix}/{config.mode.value}"

    all_param_files = generate_param_files(config)
    if not all_param_files:
        logger.error("No param files generated — check prompts and filters")
        sys.exit(1)

    # Slice param files for this batch
    if config.total_batches > 1:
        batch_size = len(all_param_files) // config.total_batches
        remainder = len(all_param_files) % config.total_batches
        # Distribute remainder across first N batches
        start = config.batch_index * batch_size + min(config.batch_index, remainder)
        end = start + batch_size + (1 if config.batch_index < remainder else 0)
        param_files = all_param_files[start:end]
        logger.info(
            "Batch %d/%d: processing %d of %d total param files (indices %d-%d)",
            config.batch_index + 1, config.total_batches,
            len(param_files), len(all_param_files), start, end - 1,
        )
    else:
        param_files = all_param_files

    logger.info("Running inference on %d param files, one at a time...", len(param_files))

    generated = 0
    skipped = 0
    failed = 0

    for idx, pf in enumerate(param_files, 1):
        clip_name = pf.stem
        output_mp4 = LOCAL_OUTPUT / f"{clip_name}.mp4"
        gcs_mp4_path = f"{output_gcs_prefix}/{clip_name}.mp4"

        # Skip if already uploaded to GCS (survives preemption restarts)
        if gcs_blob_exists(client, config.gcs_bucket, gcs_mp4_path):
            logger.info("[%d/%d] Skipping (already in GCS): %s", idx, len(param_files), clip_name)
            skipped += 1
            continue

        logger.info("[%d/%d] Generating: %s", idx, len(param_files), clip_name)

        cmd = [
            sys.executable, str(NVIDIA_INFERENCE_SCRIPT),
            "-i", str(pf),
            "-o", str(LOCAL_OUTPUT),
            "--model", config.model_key,
            "--offload-text-encoder", "--offload-tokenizer",
        ]

        if config.disable_guardrails:
            cmd.append("--disable-guardrails")

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            logger.error("[%d/%d] Failed: %s (exit code %d)", idx, len(param_files), clip_name, result.returncode)
            failed += 1
            continue

        # Upload clip to GCS immediately after generation
        if output_mp4.exists():
            gcs_upload_file(client, config.gcs_bucket, output_mp4, gcs_mp4_path)

            # Also upload the param JSON as metadata
            gcs_json_path = f"{output_gcs_prefix}/{clip_name}.json"
            gcs_upload_file(client, config.gcs_bucket, pf, gcs_json_path)

            logger.info("[%d/%d] Done + uploaded: %s", idx, len(param_files), clip_name)
            generated += 1

            # Clean up local file to save disk space
            output_mp4.unlink()
        else:
            logger.error("[%d/%d] No output file found: %s", idx, len(param_files), clip_name)
            failed += 1

    logger.info("")
    logger.info("=== Generation Summary ===")
    logger.info("  Generated: %d/%d clips", generated, len(param_files))
    if skipped:
        logger.info("  Skipped (already in GCS): %d clips", skipped)
    if failed:
        logger.warning("  Failed: %d clips", failed)

    if generated == 0 and skipped == 0:
        logger.error("No clips generated successfully")
        sys.exit(1)


def main() -> None:
    config = DatagenConfig()
    logger.info("=== Autolane Datagen — Vertex AI Job (NVIDIA native) ===")
    logger.info("  Mode: %s", config.mode.value)
    logger.info("  Model: %s", config.model_key)
    logger.info("  Steps: %d", config.num_inference_steps)
    logger.info("  Frames: %d", config.num_output_frames)
    logger.info("  GCS bucket: %s", config.gcs_bucket)
    logger.info("  Prompt IDs filter: %s", config.prompt_ids or "(all)")

    client = storage.Client()

    # Step 1: Download inputs from GCS
    logger.info("[1/5] Downloading site photos from GCS...")
    photo_count = gcs_download_prefix(
        client, config.gcs_bucket, config.gcs_site_photos_prefix, LOCAL_SITE_PHOTOS
    )
    logger.info("  Downloaded %d site photos", photo_count)

    logger.info("[2/5] Downloading prompts from GCS...")
    prompt_count = gcs_download_prefix(
        client, config.gcs_bucket, config.gcs_prompts_prefix, LOCAL_PROMPTS
    )
    logger.info("  Downloaded %d prompt files", prompt_count)

    # Step 2: Restore model cache
    logger.info("[3/5] Checking model cache...")
    cache_restored = restore_model_cache(client, config)
    if cache_restored:
        logger.info("  Model cache restored from GCS")
    else:
        logger.info("  Will download model from HuggingFace (first run)")

    # Step 3: Run generation (clips are uploaded incrementally to GCS)
    logger.info("[4/5] Running %s generation...", config.mode.value)
    run_generation(config, client)

    # Step 4: Save model cache (first run only)
    if not cache_restored:
        logger.info("[4.5/5] Saving model cache to GCS for future runs...")
        save_model_cache(client, config)

    logger.info("=== Datagen job complete ===")


if __name__ == "__main__":
    main()
