"""
generate_text2world.py — Generate synthetic curbside delivery videos using
Cosmos Predict 2.5-14B Text2World mode.

Takes text prompts only (no conditioning image) and generates 5-second video
clips at 720p, 16 FPS (93 frames per clip). Used for diverse scenario coverage
including nighttime, weather variations, and edge cases.

Usage:
    poetry run python datagen/generate_text2world.py \
        --prompts datagen/prompts/text2world_base.json \
        --output data/synthetic_clips/text2world/

    # Use augmented prompts (150+ variations)
    poetry run python datagen/generate_text2world.py \
        --prompts datagen/prompts/text2world_augmented.json \
        --output data/synthetic_clips/text2world/

    # Generate specific prompts only
    poetry run python datagen/generate_text2world.py \
        --prompts datagen/prompts/text2world_base.json \
        --output data/synthetic_clips/text2world/ \
        --prompt-ids t2w_full_handoff t2w_nighttime_delivery

    # Use 2B model for faster iteration
    poetry run python datagen/generate_text2world.py \
        --prompts datagen/prompts/text2world_base.json \
        --output data/synthetic_clips/text2world/ \
        --model nvidia/Cosmos-Predict2.5-2B
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from diffusers import Cosmos2_5_PredictBasePipeline
from diffusers.utils import export_to_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Cosmos Predict 2.5 output spec
NUM_FRAMES = 93  # ~5 seconds at 16 FPS
FPS = 16
NUM_INFERENCE_STEPS = 36

# Negative prompt for quality filtering (from NVIDIA docs)
NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, "
    "static with no motion, motion blur, over-saturation, shaky footage, "
    "low resolution, grainy, bad quality, watermark, text overlay."
)


def load_prompts(prompts_path: Path, prompt_ids: list[str] | None = None) -> list[dict]:
    """Load prompts from JSON file, optionally filtering by ID."""
    with open(prompts_path) as f:
        prompts = json.load(f)

    if prompt_ids:
        prompts = [p for p in prompts if p["id"] in prompt_ids]
        if not prompts:
            logger.error("No prompts matched the given IDs: %s", prompt_ids)
            sys.exit(1)

    logger.info("Loaded %d prompts from %s", len(prompts), prompts_path)
    return prompts


def generate_clip(
    pipe: Cosmos2_5_PredictBasePipeline,
    prompt: str,
    seed: int = 42,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
) -> list:
    """Generate a single video clip from a text prompt (no conditioning image)."""
    generator = torch.Generator(device="cuda").manual_seed(seed)

    output = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_frames=NUM_FRAMES,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )

    return output.frames[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic videos with Cosmos Predict 2.5 Text2World"
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("datagen/prompts/text2world_base.json"),
        help="Path to prompts JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic_clips/text2world"),
        help="Output directory for generated clips",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/Cosmos-Predict2.5-14B",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="diffusers/base/post-trained",
        help="Model revision/branch",
    )
    parser.add_argument(
        "--prompt-ids",
        nargs="+",
        default=None,
        help="Generate only specific prompt IDs (space-separated)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of seeds per prompt (for generating variations)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=NUM_INFERENCE_STEPS,
        help="Number of denoising steps (higher = better quality, slower)",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        help="Device map for multi-GPU (e.g. 'balanced')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without running inference",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.prompts.exists():
        logger.error("Prompts file does not exist: %s", args.prompts)
        sys.exit(1)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts = load_prompts(args.prompts, args.prompt_ids)

    total_clips = len(prompts) * args.num_seeds
    logger.info(
        "Will generate %d clips (%d prompts x %d seeds)",
        total_clips, len(prompts), args.num_seeds,
    )

    if args.dry_run:
        for prompt_data in prompts:
            for seed_idx in range(args.num_seeds):
                seed = args.seed + seed_idx
                suffix = f"_s{seed}" if args.num_seeds > 1 else ""
                output_name = f"{prompt_data['id']}{suffix}.mp4"
                logger.info(
                    "  [DRY RUN] '%s' (seed=%d) -> %s",
                    prompt_data["id"], seed, output_name,
                )
        logger.info("Dry run complete. %d clips would be generated.", total_clips)
        return

    # Load model
    logger.info("Loading Cosmos Predict 2.5 pipeline: %s", args.model)
    pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
        args.model,
        revision=args.revision,
        device_map=args.device_map,
        torch_dtype=torch.bfloat16,
    )
    if args.device_map is None:
        pipe = pipe.to("cuda")
    logger.info("Model loaded successfully")

    # Generate clips
    generated = 0
    failed = 0
    clip_num = 0

    for prompt_data in prompts:
        for seed_idx in range(args.num_seeds):
            clip_num += 1
            seed = args.seed + seed_idx
            suffix = f"_s{seed}" if args.num_seeds > 1 else ""
            output_name = f"{prompt_data['id']}{suffix}.mp4"
            output_path = args.output / output_name

            if output_path.exists():
                logger.info(
                    "[%d/%d] Skipping (exists): %s", clip_num, total_clips, output_name
                )
                generated += 1
                continue

            logger.info(
                "[%d/%d] Generating: '%s' (seed=%d)",
                clip_num, total_clips, prompt_data["id"], seed,
            )

            start_time = time.time()

            try:
                frames = generate_clip(
                    pipe,
                    prompt_data["prompt"],
                    seed=seed,
                    num_inference_steps=args.num_inference_steps,
                )
                export_to_video(frames, str(output_path), fps=FPS)
                elapsed = time.time() - start_time
                logger.info(
                    "  Saved: %s (%.1fs, %d frames)", output_path, elapsed, len(frames)
                )
                generated += 1
            except Exception:
                logger.exception("  Failed to generate %s", output_name)
                failed += 1

    logger.info("")
    logger.info("=== Text2World Generation Complete ===")
    logger.info("  Generated: %d/%d clips", generated, total_clips)
    if failed:
        logger.warning("  Failed: %d clips", failed)
    logger.info("  Output directory: %s", args.output.resolve())


if __name__ == "__main__":
    main()
