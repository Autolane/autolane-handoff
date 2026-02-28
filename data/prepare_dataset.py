#!/usr/bin/env python3
"""Convert bootstrap annotations to Llava-format training data for SFT.

Reads per-clip annotation JSONs (from bootstrap_annotations.py), filters by
quality, and outputs a single Llava-format JSON file ready for Cosmos Reason 2
SFT post-training via the ITS recipe.

Pipeline step 3 of 3 (see data/README.md).

Usage:
    # Download annotations first
    python -m data.bootstrap_annotations --download

    # Convert to Llava format
    python -m data.prepare_dataset

    # Filter to only high-quality, skip clips needing review
    python -m data.prepare_dataset --skip-needs-review

    # Custom train/val split ratio
    python -m data.prepare_dataset --val-ratio 0.15
"""

import argparse
import json
import logging
import random
from pathlib import Path

from inference.prompts import (
    HANDOFF_PROMPT,
    SAFETY_PROMPT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ANNOTATIONS_DIR = Path("data/annotations")
OUTPUT_DIR = Path("data/training")


def load_annotations(
    annotations_dir: Path,
    skip_needs_review: bool = False,
    exclude_clips: set[str] | None = None,
) -> list[dict]:
    """Load all annotation JSONs from the local annotations directory."""
    annotations = []

    for mode in ("image2world", "text2world"):
        mode_dir = annotations_dir / mode
        if not mode_dir.exists():
            continue

        for clip_dir in sorted(mode_dir.iterdir()):
            ann_file = clip_dir / "annotation.json"
            if not ann_file.exists():
                continue

            with open(ann_file) as f:
                ann = json.load(f)

            if exclude_clips and ann["clip_name"] in exclude_clips:
                logger.info("  Excluding: %s", ann["clip_name"])
                continue

            if skip_needs_review and ann.get("quality", {}).get("needs_review"):
                logger.info("  Skipping (needs review): %s", ann["clip_name"])
                continue

            annotations.append(ann)

    return annotations


def _wrap_in_tags(raw: str, think: str, answer: str) -> str:
    """Wrap model output in <think>/<answer> tags for SFT training.

    If the response already has valid tags, use the parsed content.
    Otherwise, use the full raw response as thinking and generate a
    minimal structured answer.
    """
    if think and answer:
        return (
            f"<think>\n{think}\n</think>\n"
            f"<answer>\n{answer}\n</answer>"
        )
    # Zero-shot output: raw response IS the reasoning
    return (
        f"<think>\n{raw.strip()}\n</think>\n"
        f"<answer>\n{raw.strip()}\n</answer>"
    )


def annotation_to_llava(ann: dict) -> list[dict]:
    """Convert a single annotation to Llava-format training entries.

    Each clip produces two entries: one for safety, one for handoff.
    Zero-shot responses without <think>/<answer> tags are wrapped
    automatically so the model learns the structured output format.
    """
    clip_name = ann["clip_name"]
    video_path = ann["annotated_video_gcs"]

    entries = []

    # Safety QA pair
    safety = ann["safety"]
    safety_response = _wrap_in_tags(
        safety["raw_response"], safety["think"], safety["answer"],
    )

    entries.append({
        "id": f"{clip_name}_safety",
        "video": video_path,
        "conversations": [
            {
                "from": "human",
                "value": SAFETY_PROMPT,
            },
            {
                "from": "gpt",
                "value": safety_response,
            },
        ],
    })

    # Handoff QA pair
    handoff = ann["handoff"]
    handoff_response = _wrap_in_tags(
        handoff["raw_response"], handoff["think"], handoff["answer"],
    )

    entries.append({
        "id": f"{clip_name}_handoff",
        "video": video_path,
        "conversations": [
            {
                "from": "human",
                "value": HANDOFF_PROMPT,
            },
            {
                "from": "gpt",
                "value": handoff_response,
            },
        ],
    })

    return entries


def split_train_val(
    entries: list[dict], val_ratio: float = 0.1, seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split entries into train and validation sets.

    Splits by clip (not by entry) to avoid data leakage — both QA pairs
    from the same clip go into the same split.
    """
    # Group entries by clip (pairs of safety + handoff)
    clips: dict[str, list[dict]] = {}
    for entry in entries:
        # Strip _safety / _handoff suffix to get clip name
        clip_id = entry["id"].rsplit("_", 1)[0]
        clips.setdefault(clip_id, []).append(entry)

    clip_names = sorted(clips.keys())
    random.seed(seed)
    random.shuffle(clip_names)

    val_count = max(1, int(len(clip_names) * val_ratio))
    val_clips = set(clip_names[:val_count])

    train = []
    val = []
    for name in clip_names:
        target = val if name in val_clips else train
        target.extend(clips[name])

    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert bootstrap annotations to Llava-format training data",
    )
    parser.add_argument(
        "--annotations-dir", type=Path, default=ANNOTATIONS_DIR,
        help=f"Path to annotations directory (default: {ANNOTATIONS_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"Output directory for Llava JSON files (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-needs-review", action="store_true",
        help="Exclude clips flagged as needing review",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Fraction of clips for validation set (default: 0.1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split (default: 42)",
    )
    parser.add_argument(
        "--exclude", nargs="*", default=[],
        help="Clip names to exclude (e.g. IMG_2679_i2w_package_transfer)",
    )
    parser.add_argument(
        "--exclude-file", type=Path, default=None,
        help="Path to file with clip names to exclude (one per line)",
    )
    args = parser.parse_args()

    # Build exclude set
    exclude_clips: set[str] = set(args.exclude)
    if args.exclude_file and args.exclude_file.exists():
        with open(args.exclude_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    exclude_clips.add(line)

    logger.info("=== Prepare Llava Training Dataset ===")
    logger.info("  Annotations: %s", args.annotations_dir)
    logger.info("  Output: %s", args.output_dir)
    if exclude_clips:
        logger.info("  Excluding: %d clips", len(exclude_clips))

    # Load annotations
    annotations = load_annotations(
        args.annotations_dir,
        skip_needs_review=args.skip_needs_review,
        exclude_clips=exclude_clips if exclude_clips else None,
    )
    logger.info("Loaded %d clip annotations", len(annotations))

    if not annotations:
        logger.error("No annotations found. Run bootstrap + download first.")
        return

    # Convert to Llava format
    all_entries = []
    for ann in annotations:
        all_entries.extend(annotation_to_llava(ann))

    logger.info("Generated %d QA pairs", len(all_entries))

    # Quality stats
    modes = {}
    for ann in annotations:
        m = ann["mode"]
        modes[m] = modes.get(m, 0) + 1
    for m, count in sorted(modes.items()):
        logger.info("  %s: %d clips (%d pairs)", m, count, count * 2)

    needs_review = sum(
        1 for a in annotations if a.get("quality", {}).get("needs_review")
    )
    if needs_review:
        logger.info("  Flagged for review: %d clips", needs_review)

    # Split train/val
    train, val = split_train_val(
        all_entries, val_ratio=args.val_ratio, seed=args.seed,
    )
    logger.info("Split: %d train, %d val", len(train), len(val))

    # Write output
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.output_dir / "llava_train.json"
    val_path = args.output_dir / "llava_val.json"

    with open(train_path, "w") as f:
        json.dump(train, f, indent=2)
    logger.info("Wrote %s (%d entries)", train_path, len(train))

    with open(val_path, "w") as f:
        json.dump(val, f, indent=2)
    logger.info("Wrote %s (%d entries)", val_path, len(val))

    # Also write combined (for convenience)
    combined_path = args.output_dir / "llava_all.json"
    with open(combined_path, "w") as f:
        json.dump(all_entries, f, indent=2)
    logger.info("Wrote %s (%d entries)", combined_path, len(all_entries))

    logger.info("=== Dataset preparation complete ===")


if __name__ == "__main__":
    main()
