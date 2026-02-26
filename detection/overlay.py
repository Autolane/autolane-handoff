"""Render Grounding DINO BBox overlays onto frames for Stage 2 input.

Burns bounding boxes with labels and confidence scores directly into
frames so Cosmos Reason 2 can see the detection annotations visually.

Usage:
    python -m detection.overlay --detections results.json --frames frames/ --output annotated/
    python -m detection.overlay --detections results.json --frame single.jpg --output annotated/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from detection.grounding_dino import Detection, FrameDetections, load_results
from detection.prompts import (
    HAZARD_CLASSES,
    HUMAN_CLASSES,
    ROBOT_CLASSES,
    SIGNAGE_CLASSES,
    VEHICLE_CLASSES,
)

logger = logging.getLogger(__name__)

# Color palette (BGR) — grouped by agent class for visual clarity
COLOR_VEHICLE = (0, 180, 0)       # Green — autonomous vehicles
COLOR_ROBOT = (255, 140, 0)       # Blue-ish orange — delivery/humanoid robots
COLOR_HUMAN = (0, 200, 255)       # Yellow — pedestrians
COLOR_SIGNAGE = (255, 200, 50)    # Light blue — signage
COLOR_HAZARD = (0, 0, 255)        # Red — obstructions/hazards
COLOR_DEFAULT = (200, 200, 200)   # Gray — unknown

# Overlay rendering parameters
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICKNESS = 1
BOX_THICKNESS = 2
LABEL_PAD = 4


def get_color(label: str) -> tuple[int, int, int]:
    """Get color for a detection label based on its class group."""
    label_lower = label.lower()
    for cls in VEHICLE_CLASSES:
        if cls.lower() in label_lower:
            return COLOR_VEHICLE
    for cls in ROBOT_CLASSES:
        if cls.lower() in label_lower:
            return COLOR_ROBOT
    for cls in HUMAN_CLASSES:
        if cls.lower() in label_lower:
            return COLOR_HUMAN
    for cls in SIGNAGE_CLASSES:
        if cls.lower() in label_lower:
            return COLOR_SIGNAGE
    for cls in HAZARD_CLASSES:
        if cls.lower() in label_lower:
            return COLOR_HAZARD
    return COLOR_DEFAULT


def render_detection(
    frame: np.ndarray,
    det: Detection,
    color: Optional[tuple[int, int, int]] = None,
    show_confidence: bool = True,
) -> None:
    """Draw a single detection box and label onto a frame (in-place).

    Args:
        frame: BGR numpy array to draw on.
        det: Detection with bounding box, label, confidence.
        color: Override color (BGR). If None, auto-select by class.
        show_confidence: Whether to show confidence score in label.
    """
    if color is None:
        color = get_color(det.label)

    x1, y1 = int(det.x_min), int(det.y_min)
    x2, y2 = int(det.x_max), int(det.y_max)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    # Build label text
    if show_confidence:
        label_text = f"{det.label} {det.confidence:.0%}"
    else:
        label_text = det.label

    # Measure text size for background
    (tw, th), baseline = cv2.getTextSize(label_text, FONT, FONT_SCALE, FONT_THICKNESS)

    # Label background — positioned above the box (or inside if at top edge)
    label_y = y1 - LABEL_PAD if y1 - th - 2 * LABEL_PAD > 0 else y1 + th + 2 * LABEL_PAD
    bg_y1 = label_y - th - LABEL_PAD
    bg_y2 = label_y + LABEL_PAD

    cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 2 * LABEL_PAD, bg_y2), color, -1)

    # Label text — white on colored background
    cv2.putText(
        frame, label_text,
        (x1 + LABEL_PAD, label_y),
        FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA,
    )


def render_frame_detections(
    frame: np.ndarray,
    detections: list[Detection],
    show_confidence: bool = True,
) -> np.ndarray:
    """Render all detections onto a frame.

    Args:
        frame: BGR numpy array (will be copied, not modified in-place).
        detections: List of Detection objects.
        show_confidence: Whether to show confidence scores.

    Returns:
        New BGR numpy array with overlays burned in.
    """
    annotated = frame.copy()
    for det in detections:
        render_detection(annotated, det, show_confidence=show_confidence)
    return annotated


def render_from_file(
    image_path: str,
    detections: list[Detection],
    output_path: str,
    show_confidence: bool = True,
) -> None:
    """Load an image, render detections, and save the result.

    Args:
        image_path: Path to source image.
        detections: List of Detection objects.
        output_path: Where to save the annotated image.
        show_confidence: Whether to show confidence scores.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error("Could not read image: %s", image_path)
        return

    annotated = render_frame_detections(frame, detections, show_confidence=show_confidence)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, annotated)
    logger.debug("Saved annotated frame: %s", output_path)


def render_batch(
    results: list[FrameDetections],
    output_dir: str,
    show_confidence: bool = True,
) -> list[str]:
    """Render overlays for a batch of detection results.

    Args:
        results: List of FrameDetections from grounding_dino.py.
        output_dir: Directory to save annotated frames.
        show_confidence: Whether to show confidence scores.

    Returns:
        List of output file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    output_files = []
    for frame_result in results:
        src = Path(frame_result.frame_path)
        dst = out_path / f"annotated_{src.name}"

        render_from_file(
            str(src),
            frame_result.detections,
            str(dst),
            show_confidence=show_confidence,
        )
        output_files.append(str(dst))

    logger.info(
        "Rendered %d annotated frames to %s", len(output_files), output_dir
    )
    return output_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render Grounding DINO detection overlays onto frames"
    )
    parser.add_argument(
        "--detections", "-d", required=True,
        help="Path to detection results JSON (from grounding_dino.py)",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for annotated frames",
    )
    parser.add_argument(
        "--no-confidence", action="store_true",
        help="Hide confidence scores in labels",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    results = load_results(args.detections)
    output_files = render_batch(
        results, args.output, show_confidence=not args.no_confidence
    )
    print(f"\nRendered {len(output_files)} annotated frame(s) to {args.output}")


if __name__ == "__main__":
    main()
