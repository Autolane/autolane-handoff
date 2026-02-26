"""Grounding DINO inference wrapper for Autolane Handoff Stage 1 detection.

Loads Grounding DINO (SwinT or SwinB) and runs open-vocabulary object
detection on curbside video frames. Returns bounding boxes, confidence
scores, and class labels for all detected agents.

Usage:
    python -m detection.grounding_dino --input frames/ --output results.json
    python -m detection.grounding_dino --input frame.jpg --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from groundingdino.util.inference import load_model, load_image, predict

from detection.prompts import (
    BOX_THRESHOLD,
    DETECTION_PROMPT,
    TEXT_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Model variants — SwinT is lighter, SwinB is more accurate
MODEL_VARIANTS = {
    "swint": {
        "config": "GroundingDINO_SwinT_OGC.py",
        "weights_url": (
            "https://github.com/IDEA-Research/GroundingDINO/releases/"
            "download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        ),
        "weights_file": "groundingdino_swint_ogc.pth",
    },
    "swinb": {
        "config": "GroundingDINO_SwinB_cogcoor.py",
        "weights_url": (
            "https://github.com/IDEA-Research/GroundingDINO/releases/"
            "download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
        ),
        "weights_file": "groundingdino_swinb_cogcoor.pth",
    },
}

DEFAULT_VARIANT = "swint"
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


@dataclass
class Detection:
    """A single detected object with bounding box, label, and confidence."""

    label: str
    confidence: float
    # Bounding box in xyxy format (pixel coordinates)
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


@dataclass
class FrameDetections:
    """All detections for a single frame."""

    frame_path: str
    width: int
    height: int
    detections: list[Detection]

    def to_dict(self) -> dict:
        return {
            "frame_path": self.frame_path,
            "width": self.width,
            "height": self.height,
            "detections": [asdict(d) for d in self.detections],
        }


class GroundingDINODetector:
    """Wraps Grounding DINO for open-vocabulary curbside object detection."""

    def __init__(
        self,
        variant: str = DEFAULT_VARIANT,
        config_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        box_threshold: float = BOX_THRESHOLD,
        text_threshold: float = TEXT_THRESHOLD,
    ):
        self.variant = variant
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        variant_info = MODEL_VARIANTS[variant]

        if config_path is None:
            # groundingdino package ships configs under groundingdino/config/
            import groundingdino

            pkg_root = Path(groundingdino.__file__).resolve().parent
            config_path = str(pkg_root / "config" / variant_info["config"])
        if weights_path is None:
            weights_path = str(WEIGHTS_DIR / variant_info["weights_file"])

        self._config_path = config_path
        self._weights_path = weights_path
        self._model = None

    def load(self) -> None:
        """Load the Grounding DINO model into memory."""
        weights = Path(self._weights_path)
        if not weights.exists():
            raise FileNotFoundError(
                f"Model weights not found at {weights}. "
                f"Download them with: scripts/deploy_gdino.sh"
            )
        logger.info(
            "Loading Grounding DINO (%s) on %s", self.variant, self.device
        )
        self._model = load_model(
            self._config_path, self._weights_path, device=self.device
        )
        logger.info("Model loaded successfully")

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    def detect_image(
        self,
        image_path: str,
        prompt: str = DETECTION_PROMPT,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> FrameDetections:
        """Run detection on a single image file.

        Args:
            image_path: Path to the image file.
            prompt: Period-separated detection classes.
            box_threshold: Override default box confidence threshold.
            text_threshold: Override default text confidence threshold.

        Returns:
            FrameDetections with all detected objects.
        """
        box_thresh = box_threshold if box_threshold is not None else self.box_threshold
        text_thresh = text_threshold if text_threshold is not None else self.text_threshold

        image_source, image = load_image(image_path)
        h, w = image_source.shape[:2]

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=prompt,
            box_threshold=box_thresh,
            text_threshold=text_thresh,
        )

        detections = []
        for box, score, label in zip(boxes, logits, phrases):
            # boxes from predict() are in cxcywh normalized format — convert to xyxy pixels
            cx, cy, bw, bh = box.tolist()
            x_min = (cx - bw / 2) * w
            y_min = (cy - bh / 2) * h
            x_max = (cx + bw / 2) * w
            y_max = (cy + bh / 2) * h

            detections.append(
                Detection(
                    label=label.strip(),
                    confidence=round(float(score), 4),
                    x_min=round(x_min, 1),
                    y_min=round(y_min, 1),
                    x_max=round(x_max, 1),
                    y_max=round(y_max, 1),
                )
            )

        return FrameDetections(
            frame_path=str(image_path),
            width=w,
            height=h,
            detections=detections,
        )

    def detect_frame(
        self,
        frame: np.ndarray,
        prompt: str = DETECTION_PROMPT,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        frame_id: str = "",
    ) -> FrameDetections:
        """Run detection on an in-memory numpy frame (BGR from OpenCV).

        Args:
            frame: BGR numpy array from cv2 or video capture.
            prompt: Period-separated detection classes.
            box_threshold: Override default box confidence threshold.
            text_threshold: Override default text confidence threshold.
            frame_id: Optional identifier for this frame.

        Returns:
            FrameDetections with all detected objects.
        """
        import groundingdino.datasets.transforms as T
        from PIL import Image

        box_thresh = box_threshold if box_threshold is not None else self.box_threshold
        text_thresh = text_threshold if text_threshold is not None else self.text_threshold

        h, w = frame.shape[:2]

        # Convert BGR numpy to RGB PIL for the transform
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_transformed, _ = transform(image_pil, None)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image_transformed,
            caption=prompt,
            box_threshold=box_thresh,
            text_threshold=text_thresh,
        )

        detections = []
        for box, score, label in zip(boxes, logits, phrases):
            cx, cy, bw, bh = box.tolist()
            x_min = (cx - bw / 2) * w
            y_min = (cy - bh / 2) * h
            x_max = (cx + bw / 2) * w
            y_max = (cy + bh / 2) * h

            detections.append(
                Detection(
                    label=label.strip(),
                    confidence=round(float(score), 4),
                    x_min=round(x_min, 1),
                    y_min=round(y_min, 1),
                    x_max=round(x_max, 1),
                    y_max=round(y_max, 1),
                )
            )

        return FrameDetections(
            frame_path=frame_id,
            width=w,
            height=h,
            detections=detections,
        )

    def detect_video_frames(
        self,
        frames_dir: str,
        prompt: str = DETECTION_PROMPT,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ) -> list[FrameDetections]:
        """Run detection on all image files in a directory.

        Args:
            frames_dir: Directory containing extracted video frames.
            prompt: Period-separated detection classes.
            extensions: Image file extensions to process.

        Returns:
            List of FrameDetections, one per image, sorted by filename.
        """
        frames_path = Path(frames_dir)
        image_files = sorted(
            f for f in frames_path.iterdir() if f.suffix.lower() in extensions
        )
        if not image_files:
            logger.warning("No image files found in %s", frames_dir)
            return []

        logger.info("Detecting objects in %d frames from %s", len(image_files), frames_dir)
        results = []
        for img_file in image_files:
            result = self.detect_image(str(img_file), prompt=prompt)
            n = len(result.detections)
            logger.debug("  %s: %d detections", img_file.name, n)
            results.append(result)

        total = sum(len(r.detections) for r in results)
        logger.info("Detection complete: %d total detections across %d frames", total, len(results))
        return results


def save_results(results: list[FrameDetections], output_path: str) -> None:
    """Save detection results to JSON."""
    data = [r.to_dict() for r in results]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Results saved to %s", output_path)


def load_results(json_path: str) -> list[FrameDetections]:
    """Load detection results from JSON."""
    with open(json_path) as f:
        data = json.load(f)
    results = []
    for item in data:
        detections = [Detection(**d) for d in item["detections"]]
        results.append(
            FrameDetections(
                frame_path=item["frame_path"],
                width=item["width"],
                height=item["height"],
                detections=detections,
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Grounding DINO detection on video frames"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to an image file or directory of frames",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output JSON path for detection results",
    )
    parser.add_argument(
        "--variant", default=DEFAULT_VARIANT, choices=MODEL_VARIANTS.keys(),
        help="Model variant: swint (faster) or swinb (more accurate)",
    )
    parser.add_argument(
        "--prompt", default=DETECTION_PROMPT,
        help="Detection prompt (period-separated classes)",
    )
    parser.add_argument(
        "--box-threshold", type=float, default=BOX_THRESHOLD,
        help="Box confidence threshold",
    )
    parser.add_argument(
        "--text-threshold", type=float, default=TEXT_THRESHOLD,
        help="Text confidence threshold",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device: cuda, cpu, or auto (default: auto)",
    )
    parser.add_argument(
        "--weights", default=None,
        help="Path to model weights (default: weights/<variant>.pth)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    detector = GroundingDINODetector(
        variant=args.variant,
        weights_path=args.weights,
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    input_path = Path(args.input)
    if input_path.is_dir():
        results = detector.detect_video_frames(str(input_path), prompt=args.prompt)
    elif input_path.is_file():
        result = detector.detect_image(str(input_path), prompt=args.prompt)
        results = [result]
    else:
        logger.error("Input path does not exist: %s", args.input)
        sys.exit(1)

    save_results(results, args.output)

    # Print summary
    total_detections = sum(len(r.detections) for r in results)
    print(f"\nProcessed {len(results)} frame(s), {total_detections} detection(s)")
    for r in results:
        print(f"  {Path(r.frame_path).name}: {len(r.detections)} detections")
        for d in r.detections:
            print(f"    [{d.confidence:.2f}] {d.label} @ ({d.x_min:.0f},{d.y_min:.0f})-({d.x_max:.0f},{d.y_max:.0f})")


if __name__ == "__main__":
    main()
