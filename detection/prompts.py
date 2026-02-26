"""Grounding DINO detection prompts for curbside handoff scenarios."""

from __future__ import annotations

# Open-vocabulary detection prompt — period-separated classes
# Grounding DINO uses "." as class separator in text prompts
#
# Updated to match synthetic data prompts: generic vehicle descriptions
# (white sedan, white SUV, silver sedan) instead of specific makes since
# Cosmos Predict 2.5-2B doesn't reliably render branded vehicles.
DETECTION_PROMPT = (
    "white sedan . silver sedan . white SUV . dark gray SUV . "
    "four-wheeled delivery robot . bipedal robot . pedestrian . "
    "parking sign . loading zone sign . obstruction . "
    "cardboard box . open rear hatch . stroller . bicycle"
)

# Confidence thresholds
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# Agent class groupings for downstream reasoning
VEHICLE_CLASSES = {"white sedan", "silver sedan", "white SUV", "dark gray SUV"}
ROBOT_CLASSES = {"four-wheeled delivery robot", "bipedal robot"}
HUMAN_CLASSES = {"pedestrian"}
SIGNAGE_CLASSES = {"parking sign", "loading zone sign"}
HAZARD_CLASSES = {"obstruction"}
OBJECT_CLASSES = {"cardboard box", "open rear hatch", "stroller", "bicycle"}

ALL_CLASSES = (
    VEHICLE_CLASSES | ROBOT_CLASSES | HUMAN_CLASSES
    | SIGNAGE_CLASSES | HAZARD_CLASSES | OBJECT_CLASSES
)


def classify_label(label: str) -> str:
    """Map a detected label string to its agent group name.

    Returns one of: vehicle, robot, human, signage, hazard, object, unknown.
    """
    label_lower = label.lower()
    for cls in VEHICLE_CLASSES:
        if cls.lower() in label_lower:
            return "vehicle"
    for cls in ROBOT_CLASSES:
        if cls.lower() in label_lower:
            return "robot"
    for cls in HUMAN_CLASSES:
        if cls.lower() in label_lower:
            return "human"
    for cls in SIGNAGE_CLASSES:
        if cls.lower() in label_lower:
            return "signage"
    for cls in HAZARD_CLASSES:
        if cls.lower() in label_lower:
            return "hazard"
    for cls in OBJECT_CLASSES:
        if cls.lower() in label_lower:
            return "object"
    return "unknown"
