"""Grounding DINO detection prompts for curbside handoff scenarios."""

# Open-vocabulary detection prompt — period-separated classes
DETECTION_PROMPT = (
    "Tesla Model Y . Tesla Model 3 . Waymo Jaguar I-PACE . "
    "delivery robot . humanoid robot . pedestrian . "
    "parking sign . loading zone sign . obstruction"
)

# Confidence thresholds
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# Agent class groupings for downstream reasoning
VEHICLE_CLASSES = {"Tesla Model Y", "Tesla Model 3", "Waymo Jaguar I-PACE"}
ROBOT_CLASSES = {"delivery robot", "humanoid robot"}
HUMAN_CLASSES = {"pedestrian"}
SIGNAGE_CLASSES = {"parking sign", "loading zone sign"}
HAZARD_CLASSES = {"obstruction"}
