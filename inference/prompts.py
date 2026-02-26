"""System and per-pass prompts for Cosmos Reason 2 inference.

Cosmos Reason 2-8B uses a two-pass prompting strategy over annotated video
frames (with Grounding DINO bounding box overlays):

  Pass 1 — Safety + Compliance: spatial relationships, signage OCR,
           pedestrian conflicts, obstruction detection, safety score (0-100)
  Pass 2 — Handoff Planning: agent sequencing, trajectory coordinates,
           estimated handoff time, abort conditions

Both passes receive annotated frames and reason in <think> / <answer> tags.

vLLM serving config:
  Model: nvidia/Cosmos-Reason2-8B (based on Qwen3-VL-8B-Instruct)
  Port:  8001
  API:   OpenAI-compatible /v1/chat/completions
"""

from typing import Any

# ---------------------------------------------------------------------------
# vLLM / model configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "cosmos-reason2-8b"
VLLM_BASE_URL = "http://localhost:8001/v1"
MAX_TOKENS = 4096
VIDEO_FPS = 4  # Must match training setup (4 FPS, 8 frames per chunk)

# ---------------------------------------------------------------------------
# System prompt — shared across both passes
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are Autolane Handoff, an AI system that reasons about autonomous "
    "delivery handoffs at curbside zones. You receive video frames that have "
    "been pre-processed with Grounding DINO object detection — bounding boxes "
    "and agent classifications are already overlaid on the frames. Your job "
    "is to REASON over these detections, not re-detect them. Specifically:\n"
    "\n"
    "1. Assess safety conditions for autonomous operations given the "
    "detected agents and their positions\n"
    "2. Read and interpret any signage (parking signs, loading zone markers, "
    "time restrictions) using OCR\n"
    "3. Determine zone compliance for autonomous handoff operations\n"
    "4. Plan optimal multi-agent handoff sequences with trajectories\n"
    "5. Generate alerts for safety violations or coordination failures\n"
    "\n"
    "Always reason step-by-step in <think> tags before providing your "
    "assessment in <answer> tags. Reference the detected agents by their "
    "bounding box labels when reasoning about spatial relationships."
)

# ---------------------------------------------------------------------------
# Pass 1 — Safety + Compliance (includes OCR)
# ---------------------------------------------------------------------------
SAFETY_PROMPT = (
    "<video>\n"
    "You are viewing a curbside zone with Grounding DINO detection overlays.\n"
    "Assess this zone for autonomous delivery handoff safety:\n"
    "1. Analyze the spatial relationships between all detected agents\n"
    "2. Read all visible signage and interpret restrictions\n"
    "3. Identify any pedestrian conflict zones based on detected positions\n"
    "4. Check for obstructions in the handoff path\n"
    "5. Rate overall safety (0-100) with reasoning\n"
    "\n"
    "Answer the question using the following format:\n"
    "\n"
    "<think>\n"
    "Your step-by-step reasoning about safety, signage, and compliance.\n"
    "</think>\n"
    "\n"
    "<answer>\n"
    "Provide a structured safety assessment with:\n"
    "- safety_score: integer 0-100\n"
    "- agents_detected: list of agents with bounding box labels and types\n"
    "- signage: list of signs read via OCR with interpreted restrictions\n"
    "- pedestrian_conflicts: list of potential conflict zones\n"
    "- obstructions: list of detected obstructions\n"
    "- zone_compliant: boolean — is this zone safe for autonomous handoff?\n"
    "- alerts: list of safety alerts if any\n"
    "</answer>"
)

# ---------------------------------------------------------------------------
# Pass 2 — Handoff Planning
# ---------------------------------------------------------------------------
HANDOFF_PROMPT = (
    "<video>\n"
    "Given the agents detected in this curbside zone (shown with bounding "
    "box overlays), plan the optimal handoff sequence:\n"
    "1. Which agent should move first?\n"
    "2. What trajectory should each agent follow? Provide coordinate paths.\n"
    "3. What is the estimated time for the complete handoff?\n"
    "4. What conditions would trigger an abort?\n"
    "\n"
    "Answer the question using the following format:\n"
    "\n"
    "<think>\n"
    "Your step-by-step reasoning about agent sequencing and trajectories.\n"
    "</think>\n"
    "\n"
    "<answer>\n"
    "Provide a structured handoff plan with:\n"
    "- readiness: READY | NOT_READY | CAUTION\n"
    "- sequence: ordered list of agent actions with:\n"
    "    - agent: bounding box label\n"
    "    - action: description of movement\n"
    "    - trajectory: list of (x, y) coordinate waypoints\n"
    "    - estimated_seconds: time for this step\n"
    "- total_estimated_seconds: total handoff time\n"
    "- abort_conditions: list of conditions that should trigger abort\n"
    "- time_constraint: any loading zone time limits from signage\n"
    "</answer>"
)


def build_safety_messages(video_path: str) -> list[dict[str, Any]]:
    """Build the message payload for Pass 1 (Safety + Compliance).

    Args:
        video_path: Path to the annotated video file (with GDINO overlays).

    Returns:
        List of message dicts compatible with the OpenAI chat completions API.
    """
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video_path}",
                    "fps": VIDEO_FPS,
                },
                {"type": "text", "text": SAFETY_PROMPT},
            ],
        },
    ]


def build_handoff_messages(video_path: str) -> list[dict[str, Any]]:
    """Build the message payload for Pass 2 (Handoff Planning).

    Args:
        video_path: Path to the annotated video file (with GDINO overlays).

    Returns:
        List of message dicts compatible with the OpenAI chat completions API.
    """
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video_path}",
                    "fps": VIDEO_FPS,
                },
                {"type": "text", "text": HANDOFF_PROMPT},
            ],
        },
    ]
