"""
config.py — Configuration for Cosmos Predict 2.5 Vertex AI datagen jobs.

Uses Pydantic BaseSettings with DATAGEN_ env prefix so all config can be
injected via Vertex AI Custom Job environment variables.
"""

from enum import Enum

from pydantic_settings import BaseSettings


class DatagenMode(str, Enum):
    image2world = "image2world"
    text2world = "text2world"


class DatagenConfig(BaseSettings):
    model_config = {"env_prefix": "DATAGEN_"}

    # Mode selection
    mode: DatagenMode = DatagenMode.image2world

    # GCS paths
    gcs_bucket: str = "autolane-handoff-datagen"
    gcs_site_photos_prefix: str = "datagen/site_photos"
    gcs_prompts_prefix: str = "datagen/prompts"
    gcs_output_prefix: str = "data/synthetic_clips"
    gcs_model_cache_prefix: str = "cache/hf_models"

    # Model config (NVIDIA native CLI naming)
    model_key: str = "2B/post-trained"
    num_inference_steps: int = 35
    num_output_frames: int = 77  # ~5s at 16 FPS (model's native output)
    seed: int = 42
    num_seeds: int = 1

    # Optional: filter to specific prompt IDs (comma-separated)
    prompt_ids: str | None = None

    # Batch parallelism — split work across multiple Vertex AI jobs.
    # batch_index is 0-based, total_batches is the number of parallel jobs.
    # Each job takes an equal slice of the generated param files.
    batch_index: int = 0
    total_batches: int = 1

    # Model cache — save/restore HF cache to/from GCS
    use_model_cache: bool = True

    # Guardrails
    disable_guardrails: bool = False
