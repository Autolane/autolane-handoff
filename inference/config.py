"""Configuration settings for Autolane Handoff inference."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # GCP Configuration
    gcp_project_id: str = "autolane-handoff-20260221"
    gcp_region: str = "us-central1"

    # Vertex AI Configuration (set after deployment)
    vertex_endpoint_id: str = ""
    vertex_model_id: str = ""

    # Model Configuration
    model_name: str = "nvidia/Cosmos-Reason2-8B"

    # Deployment Configuration
    machine_type: str = "a2-highgpu-1g"
    accelerator_type: str = "NVIDIA_TESLA_A100"
    accelerator_count: int = 1
    min_replicas: int = 1
    max_replicas: int = 1

    # vLLM Configuration
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"

    # Inference Configuration
    temperature: float = 0.2
    max_tokens: int = 4096
    video_fps: int = 4

    # Fine-tuned model configuration
    use_finetuned_model: bool = False
    finetuned_model_gcs_uri: str = ""

    @property
    def active_model_source(self) -> str:
        """Return the GCS URI for the fine-tuned model, or the HF model name for base."""
        if self.use_finetuned_model and self.finetuned_model_gcs_uri:
            return self.finetuned_model_gcs_uri
        return self.model_name


settings = Settings()
