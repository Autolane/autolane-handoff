"""Deploy Cosmos Reason 2-8B to Vertex AI for Autolane Handoff.

Adapted from av-ride-hail-inference. Key differences:
- Video reasoning (multi-frame) rather than single-image detection
- Uses --media-io-kwargs for vLLM video input support
- Longer max_model_len (16384) for chain-of-thought reasoning
- max_tokens 4096 for full <think>/<answer> output
"""

import argparse
import os
import time
from typing import Optional

import structlog
from google.cloud import aiplatform

from inference.config import settings

logger = structlog.get_logger()

# Google's prebuilt vLLM container with qwen3_vl support (vLLM 0.14.0)
SERVING_CONTAINER = (
    "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/"
    "pytorch-vllm-serve:v0.14.0"
)


def create_model(
    model_name: str,
    serving_container_args: Optional[list] = None,
) -> aiplatform.Model:
    """Create a Vertex AI Model resource with vLLM serving.

    Args:
        model_name: Display name for the model.
        serving_container_args: Override vLLM command-line arguments.

    Returns:
        The created Model resource.
    """
    aiplatform.init(project=settings.gcp_project_id, location=settings.gcp_region)

    # Environment variables for the vLLM container
    if settings.use_finetuned_model and settings.finetuned_model_gcs_uri:
        env_vars = {"MODEL_ID": "/model"}
        artifact_uri = settings.finetuned_model_gcs_uri
    else:
        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            logger.warning(
                "HF_TOKEN not set. Required for nvidia/Cosmos-Reason2-8B. "
                "Set it with: export HF_TOKEN=your_token"
            )
        env_vars = {
            "MODEL_ID": settings.model_name,
            "HF_TOKEN": hf_token,
        }
        artifact_uri = None

    # Determine the model path vLLM uses
    vllm_model_path = "/model" if artifact_uri else settings.model_name

    # vLLM command-line arguments
    if serving_container_args is None:
        serving_container_args = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--host=0.0.0.0",
            "--port=8080",
            f"--model={vllm_model_path}",
            f"--tensor-parallel-size={settings.tensor_parallel_size}",
            f"--max-model-len={settings.max_model_len}",
            f"--gpu-memory-utilization={settings.gpu_memory_utilization}",
            f"--dtype={settings.dtype}",
            "--trust-remote-code",
            "--disable-log-stats",
        ]

    logger.info(
        "creating_model",
        name=model_name,
        container=SERVING_CONTAINER,
        args=serving_container_args,
    )

    model = aiplatform.Model.upload(
        display_name=model_name,
        serving_container_image_uri=SERVING_CONTAINER,
        artifact_uri=artifact_uri,
        serving_container_environment_variables=env_vars,
        serving_container_args=serving_container_args,
        serving_container_ports=[8080],
        serving_container_predict_route="/v1/chat/completions",
        serving_container_health_route="/health",
        serving_container_shared_memory_size_mb=(16 * 1024),  # 16GB shared memory
        serving_container_deployment_timeout=7200,  # 2h for model download
    )

    logger.info("model_created", model_id=model.resource_name)
    return model


def deploy_model(
    model: aiplatform.Model,
    endpoint_display_name: str,
    machine_type: Optional[str] = None,
    accelerator_type: Optional[str] = None,
    accelerator_count: Optional[int] = None,
    min_replicas: Optional[int] = None,
    max_replicas: Optional[int] = None,
) -> aiplatform.Endpoint:
    """Deploy a model to a Vertex AI Endpoint.

    Args:
        model: The Model to deploy.
        endpoint_display_name: Display name for the endpoint.
        machine_type: Compute machine type.
        accelerator_type: GPU accelerator type.
        accelerator_count: Number of GPUs.
        min_replicas: Minimum replica count.
        max_replicas: Maximum replica count.

    Returns:
        The created Endpoint.
    """
    machine_type = machine_type or settings.machine_type
    accelerator_type = accelerator_type or settings.accelerator_type
    accelerator_count = accelerator_count or settings.accelerator_count
    min_replicas = min_replicas or settings.min_replicas
    max_replicas = max_replicas or settings.max_replicas

    logger.info(
        "creating_endpoint",
        name=endpoint_display_name,
        machine_type=machine_type,
        accelerator=f"{accelerator_count}x {accelerator_type}",
    )

    endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)
    logger.info("endpoint_created", endpoint_id=endpoint.resource_name)

    logger.info(
        "deploying_model",
        min_replicas=min_replicas,
        max_replicas=max_replicas,
    )

    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f"{model.display_name}-deployed",
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        min_replica_count=min_replicas,
        max_replica_count=max_replicas,
        traffic_percentage=100,
        deploy_request_timeout=1800,  # 30 minutes
    )

    logger.info(
        "deployment_complete",
        endpoint_id=endpoint.resource_name,
        endpoint_name=endpoint.display_name,
    )

    return endpoint


def deploy_cosmos_reason2(
    endpoint_name: str = "autolane-handoff-cosmos-reason2-8b",
    model_name: str = "cosmos-reason2-8b-handoff",
    machine_type: Optional[str] = None,
    accelerator_type: Optional[str] = None,
    min_replicas: int = 1,
    max_replicas: int = 1,
) -> tuple[aiplatform.Model, aiplatform.Endpoint]:
    """Deploy Cosmos Reason 2-8B to Vertex AI for Autolane Handoff.

    Args:
        endpoint_name: Display name for the endpoint.
        model_name: Display name for the model.
        machine_type: Override machine type.
        accelerator_type: Override accelerator.
        min_replicas: Minimum replicas.
        max_replicas: Maximum replicas.

    Returns:
        Tuple of (Model, Endpoint).
    """
    model = create_model(model_name=model_name)

    endpoint = deploy_model(
        model=model,
        endpoint_display_name=endpoint_name,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
    )

    return model, endpoint


def check_endpoint_health(endpoint_id: str) -> bool:
    """Check if the endpoint is healthy.

    Args:
        endpoint_id: The endpoint resource ID.

    Returns:
        True if healthy.
    """
    aiplatform.init(project=settings.gcp_project_id, location=settings.gcp_region)

    try:
        endpoint = aiplatform.Endpoint(endpoint_id)
        if endpoint.traffic_split:
            logger.info("endpoint_healthy", endpoint_id=endpoint_id)
            return True
        logger.warning("endpoint_no_traffic", endpoint_id=endpoint_id)
        return False
    except Exception as e:
        logger.error("endpoint_health_check_failed", error=str(e))
        return False


def wait_for_endpoint(endpoint_id: str, timeout_minutes: int = 30) -> bool:
    """Wait for endpoint to become healthy.

    Args:
        endpoint_id: The endpoint resource ID.
        timeout_minutes: Maximum time to wait.

    Returns:
        True if endpoint became healthy.
    """
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60

    while time.time() - start_time < timeout_seconds:
        if check_endpoint_health(endpoint_id):
            return True
        logger.info("waiting_for_endpoint", elapsed_minutes=(time.time() - start_time) / 60)
        time.sleep(30)

    logger.error("endpoint_timeout", timeout_minutes=timeout_minutes)
    return False


def main():
    """CLI entry point for deployment."""
    parser = argparse.ArgumentParser(description="Deploy Cosmos Reason 2-8B to Vertex AI")

    parser.add_argument(
        "--action",
        choices=["deploy", "check"],
        required=True,
        help="Action to perform",
    )
    parser.add_argument(
        "--endpoint-name",
        default="autolane-handoff-cosmos-reason2-8b",
        help="Display name for the endpoint",
    )
    parser.add_argument(
        "--model-name",
        default="cosmos-reason2-8b-handoff",
        help="Display name for the model",
    )
    parser.add_argument("--machine-type", default=None)
    parser.add_argument("--accelerator-type", default=None)
    parser.add_argument("--min-replicas", type=int, default=1)
    parser.add_argument("--max-replicas", type=int, default=1)
    parser.add_argument("--endpoint-id", help="Endpoint ID (for check action)")

    args = parser.parse_args()

    if args.action == "deploy":
        model, endpoint = deploy_cosmos_reason2(
            endpoint_name=args.endpoint_name,
            model_name=args.model_name,
            machine_type=args.machine_type,
            accelerator_type=args.accelerator_type,
            min_replicas=args.min_replicas,
            max_replicas=args.max_replicas,
        )
        print(f"\nDeployment complete!")
        print(f"Model ID: {model.resource_name}")
        print(f"Endpoint ID: {endpoint.resource_name}")
        print(f"\nAdd these to your .env file:")
        print(f"VERTEX_MODEL_ID={model.resource_name}")
        print(f"VERTEX_ENDPOINT_ID={endpoint.resource_name}")

    elif args.action == "check":
        if not args.endpoint_id:
            print("Error: --endpoint-id required for check action")
            return
        healthy = check_endpoint_health(args.endpoint_id)
        print(f"Endpoint healthy: {healthy}")


if __name__ == "__main__":
    main()
