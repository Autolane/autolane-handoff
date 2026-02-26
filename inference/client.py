"""Client for Cosmos Reason 2-8B inference via Vertex AI rawPredict.

Sends video frames with Grounding DINO overlays to the Vertex AI endpoint
for two-pass reasoning: safety assessment + handoff planning.
"""

import asyncio
import json
import re
import time
from typing import Any, Optional

import httpx
import structlog
from google.auth import default
from google.auth.transport.requests import Request
from google.cloud import aiplatform

from inference.config import settings
from inference.prompts import (
    MAX_TOKENS,
    MODEL_NAME,
    VIDEO_FPS,
    build_handoff_messages,
    build_safety_messages,
)

logger = structlog.get_logger()

_GCP_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class InferenceClient:
    """Client for sending video inference requests to the Vertex AI endpoint."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        endpoint_id: Optional[str] = None,
    ):
        self.project_id = project_id or settings.gcp_project_id
        self.region = region or settings.gcp_region
        self.endpoint_id = endpoint_id or settings.vertex_endpoint_id

        if not self.project_id:
            raise ValueError("GCP project ID is required. Set GCP_PROJECT_ID env var.")
        if not self.endpoint_id:
            raise ValueError("Vertex AI endpoint ID is required. Set VERTEX_ENDPOINT_ID env var.")

        # Extract numeric endpoint ID from full resource path
        if "/" in self.endpoint_id:
            match = re.search(r"endpoints/([^/]+)$", self.endpoint_id)
            if match:
                self.endpoint_id = match.group(1)

        aiplatform.init(project=self.project_id, location=self.region)
        self._endpoint: Optional[aiplatform.Endpoint] = None

    def _get_endpoint_url(self) -> str:
        """Build the rawPredict URL for the endpoint."""
        return (
            f"https://{self.region}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{self.region}/"
            f"endpoints/{self.endpoint_id}:rawPredict"
        )

    def _get_model_identifier(self) -> str:
        """Return the model name for vLLM payloads."""
        if settings.use_finetuned_model and settings.finetuned_model_gcs_uri:
            return "/model"
        return settings.model_name

    def _get_auth_token(self) -> str:
        """Get a fresh OAuth2 token."""
        credentials, _ = default(scopes=_GCP_SCOPES)
        credentials.refresh(Request())
        return credentials.token

    async def _get_auth_headers(self) -> dict:
        """Get auth headers without blocking the event loop."""
        loop = asyncio.get_event_loop()
        token = await loop.run_in_executor(None, self._get_auth_token)
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Build the vLLM API request payload.

        Args:
            messages: Chat messages with video content.
            temperature: Sampling temperature.
            max_tokens: Maximum generation tokens.

        Returns:
            Payload dict for rawPredict.
        """
        return {
            "model": self._get_model_identifier(),
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.temperature,
            "max_tokens": max_tokens if max_tokens is not None else MAX_TOKENS,
        }

    async def _send_request(
        self,
        messages: list[dict[str, Any]],
        headers: dict,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: float = 180.0,
    ) -> str:
        """Send a request to the Vertex AI endpoint and return raw text.

        Args:
            messages: Chat messages with video content.
            headers: Auth headers.
            temperature: Sampling temperature.
            max_tokens: Maximum generation tokens.
            timeout: Request timeout in seconds.

        Returns:
            Raw text response from the model.

        Raises:
            httpx.HTTPStatusError: If the request fails.
            ValueError: If the model returns no choices.
        """
        payload = self._build_payload(messages, temperature, max_tokens)
        endpoint_url = self._get_endpoint_url()

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(endpoint_url, json=payload, headers=headers)
            response.raise_for_status()

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("Model returned no choices")

        return choices[0].get("message", {}).get("content", "")

    async def assess_safety(
        self,
        video_path: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: float = 180.0,
    ) -> dict[str, Any]:
        """Run Pass 1: Safety + Compliance assessment on a video.

        Args:
            video_path: Path to annotated video (with GDINO overlays).
            temperature: Sampling temperature override.
            max_tokens: Max tokens override.
            timeout: Request timeout in seconds.

        Returns:
            Dict with 'raw_response', 'think', 'answer', and 'processing_time_ms'.
        """
        start_time = time.time()
        headers = await self._get_auth_headers()
        messages = build_safety_messages(video_path)

        logger.info("running_safety_assessment", video=video_path)

        raw_response = await self._send_request(
            messages, headers, temperature, max_tokens, timeout
        )
        processing_time_ms = (time.time() - start_time) * 1000

        think, answer = self._parse_think_answer(raw_response)

        logger.info(
            "safety_assessment_complete",
            processing_time_ms=round(processing_time_ms),
        )

        return {
            "pass": "safety",
            "raw_response": raw_response,
            "think": think,
            "answer": answer,
            "processing_time_ms": processing_time_ms,
        }

    async def plan_handoff(
        self,
        video_path: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: float = 180.0,
    ) -> dict[str, Any]:
        """Run Pass 2: Handoff Planning on a video.

        Args:
            video_path: Path to annotated video (with GDINO overlays).
            temperature: Sampling temperature override.
            max_tokens: Max tokens override.
            timeout: Request timeout in seconds.

        Returns:
            Dict with 'raw_response', 'think', 'answer', and 'processing_time_ms'.
        """
        start_time = time.time()
        headers = await self._get_auth_headers()
        messages = build_handoff_messages(video_path)

        logger.info("running_handoff_planning", video=video_path)

        raw_response = await self._send_request(
            messages, headers, temperature, max_tokens, timeout
        )
        processing_time_ms = (time.time() - start_time) * 1000

        think, answer = self._parse_think_answer(raw_response)

        logger.info(
            "handoff_planning_complete",
            processing_time_ms=round(processing_time_ms),
        )

        return {
            "pass": "handoff",
            "raw_response": raw_response,
            "think": think,
            "answer": answer,
            "processing_time_ms": processing_time_ms,
        }

    async def analyze_video(
        self,
        video_path: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: float = 180.0,
    ) -> dict[str, Any]:
        """Run the full two-pass analysis on a video.

        Pass 1: Safety + Compliance
        Pass 2: Handoff Planning

        Args:
            video_path: Path to annotated video (with GDINO overlays).
            temperature: Sampling temperature override.
            max_tokens: Max tokens override.
            timeout: Per-pass request timeout in seconds.

        Returns:
            Dict with 'safety' and 'handoff' results plus total processing time.
        """
        start_time = time.time()

        safety = await self.assess_safety(video_path, temperature, max_tokens, timeout)
        handoff = await self.plan_handoff(video_path, temperature, max_tokens, timeout)

        total_ms = (time.time() - start_time) * 1000

        return {
            "video": video_path,
            "safety": safety,
            "handoff": handoff,
            "total_processing_time_ms": total_ms,
        }

    @staticmethod
    def _parse_think_answer(raw: str) -> tuple[str, str]:
        """Extract <think> and <answer> blocks from model output.

        Args:
            raw: Raw model response text.

        Returns:
            Tuple of (think_content, answer_content). Empty strings if not found.
        """
        think = ""
        answer = ""

        think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        if think_match:
            think = think_match.group(1).strip()

        answer_match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()

        return think, answer

    async def health_check(self) -> bool:
        """Check if the endpoint is healthy."""
        try:
            endpoint = aiplatform.Endpoint(self.endpoint_id)
            return bool(endpoint.traffic_split)
        except Exception as e:
            logger.error("health_check_failed", error=str(e))
            return False
