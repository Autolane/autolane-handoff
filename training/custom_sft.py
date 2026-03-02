#!/usr/bin/env python3
"""Autolane Handoff SFT — Llava-format video post-training via cosmos-rl.

Adapted from nvidia-cosmos/cosmos-reason2 examples/cosmos_rl/scripts/llava_sft.py.

Trains Cosmos Reason 2 (2B or 8B) on curbside handoff QA pairs using the
ITS recipe SFT pipeline. Training data is in Llava format with annotated
video clips from the Grounding DINO detection stage.

Prerequisites:
    # Clone cosmos-reason2 and install cosmos-rl
    git clone https://github.com/nvidia-cosmos/cosmos-reason2.git
    cd cosmos-reason2/examples/cosmos_rl
    uv sync && source .venv/bin/activate

    # Install redis
    conda install -c conda-forge redis-server

Usage:
    # 2B fast iteration (~20 min on 4x A100)
    cosmos-rl --config sft_config_2b.toml --log-dir outputs/ custom_sft.py

    # 8B final training (~1-2 hours on 8x A100)
    cosmos-rl --config sft_config.toml --log-dir outputs/ custom_sft.py
"""

import argparse
import base64
import json
import os
import re
from pathlib import Path

import cosmos_rl.launcher.worker_entry
import cosmos_rl.policy.config
import pydantic
import toml
import torch.utils.data
from cosmos_reason2_utils.text import create_conversation
from cosmos_reason2_utils.vision import PIXELS_PER_TOKEN, VisionConfig
from cosmos_rl.utils.logging import logger


class CustomDatasetConfig(pydantic.BaseModel):
    """Dataset configuration for Llava-format training data."""

    annotation_path: str = pydantic.Field()
    media_path: str = pydantic.Field(default="")
    system_prompt: str = pydantic.Field(default="")


class CustomConfig(pydantic.BaseModel):
    """Custom config block from the TOML config file."""

    dataset: CustomDatasetConfig = pydantic.Field()
    vision: VisionConfig = pydantic.Field(default=VisionConfig(fps=1))


class AutolaneHandoffDataset(torch.utils.data.Dataset):
    """Llava-format video dataset for curbside handoff SFT.

    Each sample has:
      - id: clip identifier (e.g. "IMG_2700_i2w_package_transfer_safety")
      - video: relative path to annotated video (e.g. "data/annotations/.../annotated.mp4")
      - conversations: list of {"from": "human"/"gpt", "value": "..."}

    The video path is joined with media_path from the config to form the
    full path to the video file on disk.
    """

    def __init__(self, config, custom_config: CustomConfig):
        self.annotation = json.load(open(custom_config.dataset.annotation_path))
        self.media_path = custom_config.dataset.media_path
        self.system_prompt = custom_config.dataset.system_prompt
        self.config = config
        self.custom_config = custom_config
        self.vision_kwargs = custom_config.vision.model_dump(exclude_none=True)

        logger.info(
            "Loaded %d training samples from %s",
            len(self.annotation),
            custom_config.dataset.annotation_path,
        )

    def __len__(self) -> int:
        return len(self.annotation)

    def __getitem__(self, idx: int) -> list[dict]:
        sample = self.annotation[idx]
        user_prompt = sample["conversations"][0]["value"]
        response = sample["conversations"][1]["value"]

        # Handle image and video fields
        images = sample.get("image", None) or sample.get("images", None)
        if images and isinstance(images, str):
            images = [images]
        videos = sample.get("video", None)
        if videos and isinstance(videos, str):
            videos = [videos]

        # Prepend media_path to resolve full paths
        if self.media_path:
            if images:
                images = [os.path.join(self.media_path, img) for img in images]
            if videos:
                videos = [os.path.join(self.media_path, vid) for vid in videos]

        # cosmos-rl expects base64-encoded images
        if images:
            for i, image in enumerate(images):
                images[i] = base64.b64encode(open(image, "rb").read())

        # Strip <image>/<video> tags from user prompt — cosmos-rl handles media
        user_prompt = re.sub(r"(\n)?</?(image|video)>(\n)?", "", user_prompt)

        conversations = create_conversation(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            response=response,
            images=images,
            videos=videos,
            vision_kwargs=self.vision_kwargs,
        )
        return conversations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, required=True, help="Path to TOML config file.",
    )
    args = parser.parse_known_args()[0]

    with open(args.config) as f:
        config_kwargs = toml.load(f)

    config = cosmos_rl.policy.config.Config.from_dict(config_kwargs)
    custom_config = CustomConfig.model_validate(config_kwargs["custom"])

    # Scale vision total_pixels to fit within model_max_length budget
    custom_config.vision.total_pixels = int(
        config.policy.model_max_length * PIXELS_PER_TOKEN * 0.9
    )

    # Save resolved config for reproducibility
    role = os.environ.get("COSMOS_ROLE")
    is_controller = role == "Controller"
    if is_controller:
        output_dir = Path(config.train.output_dir).resolve().parent
        output_dir.mkdir(parents=True, exist_ok=True)
        config_kwargs_resolved = config.model_dump()
        config_kwargs_resolved["custom"] = custom_config.model_dump()
        config_path = output_dir / "config.toml"
        config_path.write_text(toml.dumps(config_kwargs_resolved))
        logger.info("Saved resolved config to %s", config_path)

    # Build dataset and sanity-check first sample
    dataset = AutolaneHandoffDataset(config=config, custom_config=custom_config)
    logger.info("Sanity-checking first sample...")
    dataset[0]
    logger.info("Dataset sanity check passed.")

    # Launch cosmos-rl training worker
    cosmos_rl.launcher.worker_entry.main(dataset=dataset)
