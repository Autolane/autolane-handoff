FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl ffmpeg git python3 python3-pip python3-venv wget && \
    rm -rf /var/lib/apt/lists/*

# Create venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python deps
RUN pip install --no-cache-dir \
    groundingdino-py \
    "transformers>=4.41,<5.0" \
    opencv-python-headless \
    Pillow \
    numpy \
    "torch==2.6.0" \
    "torchvision==0.21.0" \
    pydantic-settings \
    google-cloud-storage

# Download Grounding DINO SwinT weights
RUN mkdir -p /weights && \
    wget -q -O /weights/groundingdino_swint_ogc.pth \
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

# CUDA forward compatibility for Vertex AI A100 driver
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH}

# Copy detection code
COPY detection/ /app/detection/

WORKDIR /app
ENTRYPOINT ["python", "-m", "detection.run_vertex"]
