FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl ffmpeg git git-lfs tree wget && \
    rm -rf /var/lib/apt/lists/*

# uv (matches cosmos-predict2.5 Dockerfile)
COPY --from=ghcr.io/astral-sh/uv:0.8.12 /uv /uvx /usr/local/bin/
ENV UV_LINK_MODE=copy
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# just (matches cosmos-predict2.5 Dockerfile)
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | \
    bash -s -- --to /usr/local/bin --tag 1.42.4

# Clone cosmos-predict2.5 (shallow, skip LFS assets — we don't need example videos)
RUN git lfs install && \
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
        https://github.com/nvidia-cosmos/cosmos-predict2.5.git /workspace

WORKDIR /workspace

# Install cosmos-predict2.5 with CUDA 12.8 support (uses their tested lockfile)
RUN uv sync --locked --extra=cu128

# Install our additional deps into the same venv
RUN uv pip install pydantic-settings google-cloud-storage

# Activate the venv + enable CUDA forward compatibility for older drivers (Vertex AI A100 uses driver 535)
ENV PATH="/workspace/.venv/bin:$PATH"
ENV HF_HOME=/tmp/hf_cache
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH}

# Copy our datagen code
COPY datagen/ /app/datagen/

WORKDIR /app
ENTRYPOINT ["python", "-m", "datagen.run_vertex"]
