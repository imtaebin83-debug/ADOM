# NGC PyTorch 23.10-py3 base image with OpenMMLab dependencies installed.
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set environment variables
# to avoid interactive prompts during package installation
# and to ensure Python output is unbuffered.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install OS System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        unzip \
        vim \
        nano \
        htop \
        tmux \
        build-essential \
        ninja-build \
        cmake \
        pkg-config \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        ffmpeg \
        ca-certificates \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/openmmlab.txt /tmp/requirements/openmmlab.txt

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir "openmim==0.3.9" \
    && mim install "mmengine>=0.7.4,<1.0.0"

# Prefer the OpenMMLab prebuilt MMCV wheel. If no wheel matches this specific
# NGC PyTorch/CUDA combination, fallback to compiling from source automatically.
RUN if ! mim install "mmcv>=2.0.0,<2.2.0"; then \
        echo "No compatible prebuilt MMCV wheel was found. Falling back to source build (this will take 15-20 minutes)..."; \
        MMCV_WITH_OPS=1 python -m pip install --no-cache-dir "mmcv>=2.0.0,<2.2.0"; \
    fi \
    && python -m pip install --no-cache-dir -r /tmp/requirements/openmmlab.txt \
    && rm -rf /root/.cache/pip /root/.cache/openmim

WORKDIR /workspace/adom/repo

# Training data and generated artifacts are mounted at runtime;
# they are never copied into the image.
# Sanity Check
RUN python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())" \
    && python -c "import mmcv, mmengine, mmseg, onnx; print('All libraries imported successfully.')"

CMD ["bash"]
