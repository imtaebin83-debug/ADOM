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

# 1. Base Python packages: NumPy 2.0 C-API issue, so we limit NumPy to <2.0.0 for now.
# setuptools<70.0.0 is required for pkg_resources
RUN python -m pip install --no-cache-dir --upgrade pip wheel "setuptools<70.0.0" "numpy<2.0.0" Cython \
    && python -m pip install --no-cache-dir "openmim==0.3.9" \
    && mim install "mmengine>=0.7.4,<1.0.0"

# 2. Prefer the OpenMMLab prebuilt MMCV wheel...
RUN if ! mim install "mmcv>=2.0.0,<2.2.0"; then \
        echo "No compatible prebuilt MMCV wheel was found. Falling back to source build (this will take 15-20 minutes)..."; \
        # --no-build-isolation 옵션으로 격리 환경을 끄고 우리가 세팅한 setuptools<70.0.0을 사용하도록 강제합니다.
        MAX_JOBS=2 MMCV_WITH_OPS=1 python -m pip install --no-cache-dir --no-build-isolation "mmcv>=2.0.0,<2.2.0"; \
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
