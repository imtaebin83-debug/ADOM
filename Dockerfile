# NGC PyTorch 23.10-py3 base image with OpenMMLab dependencies installed.
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set environment variables
# to avoid interactive prompts during package installation
# and to ensure Python output is unbuffered.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/adom/src

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
COPY requirements/opencv-headless-compat /tmp/requirements/opencv-headless-compat

# Pin the packaging and numerical stacks before installing OpenMMLab.
# NumPy 1.24.4 is the common compatible version for the NGC image's numba,
# CuPy, SciPy, pandas, ONNX Runtime, and OpenCV packages.
# Several NGC packages lack an uninstallable RECORD, so a regular upgrade
# leaves their old module files behind. Overwrite these pure-Python modules.
RUN python -m pip install --no-cache-dir --upgrade \
        "pip==24.0" \
        "wheel==0.41.2" \
        "setuptools==69.5.1" \
        "numpy==1.24.4" \
        "Cython==3.0.12" \
        "requests==2.31.0" \
        "urllib3==1.26.18" \
    && python -m pip install --no-cache-dir --ignore-installed --no-deps \
        "requests==2.31.0" \
        "prettytable==3.9.0" \
        "typing-extensions==4.8.0" \
        "urllib3==1.26.18"

# The OpenCV wheel variants all own the same cv2 namespace. Remove every
# preinstalled variant before installing exactly one pinned variant below.
# RAPIDS is not used by ADOM and conflicts irreconcilably with mmdeploy's
# protobuf<=3.20.2 requirement.
# Install the pinned application wheels without dependency resolution first,
# so mmdeploy/mmseg cannot pull unpinned MMEngine, MMCV, NumPy, or OpenCV.
RUN python -m pip uninstall -y \
        opencv-python \
        opencv-python-headless \
        opencv-contrib-python \
        opencv-contrib-python-headless \
        cudf \
        cugraph \
        cugraph-service-client \
        cugraph-service-server \
        cuml \
        dask-cuda \
        dask-cudf \
        raft-dask \
    && rm -rf \
        /usr/local/lib/python3.10/dist-packages/cv2 \
        /usr/local/lib/python3.10/dist-packages/opencv_python.libs \
        /usr/local/lib/python3.10/dist-packages/opencv_python_headless.libs \
    && python -m pip install --no-cache-dir --no-deps -r /tmp/requirements/openmmlab.txt \
    && python -m pip install --no-cache-dir --no-deps "mmengine==0.10.7"

# Prefer an OpenMMLab binary wheel without allowing pip to mutate dependencies.
# NGC 23.10 uses PyTorch 2.1 and CUDA 12.2; if no matching wheel exists, build
# MMCV 2.1.0 from source with bounded parallelism and the pinned build stack.
RUN if ! python -m pip install --no-cache-dir --no-deps \
        --only-binary=mmcv \
        --find-links https://download.openmmlab.com/mmcv/dist/cu122/torch2.1.0/index.html \
        "mmcv==2.1.0"; then \
        echo "No compatible prebuilt MMCV wheel was found. Falling back to source build (this will take 15-20 minutes)..."; \
        MAX_JOBS=2 MMCV_WITH_OPS=1 python -m pip install \
            --no-cache-dir \
            --no-build-isolation \
            --no-deps \
            "mmcv==2.1.0"; \
    fi

# Resolve the remaining runtime dependencies only after every ABI-sensitive
# package is present at its pinned version. Validate that dependency graph,
# then add Albumentations without its second OpenCV wheel.
RUN python -m pip uninstall -y cugraph-dgl \
    && python -m pip install --no-cache-dir -r /tmp/requirements/openmmlab.txt \
    && python -m pip check \
    && python -m pip install --no-cache-dir --no-build-isolation --no-deps \
        "qudida==0.0.4" \
        "albumentations==1.3.1" \
        /tmp/requirements/opencv-headless-compat \
    && python -m pip check \
    && rm -rf /root/.cache/pip

# Dependency and ABI sanity checks. CUDA availability is checked at RunPod
# runtime because image builds do not have access to a GPU.
RUN python -c "import torch; print(torch.__version__)" \
    && python -c "import albumentations, cv2, numpy, pandas, scipy, onnxruntime; assert cv2.__version__ == '4.8.0'; print('Core scientific libraries imported successfully.')" \
    && python -c "import mmcv, mmcv.ops, mmengine, mmseg, mmdeploy, onnx; print('All OpenMMLab libraries imported successfully.')" \
    && python -c "import ftfy, regex; from prettytable import PrettyTable; from mmseg.datasets import BaseSegDataset; print('MMSeg dataset runtime imports successful.')" \
    && python -c "from importlib.metadata import version; expected={'numpy':'1.24.4','setuptools':'69.5.1','opencv-python':'4.8.0.76','opencv-python-headless':'4.8.0.76','mmcv':'2.1.0','mmengine':'0.10.7','mmsegmentation':'1.2.2','mmdeploy':'1.3.1','wandb':'0.22.3','ftfy':'6.1.1','regex':'2023.10.3','prettytable':'3.9.0'}; actual={k:version(k) for k in expected}; assert actual==expected, (actual, expected); print(actual)"

# Stamp the final image without invalidating the expensive dependency layers
# for every source commit.
ARG ADOM_GIT_SHA=unknown
LABEL org.opencontainers.image.revision=${ADOM_GIT_SHA}
ENV ADOM_GIT_SHA=${ADOM_GIT_SHA}

# Keep application code outside /workspace because RunPod mounts the shared
# Network Volume over /workspace. This makes each SHA-tagged image runnable
# without a second git clone inside the Pod.
WORKDIR /opt/adom
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY data ./data
RUN python -m pip install --no-cache-dir --no-deps . \
    && python -c "import adom.mmseg; print('ADOM MMSeg extensions imported successfully.')" \
    && python -m adom.data.semantic20 --help > /dev/null \
    && python -m adom.data.target_adaptation --help > /dev/null \
    && python -m adom.data.transform_audit --help > /dev/null \
    && python -m adom.runtime.semantic20_contract --help > /dev/null \
    && python scripts/check_ta0_config_imports.py \
    && python -c "from adom.data.semantic20 import resource_path; expected={'train':4435,'val':900,'test':899}; actual={s:len([v for v in resource_path('rellis','splits',s+'.txt').read_text(encoding='utf-8-sig').splitlines() if v.strip()]) for s in expected}; assert actual==expected,(actual,expected); assert resource_path('rugd','config','label_mapping.json').is_file(); assert resource_path('semantic_20','config','bridge_mapping.yaml').is_file(); print('Semantic20 preprocessing assets verified:', actual)"

CMD ["sleep", "infinity"]
