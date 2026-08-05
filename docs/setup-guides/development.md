# Development Setup

## Local Environment

권장 기본 환경:

- Python 3.10+
- Git
- CUDA-capable GPU for training experiments
- Ubuntu 24.04 + ROS 2 Jazzy for ROS integration work
- Jetson Orin Nano 8GB for deployment validation

## Clone

```bash
git clone <repo-url>
cd ADOM-Autonomous-Driving-for-Off-Road-Military-vehicles
```

## Python Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## Data Policy

Do not commit datasets, generated labels, checkpoints, exported ONNX files, TensorRT engines, ROS bags, or experiment outputs.

Use:

- `data/` for local dataset placement conventions
- `models/` for local checkpoint/export placement conventions
- `experiments/<name>/` for scripts, configs, and result summaries

## First Setup Tasks

1. Decide the first dataset split convention.
2. Decide the first baseline command format.
3. Create the first benchmark result row in `results/benchmark-results.csv`.
4. Keep ROS2 integration code under `ros2_ws/` in this mono repo.
