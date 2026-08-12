#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="${1:-.}"
OUTPUT_DIR="${2:-${PACKAGE_DIR}/jetson-validation}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
python3 -m adom.runtime.semantic20_tensorrt \
  --engine "${PACKAGE_DIR}/model_static_1x3x384x640_fp16.engine" \
  --reference-io "${PACKAGE_DIR}/reference_io" \
  --source-parity-report "${PACKAGE_DIR}/pytorch_onnx_parity.json" \
  --palette "${PACKAGE_DIR}/palette.json" \
  --output-dir "${OUTPUT_DIR}" \
  --minimum-images 10 \
  --minimum-agreement 0.99 \
  --maximum-area-difference-pp 0.2 \
  --visualization-count 3 \
  --warmup 10 \
  --benchmark-iterations 100
