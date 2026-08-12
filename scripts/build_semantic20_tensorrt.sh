#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH="${1:-model_static_1x3x384x640_fp32.onnx}"
ENGINE_PATH="${2:-model_static_1x3x384x640_fp16.engine}"
WORKSPACE_MIB="${ADOM_TRT_WORKSPACE_MIB:-2048}"

if ! [[ "${WORKSPACE_MIB}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: ADOM_TRT_WORKSPACE_MIB must be a positive MiB integer" >&2
  exit 2
fi
if ! test -s "${ONNX_PATH}"; then
  echo "ERROR: ONNX file is missing or empty: ${ONNX_PATH}" >&2
  exit 2
fi
if test -e "${ENGINE_PATH}"; then
  echo "ERROR: refusing to overwrite engine: ${ENGINE_PATH}" >&2
  exit 2
fi

TRTEXEC="${TRTEXEC:-$(command -v trtexec || true)}"
if test -z "${TRTEXEC}" && test -x /usr/src/tensorrt/bin/trtexec; then
  TRTEXEC=/usr/src/tensorrt/bin/trtexec
fi
if test -z "${TRTEXEC}" || ! test -x "${TRTEXEC}"; then
  echo "ERROR: trtexec was not found" >&2
  exit 2
fi

"${TRTEXEC}" \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${ENGINE_PATH}" \
  --fp16 \
  --memPoolSize="workspace:${WORKSPACE_MIB}" \
  --skipInference \
  --monitorMemory

if ! test -s "${ENGINE_PATH}"; then
  echo "ERROR: TensorRT reported success without a non-empty engine" >&2
  exit 3
fi
sha256sum "${ENGINE_PATH}"
