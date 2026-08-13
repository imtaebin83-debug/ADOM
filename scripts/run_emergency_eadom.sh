#!/usr/bin/env bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TA_ROOT="${EADOM_DATA_ROOT:-/workspace/adom/datasets/processed/adom_semantic20_target_adaptation_v1}"
OUT="${EADOM_RUN_ROOT:-/workspace/adom/runs/semantic20/eadom/seed42/full}"
LOG_ROOT="${EADOM_LOG_ROOT:-/workspace/adom/logs/eadom}"
IMAGE_SHA="${EADOM_IMAGE_SHA:-}"
MODE="${1:-full}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${TORCH_HOME:-/workspace/adom/cache/torch}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_PROJECT="${WANDB_PROJECT:-adom}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-eadom-emergency-seed42}"

fail() {
  echo "FAIL: $*" >&2
  return 1
}

preflight() {
  [ -n "$IMAGE_SHA" ] || fail "set EADOM_IMAGE_SHA to the full Git SHA" || return
  [ "${ADOM_GIT_SHA:-}" = "$IMAGE_SHA" ] ||
    fail "image SHA=${ADOM_GIT_SHA:-unset}, expected=$IMAGE_SHA" || return
  [ -f "$TA_ROOT/_SUCCESS" ] || fail "missing dataset _SUCCESS" || return
  [ -f "$TA_ROOT/splits/ta1_train.txt" ] || fail "missing ta1_train split" || return
  [ -f "$TA_ROOT/manifest.csv" ] || fail "missing manifest.csv" || return
  [ ! -e "$OUT" ] || [ "$MODE" = "resume" ] ||
    fail "output exists; use: $0 resume" || return
  mkdir -p "$LOG_ROOT"
}

run_cycle() {
  local log="$LOG_ROOT/eadom-seed42-full.log"
  local command=(
    python -m adom.runtime.semantic20_cycle
    --dataset "$TA_ROOT"
    --experiment eadom
    --models b0
    --output "$OUT"
    --gate full
    --micro-batch 16
    --seed 42
    --expected-image-sha "$IMAGE_SHA"
  )
  if [ "$MODE" = "resume" ]; then
    command+=(--resume)
  fi
  echo "LOG: $log"
  "${command[@]}" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  echo "E-ADOM exit code: $rc"
  return "$rc"
}

case "$MODE" in
  full|resume)
    cd "$REPO_ROOT" || exit 2
    preflight || exit 2
    run_cycle
    exit $?
    ;;
  *)
    echo "usage: $0 {full|resume}"
    exit 2
    ;;
esac
