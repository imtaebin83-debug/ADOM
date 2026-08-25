#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${B2_EADOM_DATA_ROOT:-/workspace/adom/datasets/processed/adom_semantic20_target_adaptation_v1}"
RUN_ROOT="${B2_EADOM_RUN_ROOT:-/workspace/adom/runs/semantic20/eadom/seed42}"
LOG_ROOT="${B2_EADOM_LOG_ROOT:-/workspace/adom/logs/b2-eadom-capacity-domain}"
IMAGE_SHA="${B2_EADOM_IMAGE_SHA:-}"
MICRO_BATCH="${B2_EADOM_MICRO_BATCH:-}"
MODE="${1:-}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${TORCH_HOME:-/workspace/adom/cache/torch}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_PROJECT="${WANDB_PROJECT:-adom}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-b2-eadom-capacity-domain-seed42}"

fail() {
  echo "FAIL: $*" >&2
  exit 2
}

preflight() {
  [ "$REPO_ROOT" = "/opt/adom" ] || fail "RunPod code must be rooted at /opt/adom"
  [ -n "$IMAGE_SHA" ] || fail "set B2_EADOM_IMAGE_SHA to the full immutable image Git SHA"
  [ "${ADOM_GIT_SHA:-}" = "$IMAGE_SHA" ] ||
    fail "image SHA=${ADOM_GIT_SHA:-unset}, expected=$IMAGE_SHA"
  [ -f "$DATA_ROOT/_SUCCESS" ] || fail "missing dataset _SUCCESS"
  [ -f "$DATA_ROOT/manifest.csv" ] || fail "missing manifest.csv"
  [ -f "$DATA_ROOT/splits/ta1_train.txt" ] || fail "missing ta1_train split"
  mkdir -p "$LOG_ROOT" "$RUN_ROOT/protocol"
}

require_batch_plan() {
  [ -n "$MICRO_BATCH" ] || fail "set B2_EADOM_MICRO_BATCH from the passed probe batch_plan.json"
  case "$MICRO_BATCH" in
    16|8|4) ;;
    *) fail "B2_EADOM_MICRO_BATCH must be 16, 8, or 4" ;;
  esac
}

run_logged() {
  local name="$1"
  shift
  local log="$LOG_ROOT/${name}.log"
  echo "LOG: $log"
  "$@" 2>&1 | tee -a "$log"
}

run_contract() {
  require_batch_plan
  local accumulative=$((16 / MICRO_BATCH))
  run_logged contract \
    python -m adom.runtime.b2_eadom_contract \
    --dataset "$DATA_ROOT" \
    --output "$RUN_ROOT/protocol/static_contract.json" \
    --micro-batch "$MICRO_BATCH" \
    --accumulative-counts "$accumulative"
}

run_gate() {
  local gate="$1"
  local output="$RUN_ROOT/gates/$gate"
  [ ! -e "$output" ] || fail "gate output exists: $output"
  local command=(
    python -m adom.runtime.semantic20_cycle
    --dataset "$DATA_ROOT"
    --experiment eadom
    --models b2
    --output "$output"
    --gate "$gate"
    --require-gpu-name "RTX 4090"
    --minimum-gpu-memory-gib 22
    --seed 42
    --expected-image-sha "$IMAGE_SHA"
  )
  if [ "$gate" != "probe" ]; then
    require_batch_plan
    command+=(--micro-batch "$MICRO_BATCH")
  fi
  run_logged "$gate" "${command[@]}"
}

run_full() {
  require_batch_plan
  local output="$RUN_ROOT/full"
  local command=(
    python -m adom.runtime.semantic20_cycle
    --dataset "$DATA_ROOT"
    --experiment eadom
    --models b2
    --output "$output"
    --gate full
    --micro-batch "$MICRO_BATCH"
    --require-gpu-name "RTX 4090"
    --minimum-gpu-memory-gib 22
    --seed 42
    --expected-image-sha "$IMAGE_SHA"
  )
  if [ "$MODE" = "resume" ]; then
    command+=(--resume)
  else
    [ ! -e "$output" ] || fail "full output exists; use resume"
  fi
  run_logged "$MODE" "${command[@]}"
}

cd "$REPO_ROOT"
preflight
case "$MODE" in
  probe|smoke|mini)
    run_gate "$MODE"
    ;;
  resume-gate)
    run_gate resume
    ;;
  contract)
    run_contract
    ;;
  full|resume)
    run_full
    ;;
  *)
    fail "usage: $0 {probe|contract|smoke|mini|resume-gate|full|resume}"
    ;;
esac
