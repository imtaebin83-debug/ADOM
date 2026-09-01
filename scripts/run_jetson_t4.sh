#!/usr/bin/env bash

# ROS 2 setup files read optional AMENT variables that may be unset. Do not
# enable nounset in this wrapper; required ADOM variables use guarded expansion.

adom_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
adom_repo="${ADOM_REPO:-$(cd "$adom_script_dir/.." && pwd)}"
adom_profile="${1:-}"

fail() {
    echo "ERROR: $*" >&2
    return 1
}

if [[ $# -ne 1 || ( "$adom_profile" != "b0-e0" && "$adom_profile" != "eadom" ) ]]; then
    echo "Usage: $0 {b0-e0|eadom}" >&2
    exit 2
fi

case "$adom_profile" in
    b0-e0)
        adom_model_config_default="$adom_repo/configs/adom/runtime/segformer_b0_640x384_rellis3d.py"
        adom_checkpoint_root_default="$adom_repo/models/checkpoints/b0-e0"
        adom_checkpoint_sha_default="d76229ff623eb382fd48011decf54c342d88a113bcbe650fb58cc20e42cabe73"
        adom_runpod_checkpoint="/workspace/adom/runs/semantic20/e0/20260805T122006Z-5c50bfdf2900-b0-full/b0/stage2/best_mIoU_iter_6000.pth"
        ;;
    eadom)
        adom_model_config_default="$adom_repo/configs/adom/runtime/segformer_b0_640x384_eadom.py"
        adom_checkpoint_root_default="$adom_repo/models/checkpoints/eadom"
        adom_checkpoint_sha_default="f4cc41fd91e9df8e7aa3f726498e80636b736dfadf0e1baf338fe7c82a83399c"
        adom_runpod_checkpoint="/workspace/adom/artifacts/eadom-b0-seed42-iter26000/checkpoint.pth"
        ;;
esac

# A named profile owns its runtime config. In particular, allowing a stale
# export config through ADOM_MODEL_CONFIG can reintroduce the 640x360 padding
# distortion that the runtime config fixes. Keep the variable only as output
# for the ROS launch below; callers may override checkpoint location, not the
# profile's preprocessing contract.
if [[ -n "${ADOM_MODEL_CONFIG:-}" && "$ADOM_MODEL_CONFIG" != "$adom_model_config_default" ]]; then
    echo "WARNING: ignoring ADOM_MODEL_CONFIG for locked profile $adom_profile: $ADOM_MODEL_CONFIG" >&2
fi
adom_model_config="$adom_model_config_default"
adom_checkpoint_root="${ADOM_CHECKPOINT_ROOT:-$adom_checkpoint_root_default}"
adom_expected_checkpoint_sha="${ADOM_EXPECTED_CHECKPOINT_SHA256:-$adom_checkpoint_sha_default}"

if [[ ! -r "$adom_model_config" ]]; then
    fail "$adom_profile model config를 읽을 수 없습니다: $adom_model_config"
    exit 1
fi

adom_checkpoint="${ADOM_CHECKPOINT:-}"
if [[ -z "$adom_checkpoint" || "$adom_checkpoint" == *"<CHECKPOINT_"* ]]; then
    adom_candidates=()
    if [[ -d "$adom_checkpoint_root" ]]; then
        while IFS= read -r -d '' candidate; do
            adom_candidates+=("$candidate")
        done < <(find "$adom_checkpoint_root" -maxdepth 2 -type f -name '*.pth' -print0 | sort -z)
    fi

    if [[ ${#adom_candidates[@]} -eq 0 ]]; then
        fail "$adom_profile checkpoint가 없습니다. 정확히 한 개의 .pth를 다음 위치에 복사하세요: $adom_checkpoint_root/"
        echo "RunPod 기록 위치: $adom_runpod_checkpoint" >&2
        exit 1
    fi
    if [[ ${#adom_candidates[@]} -ne 1 ]]; then
        printf 'ERROR: %s checkpoint 후보가 여러 개입니다. ADOM_CHECKPOINT로 하나를 지정하세요:\n' "$adom_profile" >&2
        printf '  %s\n' "${adom_candidates[@]}" >&2
        exit 1
    fi
    adom_checkpoint="${adom_candidates[0]}"
fi

if [[ ! -r "$adom_checkpoint" ]]; then
    fail "$adom_profile checkpoint를 읽을 수 없습니다: $adom_checkpoint"
    exit 1
fi

adom_actual_checkpoint_sha="$(sha256sum "$adom_checkpoint" | awk '{print $1}')" || exit 1
if [[ "$adom_actual_checkpoint_sha" != "$adom_expected_checkpoint_sha" ]]; then
    fail "$adom_profile checkpoint SHA256 mismatch: actual=$adom_actual_checkpoint_sha expected=$adom_expected_checkpoint_sha"
    exit 1
fi

# MMEngine training checkpoints include trusted metadata such as
# HistoryBuffer. PyTorch 2.6+ defaults torch.load() to weights_only=True, while
# the pinned MMEngine call site does not pass the argument. Only the two
# repository-reviewed canonical hashes receive the compatibility override;
# custom artifacts remain on PyTorch's safer default unless the operator makes
# an explicit environment choice.
adom_checkpoint_load_mode="pytorch-default"
if [[ "$adom_actual_checkpoint_sha" == "$adom_checkpoint_sha_default" ]]; then
    unset TORCH_FORCE_WEIGHTS_ONLY_LOAD
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    adom_checkpoint_load_mode="trusted-canonical-mmengine"
fi

export ADOM_REPO="$adom_repo"
export ADOM_MODEL_PROFILE="$adom_profile"
export ADOM_MODEL_CONFIG="$adom_model_config"
export ADOM_CHECKPOINT="$adom_checkpoint"
export PYTHONPATH="$adom_repo/src${PYTHONPATH:+:$PYTHONPATH}"

source /opt/ros/jazzy/setup.bash || exit 1

cd "$adom_repo/ros2_ws" || exit 1

colcon build \
    --symlink-install \
    --packages-select adom_perception_ros \
    --allow-overriding adom_perception_ros || exit 1

source "$adom_repo/ros2_ws/install/setup.bash" || exit 1

cd "$adom_repo" || exit 1

python3 -c 'import torch, mmcv, mmseg, adom; print("Perception imports: OK")' || exit 1

echo "Model profile: $ADOM_MODEL_PROFILE"
echo "Model config: $ADOM_MODEL_CONFIG"
echo "Checkpoint: $ADOM_CHECKPOINT"
echo "Checkpoint SHA256: $adom_actual_checkpoint_sha"
echo "Checkpoint load mode: $adom_checkpoint_load_mode"

exec ros2 launch adom_perception_ros perception.launch.py \
    model_config:="$ADOM_MODEL_CONFIG" \
    checkpoint:="$ADOM_CHECKPOINT" \
    device:=cuda:0
