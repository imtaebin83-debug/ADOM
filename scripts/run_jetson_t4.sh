#!/usr/bin/env bash

# ROS 2 setup files read optional AMENT variables that may be unset. Do not
# enable nounset in this wrapper; required ADOM variables use guarded expansion.

adom_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
adom_repo="${ADOM_REPO:-$(cd "$adom_script_dir/.." && pwd)}"
adom_model_config="$adom_repo/configs/adom/export/segformer_b0_640x384_rellis3d.py"
adom_checkpoint_root="${ADOM_CHECKPOINT_ROOT:-$adom_repo/models/checkpoints/b0-e0}"

fail() {
    echo "ERROR: $*" >&2
    return 1
}

if [[ ! -r "$adom_model_config" ]]; then
    fail "B0-E0 model config를 읽을 수 없습니다: $adom_model_config"
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
        fail "B0-E0 checkpoint가 없습니다. perception 팀의 best_mIoU_iter_6000.pth를 다음 위치에 복사하세요: $adom_checkpoint_root/"
        echo "RunPod 기록 위치: /workspace/adom/runs/semantic20/e0/20260805T122006Z-5c50bfdf2900-b0-full/b0/stage2/best_mIoU_iter_6000.pth" >&2
        exit 1
    fi
    if [[ ${#adom_candidates[@]} -ne 1 ]]; then
        printf 'ERROR: B0-E0 checkpoint 후보가 여러 개입니다. ADOM_CHECKPOINT로 하나를 지정하세요:\n' >&2
        printf '  %s\n' "${adom_candidates[@]}" >&2
        exit 1
    fi
    adom_checkpoint="${adom_candidates[0]}"
fi

if [[ ! -r "$adom_checkpoint" ]]; then
    fail "B0-E0 checkpoint를 읽을 수 없습니다: $adom_checkpoint"
    exit 1
fi

export ADOM_REPO="$adom_repo"
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

echo "B0-E0 config: $ADOM_MODEL_CONFIG"
echo "B0-E0 checkpoint: $ADOM_CHECKPOINT"

exec ros2 launch adom_perception_ros perception.launch.py \
    model_config:="$ADOM_MODEL_CONFIG" \
    checkpoint:="$ADOM_CHECKPOINT" \
    device:=cuda:0
