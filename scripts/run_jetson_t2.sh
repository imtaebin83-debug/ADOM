#!/usr/bin/env bash

# ROS 2 setup files may read optional unset variables, so do not enable nounset.

adom_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
adom_repo="${ADOM_REPO:-$(cd "$adom_script_dir/.." && pwd)}"
adom_mode="${1:-}"

if [[ $# -gt 1 || ( -n "$adom_mode" && "$adom_mode" != "mask" && "$adom_mode" != "evidence" && "$adom_mode" != "preview" ) ]]; then
    echo "Usage: t2 [mask|evidence|preview]" >&2
    echo "  t2       class statistics/status without raster mask" >&2
    echo "  t2 mask  additionally records the 2 Hz Semantic20 evidence mask" >&2
    echo "  t2 evidence  records full-rate source RGB and the 2 Hz Semantic20 mask" >&2
    echo "  t2 preview  records the 2 Hz Semantic20 mask and 45%-alpha RGB overlay" >&2
    exit 2
fi

adom_record_mask=false
adom_record_evidence=false
adom_record_preview=false
if [[ "$adom_mode" == "mask" || "$adom_mode" == "evidence" || "$adom_mode" == "preview" ]]; then
    adom_record_mask=true
fi
if [[ "$adom_mode" == "evidence" ]]; then
    adom_record_evidence=true
fi
if [[ "$adom_mode" == "preview" ]]; then
    adom_record_preview=true
fi

source /opt/ros/jazzy/setup.bash || exit 1
cd "$adom_repo/ros2_ws" || exit 1

colcon build --symlink-install --packages-up-to adom_logging || exit 1
source "$adom_repo/ros2_ws/install/setup.bash" || exit 1

export ADOM_REPO_ROOT="$adom_repo"

echo "Autonomy rosbag record_mask=$adom_record_mask record_evidence=$adom_record_evidence record_preview=$adom_record_preview"
exec ros2 launch adom_logging autonomy_logging.launch.py \
    capture_root:=data/autonomy_bags \
    record_mask:="$adom_record_mask" \
    record_evidence:="$adom_record_evidence" \
    record_preview:="$adom_record_preview"
