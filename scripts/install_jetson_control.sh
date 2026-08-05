#!/usr/bin/env bash
set -eo pipefail

ROS_DISTRO_NAME="${ROS_DISTRO:-jazzy}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="${REPO_DIR}/ros2_ws"

if [[ ! -r "/opt/ros/${ROS_DISTRO_NAME}/setup.bash" ]]; then
  echo "ROS 2 ${ROS_DISTRO_NAME} is not installed. Install ROS first, then rerun." >&2
  exit 1
fi

# ROS setup files may read optional variables that are intentionally unset.
# Enable nounset only after the environment has been loaded.
source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
set -u

source /etc/os-release
if [[ "${ROS_DISTRO_NAME}" == "jazzy" && "${VERSION_CODENAME:-}" != "noble" ]]; then
  echo "ROS 2 Jazzy control requires Ubuntu 24.04 (noble)." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  "ros-${ROS_DISTRO_NAME}-ackermann-msgs" \
  "ros-${ROS_DISTRO_NAME}-joy" \
  i2c-tools \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-smbus

sudo usermod -aG i2c "${USER}"
sudo usermod -aG input "${USER}"

cd "${WORKSPACE_DIR}"
# Remove the entry point left by versions that still shipped keyboard teleop.
rm -f "${WORKSPACE_DIR}/install/adom_control/lib/adom_control/keyboard_teleop"
colcon build --symlink-install --packages-select adom_control

echo
echo "Control package installed. Log out and reconnect so i2c/input groups apply."
echo "Then source ${WORKSPACE_DIR}/install/setup.bash before running it."
