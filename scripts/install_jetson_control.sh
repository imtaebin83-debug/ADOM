#!/usr/bin/env bash
set -euo pipefail

ROS_DISTRO_NAME="${ROS_DISTRO:-jazzy}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="${REPO_DIR}/ros2_ws"

if [[ ! -r "/opt/ros/${ROS_DISTRO_NAME}/setup.bash" ]]; then
  echo "ROS 2 ${ROS_DISTRO_NAME} is not installed. Install ROS first, then rerun." >&2
  exit 1
fi

source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"

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
  python3-pip \
  python3-rosdep \
  python3-smbus \
  python3-venv

CONTROL_VENV="${REPO_DIR}/.venv-control"
python3 -m venv --system-site-packages "${CONTROL_VENV}"
source "${CONTROL_VENV}/bin/activate"
python3 -m pip install --upgrade pip
python3 -m pip install \
  adafruit-blinka \
  adafruit-circuitpython-pca9685

sudo usermod -aG i2c "${USER}"
sudo usermod -aG input "${USER}"

cd "${WORKSPACE_DIR}"
colcon build --symlink-install --packages-select adom_control

echo
echo "Control package installed. Log out and reconnect so i2c/input groups apply."
echo "Then source ${WORKSPACE_DIR}/install/setup.bash before running it."
