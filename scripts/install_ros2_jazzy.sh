#!/usr/bin/env bash
set -euo pipefail

source /etc/os-release

if [[ "${VERSION_CODENAME:-}" != "noble" ]]; then
  echo "ROS 2 Jazzy deb packages require Ubuntu 24.04 (noble)." >&2
  echo "Detected: ${PRETTY_NAME:-unknown} (${VERSION_CODENAME:-unknown})" >&2
  echo "For Jetson Orin, install JetPack 7.2 before running this script." >&2
  exit 1
fi

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This installer is intended for a 64-bit ARM Jetson (aarch64)." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y locales software-properties-common curl
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

sudo add-apt-repository universe -y

ROS_APT_RELEASE="$(
  curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
    | awk -F'"' '/tag_name/ {print $4; exit}'
)"

if [[ -z "${ROS_APT_RELEASE}" ]]; then
  echo "Could not determine the latest ros-apt-source release." >&2
  exit 1
fi

curl -fsSL -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_RELEASE}/ros2-apt-source_${ROS_APT_RELEASE}.noble_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

sudo apt-get update
sudo apt-get install -y \
  ros-jazzy-ros-base \
  ros-dev-tools \
  ros-jazzy-ackermann-msgs \
  ros-jazzy-joy \
  ros-jazzy-navigation2 \
  ros-jazzy-nav2-bringup \
  ros-jazzy-nmea-navsat-driver \
  ros-jazzy-robot-localization \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-xacro

if [[ ! -e /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
  sudo rosdep init
fi
rosdep update

echo
echo "ROS 2 Jazzy installed. Verify with:"
echo "  source /opt/ros/jazzy/setup.bash"
echo "  ros2 doctor --report"
