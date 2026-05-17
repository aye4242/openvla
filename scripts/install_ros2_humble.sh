#!/bin/bash
# ROS2 Humble 安装脚本 (Ubuntu 22.04 Jammy)

set -e

echo "=== [1/4] 设置 locale ==="
sudo apt update
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

echo "=== [2/4] 添加 ROS2 apt 仓库 ==="
sudo apt install -y software-properties-common
sudo add-apt-repository -y universe
sudo apt update
sudo apt install -y curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

echo "=== [3/4] 安装 ROS2 Humble（桌面版 ~1.5GB）==="
sudo apt update
sudo apt install -y ros-humble-desktop

echo "=== [4/4] 安装编译工具 ==="
sudo apt install -y python3-colcon-common-extensions

echo ""
echo "========================================="
echo "  ROS2 Humble 安装完成！"
echo "========================================="
echo ""
echo "请运行以下命令验证："
echo "  source /opt/ros/humble/setup.bash"
echo "  ros2 --version"
