#!/bin/bash
# Gazebo 11 + ROS2 Humble 仿真环境安装脚本

set -e

echo "=== [1/3] 安装 Gazebo 11 + ROS2 桥接 ==="
sudo apt update
sudo apt install -y ros-humble-gazebo-ros-pkgs

echo "=== [2/3] 安装 ROS2 控制器框架 ==="
sudo apt install -y \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-gazebo-ros2-control

echo "=== [3/3] 安装额外工具 ==="
sudo apt install -y \
    ros-humble-xacro \
    ros-humble-joint-state-publisher \
    ros-humble-robot-state-publisher

echo ""
echo "========================================="
echo "  Gazebo + ROS2 安装完成！"
echo "========================================="
echo ""
echo "验证命令："
echo "  source /opt/ros/humble/setup.bash"
echo "  gazebo --version"
