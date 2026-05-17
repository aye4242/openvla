#!/bin/bash
# Franka Panda 机械臂模型 + 工具安装

set -e

echo "=== 安装 Franka Panda ROS2 包 ==="
sudo apt update
sudo apt install -y \
    ros-humble-franka-description \
    ros-humble-franka-gripper \
    ros-humble-franka-msgs

echo ""
echo "========================================="
echo "  Franka Panda 安装完成！"
echo "========================================="
echo ""
echo "验证 URDF："
echo "  source /opt/ros/humble/setup.bash"
echo "  ros2 run xacro xacro ros-humble-franka-description/urdf/panda.urdf.xacro"
