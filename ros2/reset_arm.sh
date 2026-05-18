#!/bin/bash
# 一键复位机械臂到 home，不用重启 Gazebo
pkill -9 -f panda_controller 2>/dev/null
source /opt/ros/humble/setup.bash
ros2 topic pub --once /forward_position_controller/commands std_msgs/msg/Float64MultiArray "{data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}" >/dev/null 2>&1
sleep 0.5
/usr/bin/python3 /home/wh/hb/openvla/ros2/panda_controller.py &>/tmp/panda_controller.log &
echo "Arm reset to home."
