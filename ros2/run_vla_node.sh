#!/bin/bash
# VLA ROS2 推理节点启动脚本

cd /home/wh/hb/openvla

# 激活 conda 环境
conda activate openvla

# 加载 ROS2 环境（顺序：先 conda 后 ROS2，确保 PYTHONPATH 包含两者）
source /opt/ros/humble/setup.bash

# 补充 ROS2 Python 路径（conda 可能覆盖了 PYTHONPATH）
export PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="/opt/ros/humble/lib:$LD_LIBRARY_PATH"

echo "Starting VLA ROS2 Node ..."
python3 ros2/vla_ros_node.py \
    --instruction "pick up the object" \
    --unnorm_key bridge_orig
