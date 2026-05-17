#!/bin/bash
# 启动 VLA 推理节点（控制器已由 setup_sim.sh 启动）
cd /home/wh/hb/openvla

# 初始化 conda（即使 PATH 被清理也能用）
source /home/wh/anaconda3/etc/profile.d/conda.sh
conda activate openvla

source /opt/ros/humble/setup.bash
export PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="/opt/ros/humble/lib:$LD_LIBRARY_PATH"

echo "=== 启动 VLA 推理节点 ==="
echo "  确保控制器已运行（setup_sim.sh 会自动启动）"
echo "  按 Ctrl+C 停止推理"
echo ""

python3 ros2/vla_ros_node.py --instruction "pick up the red block"
