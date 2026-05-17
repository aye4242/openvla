#!/bin/bash
# 启动 Gazebo + Franka Panda + 相机仿真场景

set -e

# 彻底退出 conda 环境，避免 Python 3.12 劫持 ROS2 的 Python 3.10
if [ -n "$CONDA_PREFIX" ]; then
    echo "检测到 conda 环境，正在退出 ..."
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda deactivate 2>/dev/null || true
    conda deactivate 2>/dev/null || true  # 双重退出防止 base 嵌套
fi
# 清除 conda 残留的 PATH
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v conda | grep -v anaconda3 | paste -sd: -)

# 环境设置
export DISPLAY=:0
source /opt/ros/humble/setup.bash

# 强制使用系统 Python
export PYTHONPATH="/usr/lib/python3/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages"
export LD_LIBRARY_PATH="/opt/ros/humble/lib:$LD_LIBRARY_PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROBOT_URDF="/tmp/panda.urdf"

# 生成带绝对路径的 URDF（避免 package:// 路径 Gazebo 找不到 mesh）
if [ ! -f "${ROBOT_URDF%.urdf}_abs.urdf" ]; then
    echo "Generating Franka Panda URDF ..."
    /usr/bin/python3 $(which xacro) /opt/ros/humble/share/franka_description/robots/fer/fer.urdf.xacro > "$ROBOT_URDF"
    sed 's|package://franka_description|/opt/ros/humble/share/franka_description|g' "$ROBOT_URDF" > "${ROBOT_URDF%.urdf}_abs.urdf"
fi
ROBOT_URDF="${ROBOT_URDF%.urdf}_abs.urdf"

echo "=== [1/3] 启动 Gazebo（空世界 + ROS2 插件）==="
ros2 launch gazebo_ros gazebo.launch.py &
sleep 15

echo "=== [2/3] Spawn Franka Panda ==="
ros2 run gazebo_ros spawn_entity.py \
    -entity panda \
    -file "$ROBOT_URDF" \
    -x 0 -y 0 -z 0
echo "Panda spawned!"

echo "=== [3/3] Spawn 相机 ==="
ros2 run gazebo_ros spawn_entity.py \
    -entity camera \
    -file "$SCRIPT_DIR/models/camera/camera.sdf"
echo "Camera spawned!"

echo ""
echo "========================================="
echo "  Gazebo 仿真场景启动完成！"
echo "========================================="
echo ""
echo "验证命令（新终端）："
echo "  source /opt/ros/humble/setup.bash"
echo "  ros2 topic list              # 应该看到 /camera/image_raw"
echo "  ros2 topic echo /camera/image_raw --no-arr  # 看相机消息"
