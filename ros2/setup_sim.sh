#!/bin/bash
# Gazebo 仿真一键启动（ros2_control 版）
#
# 流程：
#   1. 启动 Gazebo（带 ROS2 init / factory / force_system 插件）
#   2. 生成 panda_controlled.urdf（含 home pose baking + ros2_control 配置）
#   3. spawn panda（gazebo_ros2_control 插件嵌入式启动 controller_manager）
#   4. spawner 激活 joint_state_broadcaster + forward_position_controller
#   5. 启动 panda_controller.py（立即 publish [0]*9 锁住 home pose）
#   6. spawn 桌子 / 物体 / 相机
set -e

# 退出 conda 避免劫持 ROS2 Python
if [ -n "$CONDA_PREFIX" ]; then
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda deactivate 2>/dev/null || true
    conda deactivate 2>/dev/null || true
fi
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v conda | grep -v anaconda3 | paste -sd: -)

export DISPLAY=:0
export GAZEBO_MODEL_PATH="/opt/ros/humble/share/franka_description:$GAZEBO_MODEL_PATH"
export GAZEBO_MODEL_DATABASE_URI=""
source /opt/ros/humble/setup.bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
URDF="/tmp/panda_controlled.urdf"

echo "=== [1/6] 生成 panda_controlled.urdf ==="
/usr/bin/python3 "$SCRIPT_DIR/generate_controlled_urdf.py"

echo ""
echo "=== [2/6] 启动 Gazebo（后台）==="
gazebo --verbose \
    -s libgazebo_ros_init.so \
    -s libgazebo_ros_factory.so \
    -s libgazebo_ros_force_system.so \
    &>/tmp/gazebo.log &
GZ_PID=$!
echo "  Gazebo PID: $GZ_PID, 日志: /tmp/gazebo.log"
sleep 10

echo ""
echo "=== [3/6] Spawn Panda（gazebo_ros2_control 自动激活）==="
ros2 run gazebo_ros spawn_entity.py \
    -entity panda \
    -file "$URDF" \
    -x 0 -y 0 -z 0
# spawn_entity 返回后 controller_manager 应已在 Gazebo 进程中启动
sleep 3

echo ""
echo "=== [4/6] 激活控制器（joint_state_broadcaster + forward_position_controller）==="
ros2 run controller_manager spawner joint_state_broadcaster \
    --controller-manager /controller_manager \
    --controller-manager-timeout 30
ros2 run controller_manager spawner forward_position_controller \
    --controller-manager /controller_manager \
    --controller-manager-timeout 30

echo ""
echo "=== [5/6] 启动 panda_controller（立即锁 home pose）==="
/usr/bin/python3 "$SCRIPT_DIR/panda_controller.py" &>/tmp/panda_controller.log &
CTRL_PID=$!
echo "  Controller PID: $CTRL_PID, 日志: /tmp/panda_controller.log"
sleep 1

echo ""
echo "=== [6/6] Spawn 桌子 / 物体 / 相机 ==="
ros2 run gazebo_ros spawn_entity.py -entity table     -file "$SCRIPT_DIR/models/table/table.sdf"
ros2 run gazebo_ros spawn_entity.py -entity red_block -file "$SCRIPT_DIR/models/object/object.sdf"
ros2 run gazebo_ros spawn_entity.py -entity camera    -file "$SCRIPT_DIR/models/camera/camera.sdf"

echo ""
echo "========================================="
echo "  仿真环境就绪！"
echo "========================================="
echo "  Gazebo PID:    $GZ_PID"
echo "  Controller PID: $CTRL_PID"
echo ""
echo "  关闭：    kill $GZ_PID $CTRL_PID"
echo ""
echo "  验证命令："
echo "    ros2 control list_controllers"
echo "    ros2 topic echo /joint_states --once"
echo "    ros2 topic echo /forward_position_controller/commands --once"
echo ""
echo "  下一步：bash ros2/run_inference.sh"
