#!/bin/bash
# 测试图片发布节点启动脚本

cd /home/wh/hb/openvla

conda activate openvla
source /opt/ros/humble/setup.bash
export PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="/opt/ros/humble/lib:$LD_LIBRARY_PATH"

echo "Starting test image publisher ..."
python3 ros2/test_image_publisher.py \
    --image ./image.png \
    --interval 3
