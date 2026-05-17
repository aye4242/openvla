"""Gazebo + Panda(高阻尼) + 桌子/物体/相机 一键启动"""
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart


def generate_launch_description():
    urdf_file = "/tmp/panda_stable.urdf"

    gazebo = ExecuteProcess(
        cmd=["gazebo", "--verbose", "-s", "libgazebo_ros_init.so",
             "-s", "libgazebo_ros_factory.so", "-s", "libgazebo_ros_force_system.so"],
        output="screen",
        env={**os.environ, "DISPLAY": ":0",
             "GAZEBO_MODEL_PATH": "/opt/ros/humble/share/franka_description:" + os.environ.get("GAZEBO_MODEL_PATH", "")},
    )

    spawn_panda = ExecuteProcess(
        cmd=["ros2", "run", "gazebo_ros", "spawn_entity.py",
             "-entity", "panda", "-file", urdf_file,
             "-x", "0", "-y", "0", "-z", "0"],
        output="screen",
    )

    spawn_table = ExecuteProcess(
        cmd=["ros2", "run", "gazebo_ros", "spawn_entity.py",
             "-entity", "table",
             "-file", "/home/wh/hb/openvla/ros2/models/table/table.sdf"],
        output="screen",
    )

    spawn_block = ExecuteProcess(
        cmd=["ros2", "run", "gazebo_ros", "spawn_entity.py",
             "-entity", "red_block",
             "-file", "/home/wh/hb/openvla/ros2/models/object/object.sdf"],
        output="screen",
    )

    spawn_camera = ExecuteProcess(
        cmd=["ros2", "run", "gazebo_ros", "spawn_entity.py",
             "-entity", "camera",
             "-file", "/home/wh/hb/openvla/ros2/models/camera/camera.sdf"],
        output="screen",
    )

    return LaunchDescription([
        gazebo,
        # Wait 10s for Gazebo to fully start, then spawn everything
        TimerAction(period=10.0, actions=[spawn_panda]),
        TimerAction(period=12.0, actions=[spawn_table]),
        TimerAction(period=13.0, actions=[spawn_block]),
        TimerAction(period=14.0, actions=[spawn_camera]),
    ])
