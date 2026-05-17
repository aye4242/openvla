#!/usr/bin/env python3
"""Panda 位置控制器（ros2_control 版）

订阅 /vla/action（7D delta） → 累加到目标位置 → 发布到
/forward_position_controller/commands（9D 绝对位置）。

关键差异（vs ApplyJointEffort 版）：
- 启动立即 publish [0]*9 → home pose 已 baked 到 URDF，所以 0 = 站立
- 不依赖关节反馈做 PID，Gazebo 内部伺服锁定位置
- 真正"硬"位置控制，无下沉、无穿模
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np

# joint 名顺序必须严格与 controllers.yaml 中 `joints:` 列表一致
ARM_JOINTS = [f"fer_joint{i}" for i in range(1, 8)]
GRIPPER_JOINTS = ["fer_finger_joint1", "fer_finger_joint2"]
ALL_JOINTS = ARM_JOINTS + GRIPPER_JOINTS  # 7 + 2 = 9

# Home pose（已 baked 到 URDF，所以全 0）
HOME = np.zeros(9)

# baked 后的关节限位（对应 /tmp/panda_controlled.urdf）
JOINT_LOWER = np.array([-2.8973, -0.9774, -2.8973, -0.7156, -2.8973, -1.5883, -3.6827, 0.0, 0.0])
JOINT_UPPER = np.array([ 2.8973,  2.5482,  2.8973,  2.2864,  2.8973,  2.1817,  2.1119, 0.04, 0.04])

DEFAULT_SCALE = 0.03      # VLA 6D delta → 关节增量缩放
GRIPPER_OPEN = 0.04       # finger 全开（米）
GRIPPER_CLOSE = 0.0       # finger 全闭
PUBLISH_RATE = 20.0       # Hz


class PandaController(Node):
    def __init__(self):
        super().__init__("panda_controller")
        self.declare_parameter("action_scale", DEFAULT_SCALE)

        self.target_positions = HOME.copy()

        self.pub = self.create_publisher(
            Float64MultiArray, "/forward_position_controller/commands", 10
        )
        self.create_subscription(
            Float64MultiArray, "/vla/action", self.action_callback, 10
        )
        self.timer = self.create_timer(1.0 / PUBLISH_RATE, self.publish_command)

        scale = self.get_parameter("action_scale").value
        self.get_logger().info(
            f"Panda controller ready (scale={scale}, rate={PUBLISH_RATE}Hz). "
            f"Locking home pose [0]*9 ..."
        )

    def action_callback(self, msg: Float64MultiArray):
        action = np.array(msg.data, dtype=float)
        if action.size != 7:
            self.get_logger().warn(f"Expected 7D action, got {action.size}D, ignored")
            return

        scale = self.get_parameter("action_scale").value

        # 方案 B（demo 简化）：6D EEF delta × scale → 前 6 关节增量累加
        # 第 7 关节固定保持 home（baked 后即 0）
        self.target_positions[:6] += action[:6] * scale

        # gripper：>0.5 → 闭合（finger=0），否则 → 打开（finger=0.04）
        gripper_target = GRIPPER_CLOSE if action[6] > 0.5 else GRIPPER_OPEN
        self.target_positions[7] = gripper_target
        self.target_positions[8] = gripper_target

        # clip 到关节限位（防溢出）
        np.clip(self.target_positions, JOINT_LOWER, JOINT_UPPER, out=self.target_positions)

    def publish_command(self):
        msg = Float64MultiArray()
        msg.data = self.target_positions.tolist()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = PandaController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
