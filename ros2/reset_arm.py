#!/usr/bin/env python3
"""一键发 [0]*9 让 controller 回到 home pose。"""
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class ResetArm(Node):
    def __init__(self):
        super().__init__("reset_arm")
        self.pub = self.create_publisher(Float64MultiArray, "/forward_position_controller/commands", 10)
        time.sleep(0.5)
        msg = Float64MultiArray()
        msg.data = [0.0] * 9
        self.pub.publish(msg)
        self.get_logger().info("Sent [0]*9 → home pose.")


def main():
    rclpy.init()
    node = ResetArm()
    rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
