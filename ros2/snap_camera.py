#!/usr/bin/env python3
"""抓一帧 /camera/camera/image_raw 存成 /tmp/scene.png，主人用文件管理器打开看。"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class Snap(Node):
    def __init__(self):
        super().__init__("snap_camera")
        self.bridge = CvBridge()
        self.done = False
        self.sub = self.create_subscription(
            Image, "/camera/camera/image_raw", self.cb, 1
        )
        self.get_logger().info("Waiting for one frame...")

    def cb(self, msg: Image):
        if self.done:
            return
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imwrite("/tmp/scene.png", img)
        self.get_logger().info(f"Saved {img.shape} → /tmp/scene.png")
        self.done = True


def main():
    rclpy.init()
    node = Snap()
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
