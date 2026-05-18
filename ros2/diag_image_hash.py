#!/usr/bin/env python3
"""轻量诊断：每帧打印 /camera/camera/image_raw 的 hash、mean、std + 是否变化。

用途：当 VLA 输出恒定时，验证是否因为模型拿到了"不变的输入"。
- 如果 hash 一直变 → 图像在更新 → 问题是 VLA OOD（最常见）
- 如果 hash 一直同 → 相机话题没在 publish 新帧 → Gazebo 相机插件出问题

Usage:
    source /opt/ros/humble/setup.bash
    python3 ros2/diag_image_hash.py
"""
import hashlib
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImageHashDiag(Node):
    def __init__(self):
        super().__init__("image_hash_diag")
        self.bridge = CvBridge()
        self.last_hash = None
        self.count = 0
        self.sub = self.create_subscription(
            Image, "/camera/camera/image_raw", self.cb, 10
        )
        self.get_logger().info("Listening on /camera/camera/image_raw ...")

    def cb(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        h = hashlib.md5(img.tobytes()).hexdigest()[:8]
        mean = float(img.mean())
        std = float(img.std())
        changed = "CHANGED" if h != self.last_hash else "same  "
        self.get_logger().info(
            f"#{self.count:3d}  hash={h}  mean={mean:6.2f}  std={std:6.2f}  {changed}"
        )
        self.last_hash = h
        self.count += 1


def main():
    rclpy.init()
    node = ImageHashDiag()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
