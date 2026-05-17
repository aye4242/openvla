#!/usr/bin/env python3
"""
测试用图片发布节点：从文件读取图片，发布到 /camera/image_raw

用于在没有 Gazebo 的情况下测试 VLA 推理节点。

Usage:
    bash ros2/run_test_publisher.sh
"""

import argparse
import time
import numpy as np
from PIL import Image as PILImage
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class TestImagePublisher(Node):
    """从文件读取图片，循环发布到 /camera/image_raw"""

    def __init__(self, image_path: str, publish_interval: float = 1.0):
        super().__init__("test_image_publisher")

        self.bridge = CvBridge()
        self.interval = publish_interval
        self.publisher = self.create_publisher(Image, "/camera/image_raw", 10)
        self.timer = self.create_timer(self.interval, self.publish_image)

        # 加载图片
        pil_image = PILImage.open(image_path).convert("RGB").resize((224, 224))
        self.cv_image = np.array(pil_image)
        self.get_logger().info(f"Loaded image: {image_path} ({self.cv_image.shape})")

        self.count = 0

    def publish_image(self):
        msg = self.bridge.cv2_to_imgmsg(self.cv_image, encoding="rgb8")
        self.publisher.publish(msg)
        self.count += 1
        self.get_logger().info(f"Published image #{self.count}")


def main():
    parser = argparse.ArgumentParser(description="Test Image Publisher")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--interval", type=float, default=2.0, help="Publish interval (seconds)")
    args = parser.parse_args()

    rclpy.init()
    node = TestImagePublisher(args.image, args.interval)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
