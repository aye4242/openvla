#!/usr/bin/env python3
"""
VLA ROS2 推理节点

订阅 /camera/image_raw → OpenVLA 推理 → 发布 /vla/action

Usage:
    bash ros2/run_vla_node.sh
"""

import argparse
import time
import numpy as np
import torch
from PIL import Image as PILImage
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


class VLARosNode(Node):
    """OpenVLA ROS2 推理节点：订阅相机图像，发布 7D 动作"""

    def __init__(self, model_path: str, unnorm_key: str, instruction: str):
        super().__init__("vla_inference")

        self.instruction = instruction
        self.unnorm_key = unnorm_key
        self.bridge = CvBridge()

        # 加载模型
        self.get_logger().info(f"Loading OpenVLA from {model_path} ...")
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to("cuda:0").eval()
        self.get_logger().info("Model loaded!")

        # 订阅 / 发布
        image_topic = "/camera/camera/image_raw"
        self.sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.pub = self.create_publisher(Float64MultiArray, "/vla/action", 10)

        self.get_logger().info(f'Ready! Instruction: "{instruction}"')
        self.get_logger().info(f"Waiting for images on {image_topic} ...")

    def image_callback(self, msg: Image):
        t_start = time.time()

        # ROS Image → numpy → PIL
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        pil_image = PILImage.fromarray(cv_image)
        t_preprocess = time.time()

        # 推理
        prompt = f"What action should the robot take to {self.instruction.lower()}?"
        inputs = self.processor(prompt, pil_image).to("cuda:0", dtype=torch.bfloat16)
        inputs.pop("attention_mask", None)

        with torch.no_grad():
            action = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        t_inference = time.time()

        # 发布
        action_msg = Float64MultiArray()
        action_msg.data = action.tolist()
        self.pub.publish(action_msg)

        preprocess_ms = (t_preprocess - t_start) * 1000
        inference_ms = (t_inference - t_preprocess) * 1000
        total_ms = (t_inference - t_start) * 1000
        self.get_logger().info(
            f"Latency: {total_ms:.0f}ms (preprocess {preprocess_ms:.0f}ms + inference {inference_ms:.0f}ms) | "
            f"Action: [{', '.join(f'{v:.4f}' for v in action)}]"
        )


def main():
    parser = argparse.ArgumentParser(description="VLA ROS2 Inference Node")
    parser.add_argument("--model_path", type=str,
                        default="./runs/openvla-7b+bridge_orig_ep100+b16+lr-0.0005+lora-r16+dropout-0.0+q-4bit")
    parser.add_argument("--unnorm_key", type=str, default="bridge_orig")
    parser.add_argument("--instruction", type=str, default="pick up the object")
    args = parser.parse_args()

    rclpy.init()
    node = VLARosNode(args.model_path, args.unnorm_key, args.instruction)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
