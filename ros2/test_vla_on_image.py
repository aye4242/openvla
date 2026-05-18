#!/usr/bin/env python3
"""诊断脚本：用单张图像测试 VLA 输出，验证模型本身是否正常。

Usage:
    # 方式1：本地图片
    python3 ros2/test_vla_on_image.py --model_path ./openvla-7b --image_path /path/to/real_scene.jpg

    # 方式2：从 HuggingFace BridgeData 下载一张示例图
    python3 ros2/test_vla_on_image.py --model_path ./openvla-7b --download_bridge_sample

    # 方式3：用 Gazebo 当前截图（需要安装 gnome-screenshot 或类似工具）
    python3 ros2/test_vla_on_image.py --model_path ./runs/... --image_path /tmp/gazebo_camera.png
"""
import argparse
import time
import numpy as np
import torch
from PIL import Image as PILImage
import requests
from io import BytesIO

from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# BridgeData V2 HuggingFace 数据集示例图像（第一帧 = 初始场景）
# 这些 URL 来自 BridgeData V2 公开样本，展示 WidowX + 桌面 + 物体的真实场景
BRIDGE_SAMPLE_URLS = [
    # 从 HuggingFace lerobot 项目里找的示例
    "https://huggingface.co/datasets/lerobot/bridge_data_v2/resolve/main/data/bridge_data_v2/0/0/observations/images/0.jpg",
    "https://huggingface.co/datasets/lerobot/bridge_data_v2/resolve/main/data/bridge_data_v2/0/1/observations/images/0.jpg",
]


def load_model(model_path: str):
    print(f"Loading model from {model_path} ...")
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0").eval()
    print("Model loaded!")
    return processor, model


def download_image(url: str) -> PILImage.Image:
    print(f"Downloading image from {url} ...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    img = PILImage.open(BytesIO(resp.content)).convert("RGB")
    print(f"  Downloaded: {img.size}")
    return img


def infer(processor, model, image: PILImage.Image, instruction: str, unnorm_key: str):
    prompt = f"What action should the robot take to {instruction.lower()}?"
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    inputs.pop("attention_mask", None)

    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)

    # predict_action 可能返回 tensor 或 numpy，统一处理
    if hasattr(action, "cpu"):
        action = action.cpu().numpy()
    return np.asarray(action, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--download_bridge_sample", action="store_true")
    parser.add_argument("--instruction", type=str, default="pick up the object")
    parser.add_argument("--unnorm_key", type=str, default="bridge_orig")
    parser.add_argument("--n_runs", type=int, default=3, help="同一张图推理几次，看输出稳定性")
    args = parser.parse_args()

    processor, model = load_model(args.model_path)

    # 获取图像
    if args.download_bridge_sample:
        for url in BRIDGE_SAMPLE_URLS:
            try:
                image = download_image(url)
                break
            except Exception as e:
                print(f"  Failed to download {url}: {e}")
                continue
        else:
            print("All download URLs failed.")
            return
    elif args.image_path:
        print(f"Loading local image: {args.image_path}")
        image = PILImage.open(args.image_path).convert("RGB")
        print(f"  Size: {image.size}")
    else:
        print("Error: provide --image_path or --download_bridge_sample")
        return

    # 保存/显示图像信息
    print(f"\nInstruction: '{args.instruction}'")
    print(f"Unnorm key:  {args.unnorm_key}")
    print(f"Running inference {args.n_runs} times on the same image...\n")

    for i in range(args.n_runs):
        t0 = time.time()
        action = infer(processor, model, image, args.instruction, args.unnorm_key)
        dt = (time.time() - t0) * 1000
        print(f"Run {i+1}: {dt:.0f}ms | Action: [{', '.join(f'{v:+.4f}' for v in action)}]")

    # 汇总分析
    print("\n--- 输出解读 ---")
    print("前6维 [dx, dy, dz, droll, dpitch, dyaw] 是 EEF 的 Cartesian delta")
    print("第7维 [gripper] 是夹爪开合 (0=开, 1=闭)")
    print("\n如果真实场景图输出的是'靠近物体'的正值（如 +x 方向），说明模型本身正常，")
    print("问题确实是 Gazebo 合成图像的 sim2real gap。")
    print("如果真实场景图也输出'远离/后退'的负值，说明可能是 prompt 或 unnorm_key 问题。")


if __name__ == "__main__":
    main()
