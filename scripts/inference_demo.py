"""
inference_demo.py

OpenVLA-7B 零样本推理脚本。
输入图片 + 语言指令 → 输出 7 维机器人动作 [dx, dy, dz, droll, dpitch, dyaw, gripper]。

Usage:
    cd /home/wh/hb/openvla
    python scripts/inference_demo.py
"""

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL_PATH = "./openvla-7b"
IMAGE_PATH = "./table.png"
INSTRUCTIONS = [
    "pick up the cup",
    "pick up the glasses",
    "push the cup",
]


def main():
    print(f"[1/3] Loading model from {MODEL_PATH} ...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        attn_implementation="eager",  # eager mode (flash-attn not installed yet)
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0")
    print(f"  Model loaded. Device: {vla.device}")

    print(f"[2/3] Loading image from {IMAGE_PATH} ...")
    image = Image.open(IMAGE_PATH).convert("RGB")
    print(f"  Image size: {image.size}")

    print(f"[3/3] Running inference with {len(INSTRUCTIONS)} instructions ...\n")
    labels = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

    for i, instruction in enumerate(INSTRUCTIONS, 1):
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        inputs.pop("attention_mask", None)

        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

        print(f"  [{i}] Instruction: '{instruction}'")
        for label, val in zip(labels, action):
            print(f"      {label:>8s}: {val:+.4f}")
        print()


if __name__ == "__main__":
    main()
