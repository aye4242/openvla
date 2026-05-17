"""
eval_action_accuracy.py

评估 OpenVLA 模型在 bridge_orig_ep100 上的动作预测精度，对比：
  1. 零样本（预训练 openvla-7b）
  2. LoRA r=16 微调
  3. LoRA r=32 微调

指标：
  - Action Accuracy（离散 token 匹配率，越高越好）
  - L1 Loss（连续动作误差，越低越好）
  - MSE（均方误差，越低越好）

输出：
  - 控制台对比表格
  - eval_results/action_accuracy.png（Action Accuracy 对比条形图）
  - eval_results/l1_per_dim.png（逐分量 L1 条形图）
  - eval_results/trajectory_compare.png（动作轨迹对比图）

Usage:
    cd /home/wh/hb/openvla
    python scripts/eval_action_accuracy.py
"""

import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.vla.action_tokenizer import ActionTokenizer

# TF 不用 GPU
tf.config.set_visible_devices([], "GPU")

# === 配置 ===
DATA_DIR = "./data/bridge_orig_ep100"
DATASET_STATS_PATH = "./runs/openvla-7b+bridge_orig_ep100+b16+lr-0.0005+lora-r16+dropout-0.0+q-4bit/dataset_statistics.json"
NUM_EPISODES = 10
OUTPUT_DIR = Path("./eval_results")
ACTION_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]

MODELS = {
    "Zero-shot": {
        "path": "./openvla-7b",
        "unnorm_key": "bridge_orig",
    },
    "LoRA r=8": {
        "path": "./runs/openvla-7b+bridge_orig_ep100+b16+lr-0.0005+lora-r8+dropout-0.0+q-4bit",
        "unnorm_key": "bridge_orig",
    },
    "LoRA r=16": {
        "path": "./runs/openvla-7b+bridge_orig_ep100+b16+lr-0.0005+lora-r16+dropout-0.0+q-4bit",
        "unnorm_key": "bridge_orig",
    },
    "LoRA r=32": {
        "path": "./runs/openvla-7b+bridge_orig_ep100+b16+lr-0.0005+lora-r32+dropout-0.0+q-4bit",
        "unnorm_key": "bridge_orig",
    },
}


def load_model(model_path: str):
    """加载 OpenVLA 模型和处理器"""
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
    return model, processor


def normalize_action(action, q01, q99, mask):
    """原始尺度 action → [-1, 1]（和训练时一致）"""
    return np.where(mask, 2.0 * (action - q01) / (q99 - q01) - 1.0, action)


def gt_action_to_token_ids(action, vocab_size, q01, q99, mask):
    """GT action → normalize → discretize → token IDs（直接计算，不走 decode/encode）"""
    bins = np.linspace(-1, 1, 256)
    normalized = normalize_action(action, q01, q99, mask)
    normalized = np.clip(normalized, -1, 1)
    discretized = np.digitize(normalized, bins)
    return vocab_size - discretized  # shape: (7,)


def predict_with_tokens(model, processor, action_tokenizer, image, instruction, unnorm_key):
    """推理：返回 (continuous_action, predicted_token_ids, gt_token_ids)"""
    prompt = f"What action should the robot take to {instruction.lower()}?"
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    inputs.pop("attention_mask", None)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=7,
            do_sample=False,
        )

    # predicted token IDs（最后 7 个）
    pred_token_ids = generated_ids[0, -7:].cpu().numpy()

    # predicted token IDs → continuous action（复用 predict_action 的 unnormalize 逻辑）
    normalized_actions = action_tokenizer.decode_token_ids_to_actions(pred_token_ids)
    norm_stats = model.get_action_stats(unnorm_key)
    mask = np.array(norm_stats.get("mask", np.ones(7, dtype=bool)))
    q01, q99 = np.array(norm_stats["q01"]), np.array(norm_stats["q99"])
    continuous_action = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (q99 - q01) + q01,
        normalized_actions,
    )

    return continuous_action, pred_token_ids


def load_episodes(num_episodes):
    """从 bridge_orig_ep100 加载指定数量的 episode"""
    builder = tfds.builder_from_directory(DATA_DIR + "/1.0.0")
    ds = builder.as_dataset(split="train")

    episodes = []
    for episode in ds.take(num_episodes):
        steps = list(episode["steps"])
        ep_data = []
        for step in steps:
            img = step["observation"]["image"].numpy()
            image = Image.fromarray(img)
            instruction = step["language_instruction"].numpy().decode("utf-8")
            gt_action = step["action"].numpy()
            ep_data.append({"image": image, "instruction": instruction, "gt_action": gt_action})
        episodes.append(ep_data)
    return episodes


def evaluate_model(model_name, model, processor, unnorm_key, episodes, action_tokenizer, vocab_size, ds_q01, ds_q99, ds_mask):
    """评估单个模型，返回 action accuracy + L1 + MSE"""
    all_preds = []
    all_gts = []
    all_correct_tokens = []
    all_total_tokens = []
    total_steps = sum(len(ep) for ep in episodes)

    step_count = 0
    for ep_idx, ep in enumerate(episodes):
        for s_idx, step in enumerate(ep):
            # 推理
            pred_action, pred_token_ids = predict_with_tokens(
                model, processor, action_tokenizer, step["image"], step["instruction"], unnorm_key
            )

            # GT token IDs（直接 discretize，不走 decode/encode）
            gt_token_ids = gt_action_to_token_ids(step["gt_action"], vocab_size, ds_q01, ds_q99, ds_mask)

            # token accuracy
            min_len = min(len(pred_token_ids), len(gt_token_ids))
            correct = (pred_token_ids[:min_len] == gt_token_ids[:min_len]).sum()
            all_correct_tokens.append(correct)
            all_total_tokens.append(min_len)

            # continuous metrics
            all_preds.append(pred_action)
            all_gts.append(step["gt_action"])
            step_count += 1

            if step_count % 50 == 0:
                print(f"  [{model_name}] Evaluated {step_count}/{total_steps} steps")

    preds = np.stack(all_preds)
    gts = np.stack(all_gts)
    action_accuracy = sum(all_correct_tokens) / sum(all_total_tokens) if sum(all_total_tokens) > 0 else 0.0

    return {
        "preds": preds, "gts": gts,
        "l1": np.abs(preds - gts), "mse": (preds - gts) ** 2,
        "action_accuracy": action_accuracy,
    }


def print_table(results):
    """打印对比表格"""
    print("\n" + "=" * 90)
    print(f"{'Model':<16} {'Accuracy':>10} {'Avg L1':>8} {'Avg MSE':>8} {'dx':>7} {'dy':>7} {'dz':>7} {'droll':>7} {'dpitch':>7} {'dyaw':>7} {'grip':>7}")
    print("-" * 90)
    for name, res in results.items():
        acc = res["action_accuracy"]
        avg_l1 = res["l1"].mean()
        avg_mse = res["mse"].mean()
        dim_l1 = res["l1"].mean(axis=0)
        row = f"{name:<16} {acc:>9.1%} {avg_l1:>8.5f} {avg_mse:>8.6f}"
        for d in dim_l1:
            row += f" {d:>7.4f}"
        print(row)
    print("=" * 90)


def plot_accuracy_comparison(results, output_dir):
    """绘制 Action Accuracy 对比条形图"""
    model_names = list(results.keys())
    accuracies = [results[n]["action_accuracy"] for n in model_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]
    bars = ax.bar(model_names, accuracies, color=colors, width=0.5)

    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax.set_ylabel("Action Accuracy", fontsize=12)
    ax.set_title("Action Token Accuracy Comparison", fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = output_dir / "action_accuracy.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_l1_per_dim(results, output_dir):
    """绘制逐分量 L1 条形图"""
    model_names = list(results.keys())
    dim_l1 = {name: results[name]["l1"].mean(axis=0) for name in model_names}

    x = np.arange(len(ACTION_LABELS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, name in enumerate(model_names):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, dim_l1[name], width, label=name)
        for bar, val in zip(bars, dim_l1[name]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.4f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Action Dimension")
    ax.set_ylabel("Mean L1 Loss")
    ax.set_title("Per-Dimension L1 Loss Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(ACTION_LABELS)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = output_dir / "l1_per_dim.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_trajectory_comparison(results, output_dir, episode_idx=0, episodes=None):
    """绘制某个 episode 的动作轨迹对比"""
    ep_start = 0
    for i in range(episode_idx):
        ep_start += len(episodes[i])
    ep_end = ep_start + len(episodes[episode_idx])
    steps = list(range(ep_end - ep_start))

    dims = [0, 1, 2]
    dim_names = [ACTION_LABELS[d] for d in dims]
    model_names = list(results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, d, dname in zip(axes, dims, dim_names):
        gt = results[model_names[0]]["gts"][ep_start:ep_end, d]
        ax.plot(steps, gt, "k-", linewidth=2, label="Ground Truth", alpha=0.8)
        for name in model_names:
            pred = results[name]["preds"][ep_start:ep_end, d]
            ax.plot(steps, pred, "--", linewidth=1.5, label=name, alpha=0.7)
        ax.set_title(dname)
        ax.set_xlabel("Step")
        ax.set_ylabel("Action Value")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Episode {episode_idx + 1} - Trajectory Comparison (dx/dy/dz)")
    plt.tight_layout()
    path = output_dir / "trajectory_compare.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 加载数据集 normalize 统计（用于 GT action → token IDs）
    with open(DATASET_STATS_PATH) as f:
        ds_stats = json.load(f)
    ds_action_stats = ds_stats["bridge_orig_ep100"]["action"]
    ds_q01 = np.array(ds_action_stats["q01"])
    ds_q99 = np.array(ds_action_stats["q99"])
    ds_mask = np.array(ds_action_stats.get("mask", [True] * 7))
    print(f"Loaded dataset statistics from {DATASET_STATS_PATH}")

    # 加载测试数据
    print(f"Loading {NUM_EPISODES} episodes from {DATA_DIR} ...")
    episodes = load_episodes(NUM_EPISODES)
    total_steps = sum(len(ep) for ep in episodes)
    print(f"Loaded {len(episodes)} episodes, {total_steps} total steps")

    # 逐模型评估
    all_results = {}
    for name, cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"Model path: {cfg['path']}")
        print(f"{'='*60}")

        model, processor = load_model(cfg["path"])
        action_tokenizer = ActionTokenizer(processor.tokenizer)
        vocab_size = processor.tokenizer.vocab_size
        res = evaluate_model(name, model, processor, cfg["unnorm_key"], episodes,
                             action_tokenizer, vocab_size, ds_q01, ds_q99, ds_mask)
        all_results[name] = res

        del model, processor
        torch.cuda.empty_cache()
        print(f"  [{name}] Accuracy = {res['action_accuracy']:.1%}, L1 = {res['l1'].mean():.5f}")

    # 输出对比表格
    print_table(all_results)

    # 可视化
    print("\nGenerating visualizations ...")
    plot_accuracy_comparison(all_results, OUTPUT_DIR)
    plot_l1_per_dim(all_results, OUTPUT_DIR)
    plot_trajectory_comparison(all_results, OUTPUT_DIR, episode_idx=0, episodes=episodes)

    # 保存数值结果
    summary = {}
    for name, res in all_results.items():
        summary[name] = {
            "action_accuracy": float(res["action_accuracy"]),
            "avg_l1": float(res["l1"].mean()),
            "avg_mse": float(res["mse"].mean()),
            "per_dim_l1": res["l1"].mean(axis=0).tolist(),
        }
    with open(OUTPUT_DIR / "eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved numerical results to {OUTPUT_DIR / 'eval_results.json'}")
    print("Done!")


if __name__ == "__main__":
    main()
