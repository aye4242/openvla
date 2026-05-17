"""
visualize_dataset.py

可视化 BridgeData 数据集：显示图片 + 语言指令 + 动作轨迹。

Usage:
    cd /home/wh/hb/openvla
    python scripts/visualize_dataset.py
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Prevent TF from using GPU (conflicts with PyTorch)
tf.config.set_visible_devices([], "GPU")

DATA_DIR = "./data/bridge_orig_ep100"
NUM_EPISODES = 3
NUM_STEPS = 5


def main():
    print(f"Loading dataset from {DATA_DIR} ...")
    builder = tfds.builder_from_directory(DATA_DIR + "/1.0.0")
    ds = builder.as_dataset(split="train")

    for ep_idx, episode in enumerate(ds.take(NUM_EPISODES)):
        steps = list(episode["steps"])
        n_steps = len(steps)
        print(f"\n{'='*60}")
        print(f"Episode {ep_idx + 1}: {n_steps} steps")

        # Show first step's image and instruction
        first_step = steps[0]
        img = first_step["observation"]["image"].numpy()
        instruction = first_step["language_instruction"].numpy().decode("utf-8")
        print(f"Instruction: '{instruction}'")

        # Collect actions for this episode
        actions = [s["action"].numpy() for s in steps]
        actions = np.stack(actions)  # [n_steps, 7]

        print(f"Action shape: {actions.shape}")
        print(f"Action range: [{actions.min():.3f}, {actions.max():.3f}]")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Image
        axes[0].imshow(img)
        axes[0].set_title(f"Episode {ep_idx + 1}: First Frame\n'{instruction}'")
        axes[0].axis("off")

        # Actions trajectory
        labels = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
        for i, label in enumerate(labels):
            axes[1].plot(actions[:, i], label=label, alpha=0.8)
        axes[1].set_title("Action Trajectory (7-DoF)")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Action Value")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = f"./data/bridge_viz_ep{ep_idx + 1}.png"
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
        plt.close()

        # Print first few steps' actions
        print(f"\nFirst {min(NUM_STEPS, n_steps)} steps:")
        for s_idx, step in enumerate(steps[:NUM_STEPS]):
            act = step["action"].numpy()
            print(f"  Step {s_idx}: {act}")

    print(f"\n{'='*60}")
    print("Visualization complete!")


if __name__ == "__main__":
    main()
