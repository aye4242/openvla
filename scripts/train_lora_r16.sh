#!/bin/bash
# LoRA 微调脚本 (r=16, QLoRA, 3090 24GB)

cd /home/wh/hb/openvla

# 加载本地 .env 中的 WANDB_API_KEY（不污染全局 ~/.netrc）
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

torchrun \
  --standalone \
  --nnodes 1 \
  --nproc-per-node 1 \
  vla-scripts/finetune.py \
  --vla_path ./openvla-7b \
  --data_root_dir ./data \
  --dataset_name bridge_orig_ep100 \
  --run_root_dir ./runs \
  --adapter_tmp_dir ./adapter-tmp \
  --use_lora True \
  --lora_rank 16 \
  --batch_size 2 \
  --grad_accumulation_steps 8 \
  --use_quantization True \
  --save_steps 100 \
  --max_steps 500 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_entity 2112404242-
