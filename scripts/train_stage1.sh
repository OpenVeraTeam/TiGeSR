#!/usr/bin/env bash
set -euo pipefail

# Override every placeholder below for your local dataset and checkpoint paths.
accelerate launch scripts/train_stage1.py \
  --config_yaml configs/stage1.yaml \
  --ckpt_path weights/stage1_base.ckpt \
  --idm_ckpt weights/stage1_init.pth \
  --encoder_ckpt weights/stage1_encoder.pth \
  --train_folders /path/to/stage1/train_a,/path/to/stage1/train_b \
  --test_folder /path/to/stage1/val \
  --output_dir outputs/stage1 \
  --num_training_epochs 8 --train_batch_size 2 --learning_rate 5e-5 \
  --lambda_l2 1 --lambda_focal 0.1 --lambda_edge 1 \
  --checkpointing_steps 32000 --eval_freq 16000 --dataloader_num_workers 8
