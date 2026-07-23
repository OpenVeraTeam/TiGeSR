#!/usr/bin/env bash
set -euo pipefail

# Pretrain on synthetic data, then fine-tune on a real paired set. See README
# for the required directory layout of each --train_folders dataset.
accelerate launch scripts/train_stage2.py \
  --train_folders /path/to/stage2/train --test_folder /path/to/stage2/val \
  --output_dir outputs/stage2 --num_training_epochs 50 --train_batch_size 1 \
  --learning_rate 5e-6 --timestep 150 --ctrl_layers 6 \
  --lambda_l2 1 --lambda_lpips 5 --lambda_edge 0 \
  --gradient_checkpointing --checkpointing_steps 1000 --eval_freq 1000
