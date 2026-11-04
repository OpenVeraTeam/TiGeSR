#!/usr/bin/env bash
set -euo pipefail

# Pruned RoPE Stage-1 recipe. Replace only the /path/to/... entries.
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --main_process_port 29501 scripts/train_stage1.py \
  --is_shape_same \
  --lambda_l2 1 --lambda_focal 0.5 \
  --config_yaml configs/realce_stage1_train.yaml \
  --output_dir outputs/stage1_pruned \
  --train_folders /path/to/train_a,/path/to/train_b,/path/to/train_c \
  --test_folder /path/to/val \
  --num_training_epochs 8 \
  --encoder_ckpt /path/to/encoder-200000.pth \
  --ckpt_path /path/to/DiffTSR_new.ckpt \
  --idm_ckpt /path/to/model-200000.pth \
  --train_batch_size 2 --learning_rate 5e-6 \
  --eval_freq 2000 --report_to tensorboard --tracker_project_name logs \
  --checkpointing_steps 4000 --dataloader_num_workers 8
