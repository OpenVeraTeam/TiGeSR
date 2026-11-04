#!/usr/bin/env bash
set -euo pipefail

# Unpruned RoPE Stage-1 recipe. Replace only the /path/to/... entries.
accelerate launch --gpu_ids 2,3 --main_process_port 29501 scripts/train_stage1.py \
  --is_shape_same \
  --lambda_l2 1 --lambda_focal 0.5 \
  --config_yaml configs/stage1_train.yaml \
  --output_dir outputs/stage1_unpruned \
  --train_folders /path/to/train_a,/path/to/train_b,/path/to/train_c,/path/to/train_d,/path/to/train_e \
  --test_folder /path/to/val \
  --num_training_epochs 4 \
  --encoder_ckpt /path/to/encoder-296000.pth \
  --ckpt_path /path/to/DiffTSR_new.ckpt \
  --idm_ckpt /path/to/model-296000.pth \
  --train_batch_size 2 --learning_rate 5e-5 \
  --eval_freq 4000 --report_to tensorboard --tracker_project_name logs \
  --checkpointing_steps 8000 --dataloader_num_workers 8
