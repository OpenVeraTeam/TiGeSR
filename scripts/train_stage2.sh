#!/usr/bin/env bash
set -euo pipefail

# RealCE Stage-2 recipe. Replace only the /path/to/... entries.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=2400
CUDA_VISIBLE_DEVICES=0,1,2,3 HF_ENDPOINT=https://hf-mirror.com accelerate launch \
  --main_process_port 25901 scripts/train_stage2.py \
  --output_dir outputs/stage2_realce \
  --train_folders /path/to/realce/train \
  --test_folder /path/to/realce/val \
  --num_training_epochs 100 \
  --ckpt_path /path/to/model_8281.pkl \
  --timestep 150 --lambda_l2 1 --lambda_lpips 5 --lambda_edge 100 \
  --train_batch_size 1 --eval_freq 1690 \
  --report_to tensorboard --tracker_project_name logs \
  --gradient_checkpointing --num_samples_eval 10 \
  --checkpointing_steps 1690 --ctrl_layers 6
