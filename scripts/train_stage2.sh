#!/usr/bin/env bash
set -euo pipefail

# This is the original Stage-2 RealCE continuation recipe. Supply a checkpoint
# that contains the VAE through STAGE2_CKPT_PATH.
: "${STAGE2_CKPT_PATH:?Set STAGE2_CKPT_PATH to the Stage-2 model checkpoint}"
: "${STAGE2_TRAIN_FOLDER:?Set STAGE2_TRAIN_FOLDER to the paired training folder}"
: "${STAGE2_TEST_FOLDER:?Set STAGE2_TEST_FOLDER to the paired validation folder}"

TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=2400 \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" accelerate launch scripts/train_stage2.py \
  --has_vae \
  --output_dir "${STAGE2_OUTPUT_DIR:-outputs/stage2}" \
  --train_folders "$STAGE2_TRAIN_FOLDER" \
  --test_folder "$STAGE2_TEST_FOLDER" \
  --num_training_epochs 50 \
  --ckpt_path "$STAGE2_CKPT_PATH" \
  --timestep 150 --learning_rate 5e-6 \
  --lambda_l2 1 --lambda_lpips 1 --lambda_edge 0 \
  --train_batch_size 1 --eval_freq 200 \
  --report_to tensorboard --tracker_project_name logs \
  --gradient_checkpointing --num_samples_eval 10 \
  --checkpointing_steps 200 --lr_scheduler cosine --lr_warmup_steps 200 \
  --adam_weight_decay 0.05 --ctrl_layers 6 \
  --early_stopping --early_stopping_patience 15 \
  --early_stopping_min_delta 0.001 --early_stopping_monitor eval/psnr
