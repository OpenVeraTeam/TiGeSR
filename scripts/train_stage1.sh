#!/usr/bin/env bash
set -euo pipefail

# This is the original pruned-RoPE Stage-1 continuation recipe. Supply the
# required data and initialization checkpoints through environment variables.
: "${STAGE1_BASE_CKPT:?Set STAGE1_BASE_CKPT to DiffTSR_new.ckpt}"
: "${STAGE1_IDM_CKPT:?Set STAGE1_IDM_CKPT to the Stage-1 model checkpoint}"
: "${STAGE1_ENCODER_CKPT:?Set STAGE1_ENCODER_CKPT to the Stage-1 encoder checkpoint}"
: "${STAGE1_TRAIN_FOLDERS:?Set STAGE1_TRAIN_FOLDERS to comma-separated training folders}"
: "${STAGE1_TEST_FOLDER:?Set STAGE1_TEST_FOLDER to the validation folder}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" accelerate launch scripts/train_stage1.py \
  --is_shape_same \
  --config_yaml "${STAGE1_CONFIG:-configs/realce_stage1.yaml}" \
  --output_dir "${STAGE1_OUTPUT_DIR:-outputs/stage1}" \
  --train_folders "$STAGE1_TRAIN_FOLDERS" \
  --test_folder "$STAGE1_TEST_FOLDER" \
  --num_training_epochs 8 \
  --encoder_ckpt "$STAGE1_ENCODER_CKPT" \
  --ckpt_path "$STAGE1_BASE_CKPT" \
  --idm_ckpt "$STAGE1_IDM_CKPT" \
  --train_batch_size 2 --learning_rate 5e-5 \
  --lambda_l2 1 --lambda_focal 0.1 --lambda_edge 1 \
  --eval_freq 16000 --report_to tensorboard --tracker_project_name logs \
  --checkpointing_steps 32000 --dataloader_num_workers 8
