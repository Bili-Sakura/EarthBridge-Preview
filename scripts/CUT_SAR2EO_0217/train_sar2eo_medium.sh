#!/usr/bin/env bash
# CUT SAR2EO 2026/02/17 — medium size (~14.1 M params).
# Intended for simple GPUs (e.g. RTX 3090/4090); TensorBoard only (no SwanLab).
#
# Usage:
#   bash scripts/CUT_SAR2EO_0217/train_sar2eo_medium.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/CUT_SAR2EO_0217/train_sar2eo_medium.sh
#
# TensorBoard (run in another terminal):
#   tensorboard --logdir ./ckpt/4th-MAVIC-T-ckpt-0217

set -euo pipefail

# Hugging Face (set HF_TOKEN for push to hub; optional HF_ENDPOINT for mirror)
export HF_TOKEN="${HF_TOKEN:-}"
export HF_ENDPOINT="https://hf-mirror.com"

LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/train_sar2eo_cut_medium.log"
mkdir -p "${LOG_DIR}"

# CUT medium from configs/model_scaling_variants.yaml
OUTPUT_DIR="./ckpt/4th-MAVIC-T-ckpt-0217/sar2eo_medium"
NGF=64
NDF=64
N_LAYERS_D=3
NET_G="resnet_9blocks"

# Data / augmentation (same as CUT sar2eo defaults)
EXCLUDE_FILE="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"
USE_AUGMENTED=true
USE_HORIZONTAL_FLIP=true
USE_VERTICAL_FLIP=false

# Training: 10 epochs, validate and save+push checkpoint every epoch
TRAIN_BATCH_SIZE=8
N_EPOCHS=10
N_EPOCHS_DECAY=0
MAX_TRAIN_STEPS=
GRADIENT_ACCUMULATION_STEPS=1
OPTIMIZER_TYPE="prodigy"
SAVE_MODEL_EPOCHS=1
CHECKPOINTS_TOTAL_LIMIT=1
VALIDATION_EPOCHS=1
VALIDATION_STEPS=
RESUME_FROM_CHECKPOINT=

# Logging: TensorBoard only (no SwanLab)
LOG_WITH="tensorboard"

# Hugging Face Hub
PUSH_TO_HUB=true
HUB_MODEL_ID="BiliSakura/4th-MAVIC-T-ckpt-0217"

# Hardware
MIXED_PRECISION="bf16"
DATALOADER_NUM_WORKERS=4
SEED=42

ARGS=(
  --output_dir "${OUTPUT_DIR}"
  --log_with "${LOG_WITH}"
  --ngf "${NGF}"
  --ndf "${NDF}"
  --n_layers_D "${N_LAYERS_D}"
  --netG "${NET_G}"
  --exclude_file "${EXCLUDE_FILE}"
  --use_augmented "${USE_AUGMENTED}"
  --use_horizontal_flip "${USE_HORIZONTAL_FLIP}"
  --use_vertical_flip "${USE_VERTICAL_FLIP}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --n_epochs "${N_EPOCHS}"
  --n_epochs_decay "${N_EPOCHS_DECAY}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --optimizer_type "${OPTIMIZER_TYPE}"
  --save_model_epochs "${SAVE_MODEL_EPOCHS}"
  --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT}"
  --mixed_precision "${MIXED_PRECISION}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --seed "${SEED}"
  --validation_epochs "${VALIDATION_EPOCHS}"
  --push_to_hub "${PUSH_TO_HUB}"
  --hub_model_id "${HUB_MODEL_ID}"
)

[ -n "${MAX_TRAIN_STEPS}" ] && ARGS+=(--max_train_steps "${MAX_TRAIN_STEPS}")
[ -n "${VALIDATION_STEPS}" ] && ARGS+=(--validation_steps "${VALIDATION_STEPS}")
[ -n "${RESUME_FROM_CHECKPOINT}" ] && ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")

echo "CUT SAR2EO medium — log: ${LOG_FILE} (TensorBoard: ${OUTPUT_DIR}/logs). Running in background."
nohup python -m examples.cut.train_sar2eo "${ARGS[@]}" > "${LOG_FILE}" 2>&1 &
