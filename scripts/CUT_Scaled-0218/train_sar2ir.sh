#!/usr/bin/env bash
# CUT-Scaled-0218 — SAR→IR — huge size (~293 M params).
# 1024×1024 task per configs/model_scaling_variants.yaml.
#
# Usage:
#   bash scripts/CUT_Scaled-0218/train_sar2ir.sh
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/CUT_Scaled-0218/train_sar2ir.sh
#
# SwanLab logs to ./ckpt/swanlog

set -euo pipefail

export HF_TOKEN="${HF_TOKEN:-}"
export HF_ENDPOINT="https://hf-mirror.com"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"

LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/train_sar2ir_cut_scaled_huge.log"
mkdir -p "${LOG_DIR}"

# CUT huge from configs/model_scaling_variants.yaml (1024px task)
OUTPUT_DIR="./ckpt/4th-MAVIC-T-ckpt-0218/sar2ir_huge"
NGF=256
NDF=256
N_LAYERS_D=4
NET_G="resnet_9blocks"

EXCLUDE_FILE="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"
PAIRED_VAL_MANIFEST="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_sar2ir.txt"
USE_AUGMENTED=true
USE_HORIZONTAL_FLIP=true
USE_VERTICAL_FLIP=false

# Smaller batch for 1024px huge model
TRAIN_BATCH_SIZE=2
N_EPOCHS=0
N_EPOCHS_DECAY=0
MAX_TRAIN_STEPS=200000
GRADIENT_ACCUMULATION_STEPS=4
OPTIMIZER_TYPE="prodigy"
LR_POLICY="constant"
SAVE_MODEL_EPOCHS=0
CHECKPOINTING_STEPS=2000
CHECKPOINTS_TOTAL_LIMIT=1
VALIDATION_EPOCHS=
VALIDATION_STEPS=2000
RESUME_FROM_CHECKPOINT="latest"

# SwanLab
LOG_WITH="swanlab"
SWANLOG_DIR="./ckpt/swanlog"
SWANLAB_EXPERIMENT_NAME="cut-scaled-sar2ir"
SWANLAB_DESCRIPTION="CUT Scaled SAR→IR (1024, huge)"
SWANLAB_TAGS="cut,scaled,sar2ir"

PUSH_TO_HUB=true
HUB_MODEL_ID="BiliSakura/4th-MAVIC-T-ckpt-0218"

MIXED_PRECISION="bf16"
DATALOADER_NUM_WORKERS=4
SEED=42

ARGS=(
  --output_dir "${OUTPUT_DIR}"
  --log_with "${LOG_WITH}"
  --swanlab_experiment_name "${SWANLAB_EXPERIMENT_NAME}"
  --swanlab_description "${SWANLAB_DESCRIPTION}"
  --swanlab_tags "${SWANLAB_TAGS}"
  --swanlab_init_kwargs_json '{"logdir":"'"${SWANLOG_DIR}"'","workspace":"EarthBridge"}'
  --ngf "${NGF}"
  --ndf "${NDF}"
  --n_layers_D "${N_LAYERS_D}"
  --netG "${NET_G}"
  --exclude_file "${EXCLUDE_FILE}"
  --paired_val_manifest "${PAIRED_VAL_MANIFEST}"
  --use_augmented "${USE_AUGMENTED}"
  --use_horizontal_flip "${USE_HORIZONTAL_FLIP}"
  --use_vertical_flip "${USE_VERTICAL_FLIP}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --n_epochs "${N_EPOCHS}"
  --n_epochs_decay "${N_EPOCHS_DECAY}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --optimizer_type "${OPTIMIZER_TYPE}"
  --lr_policy "${LR_POLICY}"
  --save_model_epochs "${SAVE_MODEL_EPOCHS}"
  --checkpointing_steps "${CHECKPOINTING_STEPS}"
  --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT}"
  --mixed_precision "${MIXED_PRECISION}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --seed "${SEED}"
  --validation_steps "${VALIDATION_STEPS}"
  --push_to_hub "${PUSH_TO_HUB}"
  --hub_model_id "${HUB_MODEL_ID}"
)

[ -n "${MAX_TRAIN_STEPS}" ] && ARGS+=(--max_train_steps "${MAX_TRAIN_STEPS}")
[ -n "${RESUME_FROM_CHECKPOINT}" ] && ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")

echo "CUT SAR2IR huge (scaled) — log: ${LOG_FILE} (SwanLab: ${SWANLOG_DIR}). Running in background."
nohup python -m examples.cut.train_sar2ir "${ARGS[@]}" > "${LOG_FILE}" 2>&1 &
