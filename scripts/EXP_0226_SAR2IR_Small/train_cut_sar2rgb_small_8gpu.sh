#!/usr/bin/env bash
# EXP-0226 — CUT — SAR→RGB — Medium tier (~14.1 M params)
#
# Same small architecture as CUT SAR2EO (0217) that succeeded. Trains at 512×512
# (crop from 1024). Uses SwanLab for logging.
#
# Usage:
#   bash scripts/EXP_0226_SAR2IR_Small/train_cut_sar2rgb_small_8gpu.sh
#   NGPU=8 bash scripts/EXP_0226_SAR2IR_Small/train_cut_sar2rgb_small_8gpu.sh

set -euo pipefail

export HF_TOKEN="${HF_TOKEN:-}"
export HF_ENDPOINT="https://hf-mirror.com"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"

NGPU="${NGPU:-8}"
LOG_DIR="./logs/EXP_0226_SAR2IR_Small"
LOG_FILE="${LOG_DIR}/train_cut_sar2rgb_small_8gpu.log"
mkdir -p "${LOG_DIR}"

# CUT medium from configs/model_scaling_variants.yaml (~14.1 M)
OUTPUT_DIR="./ckpt/EXP_0226_SAR2IR_Small/cut/sar2rgb_small_512"
NGF=64
NDF=64
N_LAYERS_D=3
NET_G="resnet_9blocks"

EXCLUDE_FILE="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"
PAIRED_VAL_MANIFEST="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_sar2rgb.txt"
USE_AUGMENTED=true
USE_HORIZONTAL_FLIP=true
USE_VERTICAL_FLIP=false
RESOLUTION=512
USE_SAR2RGB_SUP=false

TRAIN_BATCH_SIZE=8
N_EPOCHS=0
N_EPOCHS_DECAY=0
MAX_TRAIN_STEPS=100000
GRADIENT_ACCUMULATION_STEPS=1
OPTIMIZER_TYPE="prodigy"
LR_POLICY="constant"
SAVE_MODEL_EPOCHS=0
CHECKPOINTING_STEPS=10000
CHECKPOINTS_TOTAL_LIMIT=1
VALIDATION_STEPS=10000
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

SWANLOG_DIR="./ckpt/swanlog"
SWANLAB_EXPERIMENT_NAME="exp-0226-cut-sar2rgb-small-512"
SWANLAB_DESCRIPTION="EXP-0226 CUT SAR→RGB Small (512, ~14.1M, medium tier)"
SWANLAB_TAGS="cut,exp-0226,sar2rgb,small"

PUSH_TO_HUB=true
HUB_MODEL_ID="BiliSakura/4th-MAVIC-T-ckpt-0226"
MIXED_PRECISION="bf16"
DATALOADER_NUM_WORKERS=8
SEED=42

ARGS=(
  --output_dir "${OUTPUT_DIR}"
  --log_with swanlab
  --swanlab_experiment_name "${SWANLAB_EXPERIMENT_NAME}"
  --swanlab_description "${SWANLAB_DESCRIPTION}"
  --swanlab_tags "${SWANLAB_TAGS}"
  --swanlab_init_kwargs_json '{"logdir":"'"${SWANLOG_DIR}"'","workspace":"EarthBridge"}'
  --ngf "${NGF}"
  --ndf "${NDF}"
  --n_layers_D "${N_LAYERS_D}"
  --netG "${NET_G}"
  --resolution "${RESOLUTION}"
  --exclude_file "${EXCLUDE_FILE}"
  --paired_val_manifest "${PAIRED_VAL_MANIFEST}"
  --use_sar2rgb_sup "${USE_SAR2RGB_SUP}"
  --use_augmented "${USE_AUGMENTED}"
  --use_horizontal_flip "${USE_HORIZONTAL_FLIP}"
  --use_vertical_flip "${USE_VERTICAL_FLIP}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --n_epochs "${N_EPOCHS}"
  --n_epochs_decay "${N_EPOCHS_DECAY}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --optimizer_type "${OPTIMIZER_TYPE}"
  --lr_policy "${LR_POLICY}"
  --save_model_epochs "${SAVE_MODEL_EPOCHS}"
  --checkpointing_steps "${CHECKPOINTING_STEPS}"
  --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT}"
  --validation_steps "${VALIDATION_STEPS}"
  --mixed_precision "${MIXED_PRECISION}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --seed "${SEED}"
  --push_to_hub "${PUSH_TO_HUB}"
  --hub_model_id "${HUB_MODEL_ID}"
)

[ -n "${RESUME_FROM_CHECKPOINT}" ] && ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")

if [ "${NGPU}" -gt 1 ]; then
  echo "EXP-0226 CUT SAR2RGB small — ${NGPU} GPUs. Log: ${LOG_FILE}"
  nohup accelerate launch --num_processes "${NGPU}" -m examples.cut.train_sar2rgb \
    "${ARGS[@]}" > "${LOG_FILE}" 2>&1 &
else
  echo "EXP-0226 CUT SAR2RGB small — 1 GPU. Log: ${LOG_FILE}"
  nohup python -m examples.cut.train_sar2rgb \
    "${ARGS[@]}" > "${LOG_FILE}" 2>&1 &
fi
