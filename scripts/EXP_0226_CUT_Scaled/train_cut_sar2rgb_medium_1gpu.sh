#!/usr/bin/env bash
# EXP-0226 — CUT — SAR→RGB — Medium tier (1 GPU)
#
# CUT medium: ngf=64, ndf=64, n_layers_D=3, netG=resnet_9blocks (~14.1 M params)
# Resolution: 512×512 (crop from 1024)
#
# Usage:
#   bash scripts/EXP_0226_CUT_Scaled/train_cut_sar2rgb_medium_1gpu.sh

set -euo pipefail

export HF_TOKEN="${HF_TOKEN:-}"
export HF_ENDPOINT="https://hf-mirror.com"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"

NGPU=1
LOG_DIR="./logs/EXP_0226_CUT_Scaled"
LOG_FILE="${LOG_DIR}/train_cut_sar2rgb_medium_1gpu.log"
mkdir -p "${LOG_DIR}"

OUTPUT_DIR="./ckpt/EXP_0226_CUT_Scaled/cut/sar2rgb_medium_512_1gpu"
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

TRAIN_BATCH_SIZE=16
N_EPOCHS=0
N_EPOCHS_DECAY=0
MAX_TRAIN_STEPS=200000
GRADIENT_ACCUMULATION_STEPS=1
OPTIMIZER_TYPE="prodigy"
LR_POLICY="constant"
SAVE_MODEL_EPOCHS=0
CHECKPOINTING_STEPS=20000
CHECKPOINTS_TOTAL_LIMIT=1
VALIDATION_STEPS=999999999999
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

SWANLOG_DIR="./ckpt/swanlog"
SWANLAB_EXPERIMENT_NAME="exp-0226-cut-sar2rgb-medium-512-1gpu"
SWANLAB_DESCRIPTION="EXP-0226 CUT SAR→RGB Medium (512, ~14.1M, 1 GPU)"
SWANLAB_TAGS="cut,exp-0226,sar2rgb,medium,1gpu"

PUSH_TO_HUB=true
HUB_MODEL_ID="BiliSakura/4th-MAVIC-T-ckpt-0226"
HUB_PATH_TIER="medium"
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
  --hub_path_tier "${HUB_PATH_TIER}"
)

[ -n "${RESUME_FROM_CHECKPOINT}" ] && ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")

echo "EXP-0226 CUT SAR2RGB medium (1 GPU) — log: ${LOG_FILE}"
nohup python -m examples.cut.train_sar2rgb "${ARGS[@]}" > "${LOG_FILE}" 2>&1 &
