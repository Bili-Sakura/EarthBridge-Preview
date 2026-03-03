#!/usr/bin/env bash
# EXP-0227 — CUT — SAR→IR — Huge tier (8 GPU)
#
# CUT huge: ngf=256, ndf=256, n_layers_D=4, netG=resnet_9blocks (~292.9 M params)
# Resolution: 512×512 (crop from 1024)
# Gradient clipping + NaN skip (mode collapse mitigation)
#
# Stability: nce_T=0.1, nce_idt=false, lambda_GAN=0.5
#
# Usage:
#   bash scripts/EXP_0227_CUT_Huge_8GPU/train_cut_sar2ir_huge_8gpu.sh

set -euo pipefail

export HF_TOKEN="${HF_TOKEN:-}"
export HF_ENDPOINT="https://hf-mirror.com"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"

NGPU=8
LOG_DIR="./logs/EXP_0227_CUT_Huge_8GPU"
LOG_FILE="${LOG_DIR}/train_cut_sar2ir_huge_8gpu.log"
mkdir -p "${LOG_DIR}"

OUTPUT_DIR="./ckpt/EXP_0227_CUT_Huge_8GPU/cut/sar2ir_huge_512_8gpu"
NGF=256
NDF=256
N_LAYERS_D=4
NET_G="resnet_9blocks"

EXCLUDE_FILE="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"
PAIRED_VAL_MANIFEST="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_sar2ir.txt"
USE_AUGMENTED=true
USE_HORIZONTAL_FLIP=true
USE_VERTICAL_FLIP=false
RESOLUTION=512

TRAIN_BATCH_SIZE=2
N_EPOCHS=0
N_EPOCHS_DECAY=0
MAX_TRAIN_STEPS=200000
GRADIENT_ACCUMULATION_STEPS=1
OPTIMIZER_TYPE="prodigy"
LEARNING_RATE=1.0
LR_POLICY="constant"
NCE_T=0.1
NCE_IDT=false
LAMBDA_GAN=0.5
SAVE_MODEL_EPOCHS=0
CHECKPOINTING_STEPS=10000
CHECKPOINTS_TOTAL_LIMIT=1
VALIDATION_STEPS=999999999999
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

SWANLOG_DIR="./ckpt/swanlog"
SWANLAB_EXPERIMENT_NAME="exp-0227-cut-sar2ir-huge-512-8gpu"
SWANLAB_DESCRIPTION="EXP-0227 CUT SAR→IR Huge (512, ~292.9M, 8 GPU, grad clip)"
SWANLAB_TAGS="cut,exp-0227,sar2ir,huge,8gpu"

PUSH_TO_HUB=true
HUB_MODEL_ID="BiliSakura/4th-MAVIC-T-ckpt-0227"
HUB_PATH_TIER="huge"
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
  --use_augmented "${USE_AUGMENTED}"
  --use_horizontal_flip "${USE_HORIZONTAL_FLIP}"
  --use_vertical_flip "${USE_VERTICAL_FLIP}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --n_epochs "${N_EPOCHS}"
  --n_epochs_decay "${N_EPOCHS_DECAY}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --optimizer_type "${OPTIMIZER_TYPE}"
  --learning_rate "${LEARNING_RATE}"
  --lr_policy "${LR_POLICY}"
  --nce_T "${NCE_T}"
  --nce_idt "${NCE_IDT}"
  --lambda_GAN "${LAMBDA_GAN}"
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

echo "EXP-0227 CUT SAR2IR huge (8 GPU) — log: ${LOG_FILE}"
nohup accelerate launch --num_processes "${NGPU}" -m examples.cut.train_sar2ir "${ARGS[@]}" > "${LOG_FILE}" 2>&1 &
