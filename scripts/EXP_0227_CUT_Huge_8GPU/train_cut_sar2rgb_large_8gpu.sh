#!/usr/bin/env bash
# EXP-0227 — CUT — SAR→RGB — Large tier (8 GPU)
#
# CUT large: ngf=128, ndf=128, n_layers_D=3, netG=resnet_9blocks (~56.5 M params)
# Resolution: 512×512 (crop from 1024)
# Gradient clipping + NaN skip (mode collapse mitigation)
#
# Stability: nce_T=0.1, nce_idt=false, lambda_GAN=0.5
#
# Usage:
#   bash scripts/EXP_0227_CUT_Huge_8GPU/train_cut_sar2rgb_large_8gpu.sh

set -euo pipefail

export HF_TOKEN="${HF_TOKEN:-}"
export HF_ENDPOINT="https://hf-mirror.com"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"

NGPU=8
LOG_DIR="./logs/EXP_0227_CUT_Huge_8GPU"
LOG_FILE="${LOG_DIR}/train_cut_sar2rgb_large_8gpu.log"
mkdir -p "${LOG_DIR}"

OUTPUT_DIR="./ckpt/EXP_0227_CUT_Huge_8GPU/cut/sar2rgb_large_512_8gpu"
NGF=128
NDF=128
N_LAYERS_D=3
NET_G="resnet_9blocks"

EXCLUDE_FILE="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"
PAIRED_VAL_MANIFEST="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_sar2rgb.txt"
USE_AUGMENTED=true
USE_HORIZONTAL_FLIP=false
USE_VERTICAL_FLIP=false
RESOLUTION=512
USE_SAR2RGB_SUP=false

TRAIN_BATCH_SIZE=4
N_EPOCHS=0
N_EPOCHS_DECAY=0
MAX_TRAIN_STEPS=100000
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

USE_REP_ALIGNMENT=false
REP_ALIGNMENT_MODEL_PATH="./models/BiliSakura/MaRS-Base-RGB"
LAMBDA_REP_ALIGNMENT=0.2
LAMBDA_REP_ALIGNMENT_DECAY_STEPS=10000
LAMBDA_REP_ALIGNMENT_END=0.0

SWANLOG_DIR="./ckpt/swanlog"
SWANLAB_EXPERIMENT_NAME="exp-0227-cut-sar2rgb-large-512-8gpu"
SWANLAB_DESCRIPTION="EXP-0227 CUT SAR→RGB Large (512, ~56.5M, 8 GPU, grad clip)"
SWANLAB_TAGS="cut,exp-0227,sar2rgb,large,8gpu"

PUSH_TO_HUB=true
HUB_MODEL_ID="BiliSakura/4th-MAVIC-T-ckpt-0227"
HUB_PATH_TIER="large"
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
  --use_rep_alignment "${USE_REP_ALIGNMENT}"
  --rep_alignment_model_path "${REP_ALIGNMENT_MODEL_PATH}"
  --lambda_rep_alignment "${LAMBDA_REP_ALIGNMENT}"
  --lambda_rep_alignment_decay_steps "${LAMBDA_REP_ALIGNMENT_DECAY_STEPS}"
  --lambda_rep_alignment_end "${LAMBDA_REP_ALIGNMENT_END}"
)

[ -n "${RESUME_FROM_CHECKPOINT}" ] && ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")

echo "EXP-0227 CUT SAR2RGB large (8 GPU) — log: ${LOG_FILE}"
nohup accelerate launch --num_processes "${NGPU}" -m examples.cut.train_sar2rgb "${ARGS[@]}" > "${LOG_FILE}" 2>&1 &
