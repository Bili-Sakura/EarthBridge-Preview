#!/usr/bin/env bash
# DDBM-Pixel-Medium-0213 — RGB→IR — Pixel-space DDBM (no VAE)
# DDBM medium config, (512, 512) crop per roadmap
#
# Usage:
#   bash scripts/DDBM_Pixel_Medium-0213/train_rgb2ir.sh
#   # run on a specific GPU (e.g., cuda:0):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/DDBM_Pixel_Medium-0213/train_rgb2ir.sh
#   # (pick 0-3 to spread stages across 4 GPUs)
#   # or multi-GPU:
#   NGPU=4 bash scripts/DDBM_Pixel_Medium-0213/train_rgb2ir.sh

set -euo pipefail

export HF_TOKEN="${HF_TOKEN:-}"
export HF_ENDPOINT="https://hf-mirror.com"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"

NGPU="${NGPU:-1}"
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/train_rgb2ir.log"

mkdir -p "${LOG_DIR}"

# --- Medium config from configs/model_scaling_variants.yaml ---
NUM_CHANNELS=128
NUM_RES_BLOCKS=2
ATTENTION_RESOLUTIONS="32,16,8"
CHANNEL_MULT="1,1,2,2,4,4"

# --- Pixel-space DDBM (no VAE) ---
USE_LATENT_TARGET=false
RESOLUTION=512
OUTPUT_RESOLUTION=1024  # inference: load 1024, run 512-trained model → 1024 output (no upscale)

# --- Representation alignment (REPA) ---
USE_REP_ALIGNMENT=false
LAMBDA_REP_ALIGNMENT=0.1

# --- Data augmentation and filtering ---
USE_AUGMENTED=true
USE_HORIZONTAL_FLIP=true
USE_VERTICAL_FLIP=true
EXCLUDE_FILE="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"

# --- Training settings ---
OPTIMIZER_TYPE="prodigy"
USE_MAVIC_LOSS=false
TRAIN_BATCH_SIZE=8
NUM_EPOCHS=0
MAX_TRAIN_STEPS=100000
GRADIENT_ACCUMULATION_STEPS=1
USE_EMA=true
SAVE_MODEL_EPOCHS=0
CHECKPOINTING_STEPS=10000
CHECKPOINTS_TOTAL_LIMIT=1
VALIDATION_STEPS=10000
VALIDATION_EPOCHS=
NUM_INFERENCE_STEPS=100
PUSH_TO_HUB=true
HUB_MODEL_ID="BiliSakura/4th-MAVIC-T-ckpt-0213"
MIXED_PRECISION="bf16"
DATALOADER_NUM_WORKERS=8
SEED=42

OUTPUT_DIR="./ckpt/DDBM_Pixel_Medium-0213/rgb2ir"

# --- SwanLab ---
SWANLOG_DIR="./ckpt/swanlog"
SWANLAB_EXPERIMENT_NAME="ddbm-pixel-medium-rgb2ir"
SWANLAB_DESCRIPTION="DDBM Pixel Medium RGB→IR"
SWANLAB_TAGS="ddbm,pixel,rgb2ir"

# --- Resume from checkpoint ---
RESUME_FROM_CHECKPOINT="latest"

COMMON_ARGS=(
  --log_with swanlab
  --swanlab_experiment_name "${SWANLAB_EXPERIMENT_NAME}"
  --swanlab_description "${SWANLAB_DESCRIPTION}"
  --swanlab_tags "${SWANLAB_TAGS}"
  --swanlab_init_kwargs_json '{"logdir":"'"${SWANLOG_DIR}"'"}'
  --num_channels "${NUM_CHANNELS}"
  --num_res_blocks "${NUM_RES_BLOCKS}"
  --attention_resolutions "${ATTENTION_RESOLUTIONS}"
  --channel_mult "${CHANNEL_MULT}"
  --use_latent_target "${USE_LATENT_TARGET}"
  --resolution "${RESOLUTION}"
  --use_rep_alignment "${USE_REP_ALIGNMENT}"
  --lambda_rep_alignment "${LAMBDA_REP_ALIGNMENT}"
  --use_augmented "${USE_AUGMENTED}"
  --use_horizontal_flip "${USE_HORIZONTAL_FLIP}"
  --use_vertical_flip "${USE_VERTICAL_FLIP}"
  --exclude_file "${EXCLUDE_FILE}"
  --optimizer_type "${OPTIMIZER_TYPE}"
  --use_mavic_loss "${USE_MAVIC_LOSS}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --num_epochs "${NUM_EPOCHS}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --use_ema "${USE_EMA}"
  --save_model_epochs "${SAVE_MODEL_EPOCHS}"
  --checkpointing_steps "${CHECKPOINTING_STEPS}"
  --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT}"
  --push_to_hub "${PUSH_TO_HUB}"
  --hub_model_id "${HUB_MODEL_ID}"
  --mixed_precision "${MIXED_PRECISION}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --seed "${SEED}"
  --output_dir "${OUTPUT_DIR}"
  --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
)

if [ -n "${OUTPUT_RESOLUTION}" ]; then
  COMMON_ARGS+=(--output_resolution "${OUTPUT_RESOLUTION}")
fi
if [ -n "${VALIDATION_STEPS}" ]; then
  COMMON_ARGS+=(--validation_steps "${VALIDATION_STEPS}")
fi
if [ -n "${VALIDATION_EPOCHS}" ]; then
  COMMON_ARGS+=(--validation_epochs "${VALIDATION_EPOCHS}")
fi
if [ "${NGPU}" -gt 1 ]; then
  nohup accelerate launch --num_processes "${NGPU}" -m examples.ddbm.train_rgb2ir \
    "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
else
  nohup python -m examples.ddbm.train_rgb2ir \
    "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
fi
