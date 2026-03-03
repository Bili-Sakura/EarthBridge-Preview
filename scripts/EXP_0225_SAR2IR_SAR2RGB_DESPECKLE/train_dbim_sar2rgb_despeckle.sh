#!/usr/bin/env bash
# EXP-0225 — DBIM — SAR→RGB (despeckle + CFG dropout + REPA)
#
# Usage:
#   bash scripts/EXP_0225_SAR2IR_SAR2RGB_DESPECKLE/train_dbim_sar2rgb_despeckle.sh
#   NGPU=8 bash scripts/EXP_0225_SAR2IR_SAR2RGB_DESPECKLE/train_dbim_sar2rgb_despeckle.sh

set -euo pipefail

export HF_TOKEN="${HF_TOKEN:-}"
export HF_ENDPOINT="https://hf-mirror.com"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"

NGPU="${NGPU:-1}"
LOG_DIR="./logs/EXP_0225_SAR2IR_SAR2RGB_DESPECKLE"
LOG_FILE="${LOG_DIR}/train_dbim_sar2rgb_despeckle_ngpu${NGPU}.log"
mkdir -p "${LOG_DIR}"

# --- Model config (SAR-focused, no self-attention) ---
NUM_CHANNELS=96
NUM_RES_BLOCKS=2
ATTENTION_RESOLUTIONS=""
CHANNEL_MULT="1,1,2,2,4,4"

# --- Data / resolution ---
RESOLUTION=512
OUTPUT_RESOLUTION=1024
USE_AUGMENTED=true
USE_RANDOM_CROP=true
USE_HORIZONTAL_FLIP=true
USE_VERTICAL_FLIP=true
EXCLUDE_FILE="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"
PAIRED_VAL_MANIFEST="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_sar2rgb.txt"
USE_SAR2RGB_SUP=true
SAR2RGB_SUP_MANIFEST="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_sar2rgb_sup.txt"

# --- SAR despeckling ---
USE_SAR_DESPECKLE=true
SAR_DESPECKLE_KERNEL_SIZE=5
SAR_DESPECKLE_STRENGTH=0.6

# --- Training ---
OPTIMIZER_TYPE="prodigy"
LEARNING_RATE=1.0
PRODIGY_D0=1e-5
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=0
TRAIN_BATCH_SIZE=8
MAX_TRAIN_STEPS=120000
NUM_EPOCHS=0
GRADIENT_ACCUMULATION_STEPS=1
USE_EMA=true
SAVE_MODEL_EPOCHS=0
CHECKPOINTING_STEPS=10000
CHECKPOINTS_TOTAL_LIMIT=2
VALIDATION_STEPS=10000
NUM_INFERENCE_STEPS=100
MIXED_PRECISION="bf16"
DATALOADER_NUM_WORKERS=8
SEED=42

# --- CFG fine-tuning ---
CONDITIONING_DROPOUT_PROB=0.15

# --- REPA (target domain RGB encoder) ---
USE_REP_ALIGNMENT=false
LAMBDA_REP_ALIGNMENT=0.2
LAMBDA_REP_ALIGNMENT_DECAY_STEPS=2000
LAMBDA_REP_ALIGNMENT_END=0.0
REP_ALIGNMENT_MODEL_PATH="./models/BiliSakura/MaRS-Base-RGB"

PUSH_TO_HUB=true
HUB_MODEL_ID="BiliSakura/4th-MAVIC-T-ckpt-0225-v2"
OUTPUT_DIR="./ckpt/EXP_0225_SAR2IR_SAR2RGB_DESPECKLE/dbim/sar2rgb"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-latest}"

# --- Logging ---
SWANLOG_DIR="./ckpt/swanlog"
SWANLAB_EXPERIMENT_NAME="exp-0225-sar2rgb-despeckle"
SWANLAB_DESCRIPTION="EXP-0225 DBIM SAR→RGB with SAR despeckling, CFG dropout and REPA"
SWANLAB_TAGS="dbim,exp-0225,sar2rgb,despeckle,cfg,repa"

COMMON_ARGS=(
  --log_with swanlab
  --swanlab_experiment_name "${SWANLAB_EXPERIMENT_NAME}"
  --swanlab_description "${SWANLAB_DESCRIPTION}"
  --swanlab_tags "${SWANLAB_TAGS}"
  --swanlab_init_kwargs_json '{"logdir":"'"${SWANLOG_DIR}"'","workspace":"EarthBridge"}'
  --num_channels "${NUM_CHANNELS}"
  --num_res_blocks "${NUM_RES_BLOCKS}"
  --attention_resolutions "${ATTENTION_RESOLUTIONS}"
  --channel_mult "${CHANNEL_MULT}"
  --resolution "${RESOLUTION}"
  --output_resolution "${OUTPUT_RESOLUTION}"
  --use_augmented "${USE_AUGMENTED}"
  --use_random_crop "${USE_RANDOM_CROP}"
  --use_horizontal_flip "${USE_HORIZONTAL_FLIP}"
  --use_vertical_flip "${USE_VERTICAL_FLIP}"
  --exclude_file "${EXCLUDE_FILE}"
  --paired_val_manifest "${PAIRED_VAL_MANIFEST}"
  --use_sar2rgb_sup "${USE_SAR2RGB_SUP}"
  --sar2rgb_sup_manifest "${SAR2RGB_SUP_MANIFEST}"
  --use_sar_despeckle "${USE_SAR_DESPECKLE}"
  --sar_despeckle_kernel_size "${SAR_DESPECKLE_KERNEL_SIZE}"
  --sar_despeckle_strength "${SAR_DESPECKLE_STRENGTH}"
  --optimizer_type "${OPTIMIZER_TYPE}"
  --learning_rate "${LEARNING_RATE}"
  --prodigy_d0 "${PRODIGY_D0}"
  --lr_scheduler "${LR_SCHEDULER}"
  --lr_warmup_steps "${LR_WARMUP_STEPS}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --num_epochs "${NUM_EPOCHS}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --use_ema "${USE_EMA}"
  --save_model_epochs "${SAVE_MODEL_EPOCHS}"
  --checkpointing_steps "${CHECKPOINTING_STEPS}"
  --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT}"
  --validation_steps "${VALIDATION_STEPS}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --mixed_precision "${MIXED_PRECISION}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --seed "${SEED}"
  --conditioning_dropout_prob "${CONDITIONING_DROPOUT_PROB}"
  --use_rep_alignment "${USE_REP_ALIGNMENT}"
  --rep_alignment_model_path "${REP_ALIGNMENT_MODEL_PATH}"
  --lambda_rep_alignment "${LAMBDA_REP_ALIGNMENT}"
  --lambda_rep_alignment_decay_steps "${LAMBDA_REP_ALIGNMENT_DECAY_STEPS}"
  --lambda_rep_alignment_end "${LAMBDA_REP_ALIGNMENT_END}"
  --push_to_hub "${PUSH_TO_HUB}"
  --hub_model_id "${HUB_MODEL_ID}"
  --output_dir "${OUTPUT_DIR}"
  --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}"
)

if [ "${NGPU}" -gt 1 ]; then
  nohup accelerate launch --num_processes "${NGPU}" -m examples.dbim.train_sar2rgb \
    "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
else
  nohup python -m examples.dbim.train_sar2rgb \
    "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
fi
