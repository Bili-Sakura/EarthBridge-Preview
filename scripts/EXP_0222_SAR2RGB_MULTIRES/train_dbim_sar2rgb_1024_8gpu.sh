#!/usr/bin/env bash
# EXP-0222 — DBIM — SAR→RGB stage B: direct 1024 fine-tune (multi-resolution)
#
# This stage is intended to resume from stage A (512 crop) checkpoint.
#
# Usage:
#   bash scripts/EXP_0222_SAR2RGB_MULTIRES/train_dbim_sar2rgb_1024_8gpu.sh
#   NGPU=8 bash scripts/EXP_0222_SAR2RGB_MULTIRES/train_dbim_sar2rgb_1024_8gpu.sh
#   RESUME_FROM_CHECKPOINT=/path/to/checkpoint-XXXX bash scripts/EXP_0222_SAR2RGB_MULTIRES/train_dbim_sar2rgb_1024_8gpu.sh

set -euo pipefail

export HF_TOKEN="${HF_TOKEN:-}"
export HF_ENDPOINT="https://hf-mirror.com"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"

NGPU="${NGPU:-8}"
LOG_DIR="./logs/EXP_0222_SAR2RGB_MULTIRES"
LOG_FILE="${LOG_DIR}/train_dbim_sar2rgb_1024_8gpu.log"
mkdir -p "${LOG_DIR}"

# --- Model config (SAR-lite; must match 512 stage for checkpoint loading) ---
NUM_CHANNELS=96
NUM_RES_BLOCKS=2
ATTENTION_RESOLUTIONS="64,32,16"
CHANNEL_MULT="1,1,2,2,4,4"

# --- Data / resolution ---
RESOLUTION=1024
OUTPUT_RESOLUTION=1024
USE_AUGMENTED=true
USE_RANDOM_CROP=false
USE_HORIZONTAL_FLIP=true
USE_VERTICAL_FLIP=true
EXCLUDE_FILE="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"
PAIRED_VAL_MANIFEST="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_sar2rgb.txt"
USE_SAR2RGB_SUP=true
SAR2RGB_SUP_MANIFEST="datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_sar2rgb_sup.txt"

# --- Training ---
OPTIMIZER_TYPE="prodigy"
TRAIN_BATCH_SIZE=8
FURTHER_TRAIN_STEPS="${FURTHER_TRAIN_STEPS:-100000}"  # additional optimizer steps after resume
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"  # optional absolute total-step override
NUM_EPOCHS=0
GRADIENT_ACCUMULATION_STEPS=1
USE_EMA=true
SAVE_MODEL_EPOCHS=0
CHECKPOINTING_STEPS=10000
CHECKPOINTS_TOTAL_LIMIT=1
VALIDATION_STEPS=  # empty = disabled (no validation logging)
NUM_INFERENCE_STEPS=100
MIXED_PRECISION="bf16"
DATALOADER_NUM_WORKERS=8
SEED=42

PUSH_TO_HUB=true
HUB_MODEL_ID="BiliSakura/4th-MAVIC-T-ckpt-0222"
BASE_512_DIR="./ckpt/EXP_0222_SAR2RGB_MULTIRES/dbim/sar2rgb_512"
OUTPUT_DIR="./ckpt/EXP_0222_SAR2RGB_MULTIRES/dbim/sar2rgb_1024"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

# --- SwanLab ---
SWANLOG_DIR="./ckpt/swanlog"
SWANLAB_EXPERIMENT_NAME="exp-0222-sar2rgb-1024"
SWANLAB_DESCRIPTION="EXP-0222 DBIM SAR→RGB 1024 (from 512 stage)"
SWANLAB_TAGS="dbim,exp-0222,sar2rgb"

# Auto-pick latest 512-stage checkpoint when not provided.
if [ -z "${RESUME_FROM_CHECKPOINT}" ]; then
  for candidate_dir in "${BASE_512_DIR}" "${BASE_512_DIR}/dbim/sar2rgb"; do
    if [ -d "${candidate_dir}" ]; then
      latest_ckpt="$(ls -d "${candidate_dir}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"
      if [ -n "${latest_ckpt}" ]; then
        RESUME_FROM_CHECKPOINT="${latest_ckpt}"
        break
      fi
    fi
  done
fi

if [ -z "${RESUME_FROM_CHECKPOINT}" ]; then
  echo "[ERROR] Missing resume checkpoint for SAR2RGB 1024 tuning."
  echo "Set RESUME_FROM_CHECKPOINT or ensure 512-stage checkpoints exist in:"
  echo "  ${BASE_512_DIR}"
  echo "  ${BASE_512_DIR}/dbim/sar2rgb"
  exit 1
fi

CKPT_BASENAME="$(basename "${RESUME_FROM_CHECKPOINT%/}")"
if [[ "${CKPT_BASENAME}" =~ ^checkpoint-([0-9]+)$ ]]; then
  RESUME_STEP="${BASH_REMATCH[1]}"
else
  echo "[ERROR] Resume checkpoint must use 'checkpoint-<step>' naming for further training."
  echo "Got: ${CKPT_BASENAME}"
  exit 1
fi

if [ -z "${MAX_TRAIN_STEPS}" ]; then
  MAX_TRAIN_STEPS=$((RESUME_STEP + FURTHER_TRAIN_STEPS))
fi
echo "[INFO] Resume step: ${RESUME_STEP}; target max_train_steps: ${MAX_TRAIN_STEPS}"

# --- Optional extras ---
USE_LATENT_TARGET=false
USE_REP_ALIGNMENT=false
LAMBDA_REP_ALIGNMENT=0.1
USE_MAVIC_LOSS=false
SAMPLER="dbim"

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
  --optimizer_type "${OPTIMIZER_TYPE}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --num_epochs "${NUM_EPOCHS}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --use_ema "${USE_EMA}"
  --save_model_epochs "${SAVE_MODEL_EPOCHS}"
  --checkpointing_steps "${CHECKPOINTING_STEPS}"
  --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --mixed_precision "${MIXED_PRECISION}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --seed "${SEED}"
  --push_to_hub "${PUSH_TO_HUB}"
  --hub_model_id "${HUB_MODEL_ID}"
  --output_dir "${OUTPUT_DIR}"
  --use_latent_target "${USE_LATENT_TARGET}"
  --use_rep_alignment "${USE_REP_ALIGNMENT}"
  --lambda_rep_alignment "${LAMBDA_REP_ALIGNMENT}"
  --use_mavic_loss "${USE_MAVIC_LOSS}"
  --sampler "${SAMPLER}"
)

if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
  COMMON_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [ -n "${VALIDATION_STEPS}" ]; then
  COMMON_ARGS+=(--validation_steps "${VALIDATION_STEPS}")
fi

if [ "${NGPU}" -gt 1 ]; then
  nohup accelerate launch --num_processes "${NGPU}" -m examples.dbim.train_sar2rgb \
    "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
else
  nohup python -m examples.dbim.train_sar2rgb \
    "${COMMON_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
fi
