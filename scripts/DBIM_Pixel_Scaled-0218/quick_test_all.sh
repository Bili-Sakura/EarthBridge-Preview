#!/usr/bin/env bash
# Quick smoke test for all four DBIM experiments.
# Runs each one by one with tiny steps to verify:
#   - SwanLab logging
#   - No GPU OOM
#   - Checkpoint saving
#   - Validation/evaluation
#
# Usage: bash scripts/DBIM_Pixel_Scaled-0218/quick_test_all.sh

set -euo pipefail

export HF_TOKEN="${HF_TOKEN:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"

# --- Quick-test overrides (extremely small) ---
MAX_TRAIN_STEPS=40
CHECKPOINTING_STEPS=20
VALIDATION_STEPS=20
NUM_INFERENCE_STEPS=20
PUSH_TO_HUB=true
HUB_MODEL_ID_TEST="BiliSakura/4th-MAVIC-T-ckpt-0218-test"
OUTPUT_BASE="./ckpt/DBIM_Pixel_Scaled-0218_quicktest"
LOG_DIR="./logs"
mkdir -p "${LOG_DIR}" "${OUTPUT_BASE}"

NGPU="${NGPU:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

run_experiment() {
  local name="$1"
  local module="$2"
  shift 2
  local args=("$@")

  echo ""
  echo "========== Running ${name} =========="
  if [ "${NGPU}" -gt 1 ]; then
    accelerate launch --num_processes "${NGPU}" -m "${module}" "${args[@]}"
  else
    python -m "${module}" "${args[@]}"
  fi
  echo "========== ${name} OK =========="
}

# Common args for all experiments
COMMON=(
  --log_with swanlab
  --swanlab_experiment_name "dbim-pixel-scaled-quicktest"
  --swanlab_description "Quick smoke test (SwanLab, OOM, checkpoint, eval)"
  --swanlab_tags "dbim,pixel,quicktest"
  --swanlab_init_kwargs_json '{"logdir":"./ckpt/swanlog","workspace":"EarthBridge"}'
  --use_latent_target false
  --use_augmented true
  --use_horizontal_flip true
  --use_vertical_flip true
  --exclude_file "datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"
  --optimizer_type prodigy
  --use_mavic_loss false
  --gradient_accumulation_steps 1
  --use_ema true
  --save_model_epochs 0
  --checkpointing_steps "${CHECKPOINTING_STEPS}"
  --checkpoints_total_limit 1
  --validation_steps "${VALIDATION_STEPS}"
  --num_epochs 0
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --push_to_hub "${PUSH_TO_HUB}"
  --mixed_precision bf16
  --dataloader_num_workers 2
  --seed 42
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
)

# --- 1. RGB→IR (huge, 1024, ~416M) — configs/model_scaling_variants.yaml ---
run_experiment "RGB→IR" examples.dbim.train_rgb2ir \
  "${COMMON[@]}" \
  --num_channels 160 --num_res_blocks 2 \
  --attention_resolutions "128,64,32" --channel_mult "1,1,2,2,4,8" \
  --resolution 1024 --train_batch_size 8 \
  --use_rep_alignment false --lambda_rep_alignment 0.1 \
  --paired_val_manifest "datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_rgb2ir.txt" \
  --output_dir "${OUTPUT_BASE}/rgb2ir" \
  --hub_model_id "${HUB_MODEL_ID_TEST}"

# --- 2. SAR→IR (huge, 1024, ~416M) ---
run_experiment "SAR→IR" examples.dbim.train_sar2ir \
  "${COMMON[@]}" \
  --num_channels 160 --num_res_blocks 2 \
  --attention_resolutions "128,64,32" --channel_mult "1,1,2,2,4,8" \
  --resolution 1024 --train_batch_size 8 \
  --use_rep_alignment false --lambda_rep_alignment 0.1 \
  --paired_val_manifest "datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_sar2ir.txt" \
  --output_dir "${OUTPUT_BASE}/sar2ir" \
  --hub_model_id "${HUB_MODEL_ID_TEST}"

# --- 3. SAR→RGB (huge, 1024, ~416M, REPA) ---
run_experiment "SAR→RGB" examples.dbim.train_sar2rgb \
  "${COMMON[@]}" \
  --num_channels 160 --num_res_blocks 2 \
  --attention_resolutions "128,64,32" --channel_mult "1,1,2,2,4,8" \
  --resolution 1024 --train_batch_size 8 \
  --use_rep_alignment true --lambda_rep_alignment 1.0 \
  --lambda_rep_alignment_decay_steps 2500 --lambda_rep_alignment_end 0.0 \
  --paired_val_manifest "datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_sar2rgb.txt" \
  --output_dir "${OUTPUT_BASE}/sar2rgb" \
  --hub_model_id "${HUB_MODEL_ID_TEST}"

# --- 4. SAR→EO (medium, 256, ~74M) ---
run_experiment "SAR→EO" examples.dbim.train_sar2eo \
  "${COMMON[@]}" \
  --num_channels 128 --num_res_blocks 2 \
  --attention_resolutions "64,32" --channel_mult "1,2,2,4" \
  --resolution 256 --train_batch_size 256 \
  --use_rep_alignment false --lambda_rep_alignment 0.1 \
  --paired_val_manifest "datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_sar2eo.txt" \
  --output_dir "${OUTPUT_BASE}/sar2eo" \
  --hub_model_id "${HUB_MODEL_ID_TEST}"

echo ""
echo "========== All four quick tests passed =========="
