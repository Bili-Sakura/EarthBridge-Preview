#!/usr/bin/env bash
# DDBM-Pixel-Medium-0213 — RGB->IR test-set inference for submission
#
# Usage:
#   CKPT_PATH=/path/to/checkpoint bash scripts/DDBM_Pixel_Medium-0213/run_rgb2ir.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CKPT_PATH="${CKPT_PATH:-}"
MODEL_NAME="${MODEL_NAME:-ddbm_pixel_medium_0213_rgb2ir}"
SUBMISSION_ROOT="${SUBMISSION_ROOT:-${PROJECT_ROOT}/datasets/BiliSakura/MACIV-T-2025-Submissions}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_STEPS="${NUM_STEPS:-100}"
DEVICES="${DEVICES:-cuda:0}"

if [[ -z "${CKPT_PATH}" ]]; then
  echo "Please set CKPT_PATH to the DDBM checkpoint directory."
  exit 1
fi
if [[ ! -d "${CKPT_PATH}" ]]; then
  echo "Checkpoint directory not found: ${CKPT_PATH}"
  exit 1
fi

read -r -a DEVICE_ARR <<< "${DEVICES}"

python -m examples.ddbm.sample \
  --task rgb2ir \
  --pretrained_model_name_or_path "${CKPT_PATH}" \
  --split test \
  --model_name "${MODEL_NAME}" \
  --submission_root "${SUBMISSION_ROOT}" \
  --batch_size "${BATCH_SIZE}" \
  --num_inference_steps "${NUM_STEPS}" \
  --device "${DEVICE_ARR[@]}" \
  "$@"

echo "Generated files: ${SUBMISSION_ROOT}/rgb2ir/${MODEL_NAME}"
