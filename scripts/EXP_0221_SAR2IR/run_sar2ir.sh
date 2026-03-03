#!/usr/bin/env bash
# EXP_0221_SAR2IR — SAR->IR test-set inference for submission
#
# Default checkpoint:
#   /data/projects/4th-MAVIC-T/models/BiliSakura/4th-MAVIC-T-ckpt-0221/dbim/sar2ir/checkpoint-50000
#
# Usage:
#   bash scripts/EXP_0221_SAR2IR/run_sar2ir.sh
#   bash scripts/EXP_0221_SAR2IR/run_sar2ir.sh --NUM_STEPS 500 --BATCH_SIZE 8 --DEVICES "cuda:1"
#   CKPT_PATH=/path/to/checkpoint bash scripts/EXP_0221_SAR2IR/run_sar2ir.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CKPT_PATH="${CKPT_PATH:-/data/projects/4th-MAVIC-T/models/BiliSakura/4th-MAVIC-T-ckpt-0221/dbim/sar2ir/checkpoint-50000}"
MODEL_NAME="${MODEL_NAME:-exp_0221_sar2ir}"
SUBMISSION_ROOT="${SUBMISSION_ROOT:-${PROJECT_ROOT}/datasets/BiliSakura/MACIV-T-2025-Submissions/ckpt-0221}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_STEPS="${NUM_STEPS:-100}"
SAMPLER="${SAMPLER:-dbim}"
RESOLUTION="${RESOLUTION:-1024}"
DEVICES="${DEVICES:-cuda:0}"

# Parse command-line overrides (e.g. --NUM_STEPS 500)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --CKPT_PATH)
      CKPT_PATH="$2"
      shift 2
      ;;
    --MODEL_NAME)
      MODEL_NAME="$2"
      shift 2
      ;;
    --SUBMISSION_ROOT)
      SUBMISSION_ROOT="$2"
      shift 2
      ;;
    --BATCH_SIZE)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --NUM_STEPS)
      NUM_STEPS="$2"
      shift 2
      ;;
    --SAMPLER)
      SAMPLER="$2"
      shift 2
      ;;
    --RESOLUTION)
      RESOLUTION="$2"
      shift 2
      ;;
    --DEVICES)
      DEVICES="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "${CKPT_PATH}" ]]; then
  echo "Checkpoint directory not found: ${CKPT_PATH}"
  exit 1
fi

read -r -a DEVICE_ARR <<< "${DEVICES}"

echo "=== Running DBIM sar2ir test inference ==="
echo "checkpoint: ${CKPT_PATH}"
echo "output:     ${SUBMISSION_ROOT}/sar2ir/${MODEL_NAME}"
echo "devices:    ${DEVICES}"

python -m examples.dbim.sample \
  --task sar2ir \
  --pretrained_model_name_or_path "${CKPT_PATH}" \
  --split test \
  --model_name "${MODEL_NAME}" \
  --submission_root "${SUBMISSION_ROOT}" \
  --batch_size "${BATCH_SIZE}" \
  --num_inference_steps "${NUM_STEPS}" \
  --sampler "${SAMPLER}" \
  --resolution "${RESOLUTION}" \
  --device "${DEVICE_ARR[@]}" \
  "$@"

echo "=== Done ==="
echo "Generated files: ${SUBMISSION_ROOT}/sar2ir/${MODEL_NAME}"
echo "To package zip later: python scripts/create_submission_zip.py --submission_root \"${SUBMISSION_ROOT}\" --model_name \"${MODEL_NAME}\" --output \"${SUBMISSION_ROOT}/submission.zip\""
