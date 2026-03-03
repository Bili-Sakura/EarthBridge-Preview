#!/usr/bin/env bash
# EXP-0226 CUT — SAR→RGB — test-set inference for submission
#
# Selects checkpoint from the trained tier directory. Override TIER to switch
# between medium (~14.1M), large (~56.5M), and huge (~292.9M) runs.
#
# Usage:
#   bash scripts/EXP_0226_CUT_Scaled/run_cut_sar2rgb.sh
#   TIER=large bash scripts/EXP_0226_CUT_Scaled/run_cut_sar2rgb.sh
#   TIER=huge  CKPT_PATH=/path/to/checkpoint bash scripts/EXP_0226_CUT_Scaled/run_cut_sar2rgb.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

TIER="${TIER:-medium}"          # medium | large | huge
CKPT_PATH="${CKPT_PATH:-}"
MODEL_NAME="${MODEL_NAME:-exp0226_cut_sar2rgb_${TIER}}"
SUBMISSION_ROOT="${SUBMISSION_ROOT:-${PROJECT_ROOT}/datasets/BiliSakura/MACIV-T-2025-Submissions/ckpt-0226}"
DEVICES="${DEVICES:-cuda:0}"
INFER_RESOLUTION="${INFER_RESOLUTION:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"

if [[ -z "${CKPT_PATH}" ]]; then
  BASE_DIR="./ckpt/EXP_0226_CUT_Scaled/cut/sar2rgb_${TIER}_512_1gpu"
  CKPT_PATH="$(ls -d "${BASE_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -z "${CKPT_PATH}" ]]; then
    echo "[ERROR] No checkpoint found under ${BASE_DIR}"
    echo "Set CKPT_PATH manually, e.g.:"
    echo "  CKPT_PATH=./ckpt/EXP_0226_CUT_Scaled/cut/sar2rgb_${TIER}_512_1gpu/checkpoint-100000 \\"
    echo "  bash scripts/EXP_0226_CUT_Scaled/run_cut_sar2rgb.sh"
    exit 1
  fi
fi

if [[ ! -d "${CKPT_PATH}" ]]; then
  echo "[ERROR] Checkpoint directory not found: ${CKPT_PATH}"
  exit 1
fi

read -r -a DEVICE_ARR <<< "${DEVICES}"

echo "=== EXP-0226 CUT SAR→RGB inference (tier=${TIER}) ==="
echo "checkpoint : ${CKPT_PATH}"
echo "output     : ${SUBMISSION_ROOT}/sar2rgb/${MODEL_NAME}"
echo "devices    : ${DEVICES}"
echo "resolution : ${INFER_RESOLUTION}"

python -m examples.cut.sample \
  --task sar2rgb \
  --pretrained_model_name_or_path "${CKPT_PATH}" \
  --split test \
  --output_dir "${SUBMISSION_ROOT}/sar2rgb/${MODEL_NAME}" \
  --batch_size "${BATCH_SIZE}" \
  --resolution "${INFER_RESOLUTION}" \
  --device "${DEVICE_ARR[@]}" \
  "$@"

echo "=== Done ==="
echo "Generated files: ${SUBMISSION_ROOT}/sar2rgb/${MODEL_NAME}"
