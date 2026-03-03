#!/usr/bin/env bash
# EXP-0226 CUT — SAR→EO — test-set inference for submission
#
# Selects checkpoint from the trained tier directory. Override TIER to switch
# between medium (~14.1M) and large (~56.5M) runs.
#
# Usage:
#   bash scripts/EXP_0226_CUT_Scaled/run_cut_sar2eo.sh
#   TIER=large bash scripts/EXP_0226_CUT_Scaled/run_cut_sar2eo.sh
#   TIER=large CKPT_PATH=/path/to/checkpoint bash scripts/EXP_0226_CUT_Scaled/run_cut_sar2eo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

TIER="${TIER:-medium}"          # medium | large
CKPT_PATH="${CKPT_PATH:-}"
MODEL_NAME="${MODEL_NAME:-exp0226_cut_sar2eo_${TIER}}"
SUBMISSION_ROOT="${SUBMISSION_ROOT:-${PROJECT_ROOT}/datasets/BiliSakura/MACIV-T-2025-Submissions/ckpt-0226}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICES="${DEVICES:-cuda:0}"

if [[ -z "${CKPT_PATH}" ]]; then
  BASE_DIR="./ckpt/EXP_0226_CUT_Scaled/cut/sar2eo_${TIER}_256_1gpu"
  CKPT_PATH="$(ls -d "${BASE_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"
  if [[ -z "${CKPT_PATH}" ]]; then
    echo "[ERROR] No checkpoint found under ${BASE_DIR}"
    echo "Set CKPT_PATH manually, e.g.:"
    echo "  CKPT_PATH=./ckpt/EXP_0226_CUT_Scaled/cut/sar2eo_${TIER}_256_1gpu/checkpoint-100000 \\"
    echo "  bash scripts/EXP_0226_CUT_Scaled/run_cut_sar2eo.sh"
    exit 1
  fi
fi

if [[ ! -d "${CKPT_PATH}" ]]; then
  echo "[ERROR] Checkpoint directory not found: ${CKPT_PATH}"
  exit 1
fi

read -r -a DEVICE_ARR <<< "${DEVICES}"

echo "=== EXP-0226 CUT SAR→EO inference (tier=${TIER}) ==="
echo "checkpoint : ${CKPT_PATH}"
echo "output     : ${SUBMISSION_ROOT}/sar2eo/${MODEL_NAME}"
echo "devices    : ${DEVICES}"

python -m examples.cut.sample \
  --task sar2eo \
  --pretrained_model_name_or_path "${CKPT_PATH}" \
  --split test \
  --output_dir "${SUBMISSION_ROOT}/sar2eo/${MODEL_NAME}" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE_ARR[@]}" \
  "$@"

echo "=== Done ==="
echo "Generated files: ${SUBMISSION_ROOT}/sar2eo/${MODEL_NAME}"
