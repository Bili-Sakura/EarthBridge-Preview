#!/usr/bin/env bash
# Run inference-step ablation on paired_val for all four tasks: sar2ir, sar2eo, sar2rgb, rgb2ir.
# Steps: 1, 10, 100, 1000. Grid: input | ground_truth | pred_1 | pred_10 | pred_100 | pred_1000 per row.
# Output: /data/projects/4th-MAVIC-T/temp/ablation_steps_{task}_*.png
#
# Usage:
#   bash scripts/DBIM_Pixel_Medium-0216/run_ablation_steps.sh
#   bash scripts/DBIM_Pixel_Medium-0216/run_ablation_steps.sh --DEVICE "cuda:1"
#   CKPT_ROOT=/path/to/ckpt bash scripts/DBIM_Pixel_Medium-0216/run_ablation_steps.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CKPT_ROOT="${CKPT_ROOT:-/data/projects/models/hf_models/BiliSakura/4th-MAVIC-T-ckpt-0216/dbim}"
MANIFEST_DIR="${MANIFEST_DIR:-datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests}"
DEVICE="${DEVICE:-cuda:0}"

# Parse command-line overrides
while [[ $# -gt 0 ]]; do
  case "$1" in
    --CKPT_ROOT)
      CKPT_ROOT="$2"
      shift 2
      ;;
    --MANIFEST_DIR)
      MANIFEST_DIR="$2"
      shift 2
      ;;
    --DEVICE)
      DEVICE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

PYTHON="${RSGEN_PYTHON:-/data/miniconda3/envs/rsgen/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=python
fi

run_ablation() {
  local task="$1"
  local ckpt="${CKPT_ROOT}/${task}/checkpoint-100000"
  local manifest="${MANIFEST_DIR}/paired_val_${task}.txt"
  if [[ ! -d "${ckpt}" ]]; then
    echo "  [skip] checkpoint not found: ${ckpt}"
    return 0
  fi
  if [[ ! -f "${manifest}" ]]; then
    echo "  [skip] manifest not found: ${manifest}"
    return 0
  fi
  echo "--- ${task} ---"
  TASK="${task}" CKPT_PATH="${ckpt}" MANIFEST="${manifest}" DEVICE="${DEVICE}" \
    "${PYTHON}" scripts/DBIM_Pixel_Medium-0216/ablation_inference_steps.py
}

echo "=== Ablation: steps 1, 10, 100, 1000 on paired_val (all four tasks) ==="
echo "CKPT_ROOT:   ${CKPT_ROOT}"
echo "MANIFEST_DIR: ${MANIFEST_DIR}"
echo "DEVICE:      ${DEVICE}"
echo "Output:      /data/projects/4th-MAVIC-T/temp/"
echo ""

# run_ablation sar2ir
# echo ""

# run_ablation sar2eo
# echo ""

run_ablation sar2rgb
echo ""

# run_ablation rgb2ir

echo "=== Done (all tasks) ==="
