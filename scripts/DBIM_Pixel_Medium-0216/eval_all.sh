#!/usr/bin/env bash
# DBIM-Pixel-Medium-0216 — Evaluate all four tasks on paired_val manifests (LPIPS, L1, FID).
# Reports score for each task: sar2ir, sar2eo, sar2rgb, rgb2ir.
#
# Default checkpoints (override with CKPT_ROOT or per-task CKPT_<TASK>):
#   CKPT_ROOT/dbim/{sar2ir,sar2eo,sar2rgb,rgb2ir}/checkpoint-100000
#
# Without MultiDiffusion (default): 1024px tasks use original 1024px input (full-res, slow).
#   To use 512px input instead (faster, except sar2eo which stays 256px):
#   RESOLUTION_SAR2IR=512 RESOLUTION_SAR2RGB=512 RESOLUTION_RGB2IR=512 bash scripts/DBIM_Pixel_Medium-0216/eval_all.sh
#
# With MultiDiffusion: OUTPUT_SIZE="1024 1024" for fast tiled inference (val 1024→512 input, tiles→1024).
#   OUTPUT_SIZE="1024 1024" VIEW_BATCH_SIZE=4 bash scripts/DBIM_Pixel_Medium-0216/eval_all.sh
#
# If OOM or dead GPU memory: pkill -9 -f "python.*evaluate"  (kill eval processes)
# Override MultiDiffusion defaults (from src/utils/multidiffusion.py):
#   MD_INPUT_SIZE=512  (source resize before pipeline; default 512)
#   MD_WINDOW_SIZE=512 (tile size in px; default 512)
#   MD_STRIDE=64       (stride between tiles; default 64)
#   e.g. MD_WINDOW_SIZE=256 MD_STRIDE=32 bash scripts/DBIM_Pixel_Medium-0216/eval_all.sh
#
# Usage:
#   bash scripts/DBIM_Pixel_Medium-0216/eval_all.sh
#   bash scripts/DBIM_Pixel_Medium-0216/eval_all.sh --NUM_STEPS 500
#   CKPT_ROOT=/path/to/ckpt bash scripts/DBIM_Pixel_Medium-0216/eval_all.sh
#   MANIFEST_DIR=path/to/manifests bash scripts/DBIM_Pixel_Medium-0216/eval_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

CKPT_ROOT="${CKPT_ROOT:-/data/projects/models/hf_models/BiliSakura/4th-MAVIC-T-ckpt-0216/dbim}"
MANIFEST_DIR="${MANIFEST_DIR:-datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_STEPS="${NUM_STEPS:-100}"
DEVICE="${DEVICE:-cuda:0}"
# MultiDiffusion: off by default. Set OUTPUT_SIZE="1024 1024" for fast tiled 1024 (sar2ir, sar2rgb, rgb2ir).
OUTPUT_SIZE="${OUTPUT_SIZE:-}"
VIEW_BATCH_SIZE="${VIEW_BATCH_SIZE:-1}"
MD_INPUT_SIZE="${MD_INPUT_SIZE:-512}"
MD_WINDOW_SIZE="${MD_WINDOW_SIZE:-}"
MD_STRIDE="${MD_STRIDE:-}"

# Parse command-line overrides (e.g. --NUM_STEPS 500)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --NUM_STEPS)
      NUM_STEPS="$2"
      shift 2
      ;;
    --BATCH_SIZE)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --DEVICE)
      DEVICE="$2"
      shift 2
      ;;
    --CKPT_ROOT)
      CKPT_ROOT="$2"
      shift 2
      ;;
    --MANIFEST_DIR)
      MANIFEST_DIR="$2"
      shift 2
      ;;
    --OUTPUT_SIZE)
      OUTPUT_SIZE="$2"
      shift 2
      ;;
    --VIEW_BATCH_SIZE)
      VIEW_BATCH_SIZE="$2"
      shift 2
      ;;
    --MD_INPUT_SIZE)
      MD_INPUT_SIZE="$2"
      shift 2
      ;;
    --MD_WINDOW_SIZE)
      MD_WINDOW_SIZE="$2"
      shift 2
      ;;
    --MD_STRIDE)
      MD_STRIDE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# Task config: task_name | resolution
# sar2eo uses 256; others use 1024
eval_sar2ir() {
  local res="${RESOLUTION_SAR2IR:-1024}"
  local ckpt="${CKPT_SAR2IR:-${CKPT_ROOT}/sar2ir/checkpoint-100000}"
  local manifest="${MANIFEST_DIR}/paired_val_sar2ir.txt"
  if [[ ! -d "${ckpt}" ]]; then echo "  [skip] checkpoint not found: ${ckpt}"; return 0; fi
  echo "--- sar2ir (resolution=${res}) ---"
  python -m examples.dbim.evaluate_metrics \
    --checkpoint_dir "${ckpt}" \
    --manifest "${manifest}" \
    --task sar2ir \
    --batch_size "${BATCH_SIZE}" \
    --num_inference_steps "${NUM_STEPS}" \
    --resolution "${res}" \
    --device "${DEVICE}" \
    "${EXTRA_MD_ARGS[@]}" \
    "$@"
}

eval_sar2eo() {
  local res="${RESOLUTION_SAR2EO:-256}"
  local ckpt="${CKPT_SAR2EO:-${CKPT_ROOT}/sar2eo/checkpoint-100000}"
  local manifest="${MANIFEST_DIR}/paired_val_sar2eo.txt"
  if [[ ! -d "${ckpt}" ]]; then echo "  [skip] checkpoint not found: ${ckpt}"; return 0; fi
  echo "--- sar2eo (resolution=${res}, no MultiDiffusion) ---"
  python -m examples.dbim.evaluate_metrics \
    --checkpoint_dir "${ckpt}" \
    --manifest "${manifest}" \
    --task sar2eo \
    --batch_size "${BATCH_SIZE}" \
    --num_inference_steps "${NUM_STEPS}" \
    --resolution "${res}" \
    --device "${DEVICE}" \
    "$@"
}

eval_sar2rgb() {
  local res="${RESOLUTION_SAR2RGB:-1024}"
  local ckpt="${CKPT_SAR2RGB:-${CKPT_ROOT}/sar2rgb/checkpoint-100000}"
  local manifest="${MANIFEST_DIR}/paired_val_sar2rgb.txt"
  if [[ ! -d "${ckpt}" ]]; then echo "  [skip] checkpoint not found: ${ckpt}"; return 0; fi
  echo "--- sar2rgb (resolution=${res}) ---"
  python -m examples.dbim.evaluate_metrics \
    --checkpoint_dir "${ckpt}" \
    --manifest "${manifest}" \
    --task sar2rgb \
    --batch_size "${BATCH_SIZE}" \
    --num_inference_steps "${NUM_STEPS}" \
    --resolution "${res}" \
    --device "${DEVICE}" \
    "${EXTRA_MD_ARGS[@]}" \
    "$@"
}

eval_rgb2ir() {
  local res="${RESOLUTION_RGB2IR:-1024}"
  local ckpt="${CKPT_RGB2IR:-${CKPT_ROOT}/rgb2ir/checkpoint-100000}"
  local manifest="${MANIFEST_DIR}/paired_val_rgb2ir.txt"
  if [[ ! -d "${ckpt}" ]]; then echo "  [skip] checkpoint not found: ${ckpt}"; return 0; fi
  echo "--- rgb2ir (resolution=${res}) ---"
  python -m examples.dbim.evaluate_metrics \
    --checkpoint_dir "${ckpt}" \
    --manifest "${manifest}" \
    --task rgb2ir \
    --batch_size "${BATCH_SIZE}" \
    --num_inference_steps "${NUM_STEPS}" \
    --resolution "${res}" \
    --device "${DEVICE}" \
    "${EXTRA_MD_ARGS[@]}" \
    "$@"
}

# Build optional MultiDiffusion args for python (overridable from env / CLI)
EXTRA_MD_ARGS=()
if [[ -n "${OUTPUT_SIZE}" ]]; then
  # Pass H W as two separate args for Python argparse nargs=2
  EXTRA_MD_ARGS+=(--output_size $OUTPUT_SIZE)
fi
EXTRA_MD_ARGS+=(--view_batch_size "${VIEW_BATCH_SIZE}")
EXTRA_MD_ARGS+=(--multidiffusion_input_size "${MD_INPUT_SIZE}")
if [[ -n "${MD_WINDOW_SIZE}" ]]; then
  EXTRA_MD_ARGS+=(--multidiffusion_window_size "${MD_WINDOW_SIZE}")
fi
if [[ -n "${MD_STRIDE}" ]]; then
  EXTRA_MD_ARGS+=(--multidiffusion_stride "${MD_STRIDE}")
fi

echo "=== Evaluating DBIM on all four tasks ==="
echo "CKPT_ROOT:   ${CKPT_ROOT}"
echo "MANIFEST_DIR: ${MANIFEST_DIR}"
echo "BATCH_SIZE:  ${BATCH_SIZE}"
echo "NUM_STEPS:   ${NUM_STEPS}"
echo "DEVICE:      ${DEVICE}"
if [[ -n "${OUTPUT_SIZE}" ]]; then
  echo "OUTPUT_SIZE: ${OUTPUT_SIZE} (MultiDiffusion)"
  echo "VIEW_BATCH_SIZE: ${VIEW_BATCH_SIZE}"
  echo "MD_INPUT_SIZE: ${MD_INPUT_SIZE}"
  [[ -n "${MD_WINDOW_SIZE}" ]] && echo "MD_WINDOW_SIZE: ${MD_WINDOW_SIZE}"
  [[ -n "${MD_STRIDE}" ]] && echo "MD_STRIDE: ${MD_STRIDE}"
fi
echo ""

eval_sar2ir
echo ""

eval_sar2eo
echo ""

eval_sar2rgb
echo ""

eval_rgb2ir

echo "=== Done (all tasks) ==="
