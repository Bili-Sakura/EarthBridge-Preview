#!/usr/bin/env bash
# Run CFG sweep inference on paired val set (default: task config manifest).
#
# Examples:
#   TASK=sar2ir bash scripts/EXP_0225_SAR2IR_SAR2RGB_DESPECKLE/run_cfg_inference_paired_val.sh
#   TASK=sar2rgb CFG_SCALES="1.0,1.5,2.0,3.0" MAX_SAMPLES=32 bash scripts/EXP_0225_SAR2IR_SAR2RGB_DESPECKLE/run_cfg_inference_paired_val.sh
#   TASK=sar2ir CKPT_DIR=/path/to/checkpoint-120000 bash scripts/EXP_0225_SAR2IR_SAR2RGB_DESPECKLE/run_cfg_inference_paired_val.sh

set -euo pipefail

TASK="${TASK:-sar2ir}"                                  # sar2ir | sar2rgb | sar2eo | rgb2ir
CKPT_DIR="${CKPT_DIR:-}"                                # optional: explicit checkpoint dir
MANIFEST="${MANIFEST:-}"                                # optional: override manifest path
CFG_SCALES="${CFG_SCALES:-1.0,1.25,1.5,2.0,3.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-100}"
GUIDANCE="${GUIDANCE:-1.0}"
SAMPLER="${SAMPLER:-dbim}"                              # dbim | dbim_high_order | heun
CHURN_STEP_RATIO="${CHURN_STEP_RATIO:-0.33}"
ETA="${ETA:-1.0}"
ORDER="${ORDER:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-64}"
DEVICE="${DEVICE:-}"                                    # e.g. cuda:0
OUT_DIR="${OUT_DIR:-./ckpt/EXP_0225_SAR2IR_SAR2RGB_DESPECKLE/cfg_grids/${TASK}}"

if [ -z "${CKPT_DIR}" ]; then
  BASE="./ckpt/EXP_0225_SAR2IR_SAR2RGB_DESPECKLE/dbim/${TASK}"
  CKPT_DIR="$(ls -d "${BASE}"/dbim/"${TASK}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"
  if [ -z "${CKPT_DIR}" ]; then
    CKPT_DIR="$(ls -d "${BASE}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"
  fi
fi

if [ -z "${CKPT_DIR}" ]; then
  echo "[ERROR] Could not locate checkpoint automatically."
  echo "Set CKPT_DIR manually, e.g.:"
  echo "  CKPT_DIR=./ckpt/EXP_0225_SAR2IR_SAR2RGB_DESPECKLE/dbim/${TASK}/dbim/${TASK}/checkpoint-120000"
  exit 1
fi

echo "[INFO] TASK=${TASK}"
echo "[INFO] CKPT_DIR=${CKPT_DIR}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] CFG_SCALES=${CFG_SCALES}"

CMD=(
  python "scripts/EXP_0225_SAR2IR_SAR2RGB_DESPECKLE/infer_cfg_paired_val.py"
  --task "${TASK}"
  --checkpoint_dir "${CKPT_DIR}"
  --output_dir "${OUT_DIR}"
  --cfg_scales "${CFG_SCALES}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --guidance "${GUIDANCE}"
  --sampler "${SAMPLER}"
  --churn_step_ratio "${CHURN_STEP_RATIO}"
  --eta "${ETA}"
  --order "${ORDER}"
  --max_samples "${MAX_SAMPLES}"
)

if [ -n "${MANIFEST}" ]; then
  CMD+=(--manifest "${MANIFEST}")
fi
if [ -n "${DEVICE}" ]; then
  CMD+=(--device "${DEVICE}")
fi

"${CMD[@]}"
