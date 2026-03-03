#!/usr/bin/env bash
# EXP-0221 — Launch all 8 architecture-search experiments in parallel
# Target: find config with loss < 0.1 (v2/v3 stuck at ~0.15)
#
# Usage:
#   bash scripts/EXP_0221_SAR2IR/run_arch_search_8gpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=== Model parameter counts ==="
python scripts/EXP_0221_SAR2IR/count_params.py --arch-search
echo ""

echo "=== Launching 8 arch-search experiments (20k steps each) ==="
echo "cuda0: 4 stages no attn (sar2eo-style)"
echo "cuda1: 5 stages no attn nc96"
echo "cuda2: 6 stages no attn"
echo "cuda3: 5 stages attn at 32px only"
echo "cuda4: 5 stages no attn num_res_blocks=3"
echo "cuda5: 5 stages no attn nc64 (v3 baseline)"
echo "cuda6: 4 stages no attn nc128"
echo "cuda7: 5 stages no attn nc32 (minimal)"
echo ""

for script in \
  train_dbim_sar2ir_arch_cuda0_4st_noattn.sh \
  train_dbim_sar2ir_arch_cuda1_5st_nc96.sh \
  train_dbim_sar2ir_arch_cuda2_6st_noattn.sh \
  train_dbim_sar2ir_arch_cuda3_attn32.sh \
  train_dbim_sar2ir_arch_cuda4_nrb3.sh \
  train_dbim_sar2ir_arch_cuda5_nc48.sh \
  train_dbim_sar2ir_arch_cuda6_nc128.sh \
  train_dbim_sar2ir_arch_cuda7_nc32.sh; do
  bash "${SCRIPT_DIR}/${script}"
  sleep 5  # stagger launches
done

echo "=== All 8 experiments launched. Check SwanLab / logs ==="
echo "Logs: ./logs/EXP_0221_SAR2IR/early_loss/"
