# Quick Start

Use this short guide to get a baseline training run, export predictions, and measure MAVIC-T metrics.

## 1. Environment

- Create the provided conda environment:  
  `conda env create -f environment.yaml && conda activate rsgen`
- Place the refined dataset at `datasets/BiliSakura/MACIV-T-2025-Structure-Refined`.
- Note: the challenge is **MAVIC-T**, but the released folder is spelled **MACIV-T-2025-Structure-Refined**—keep that exact name (see `docs/dataset.md` for layout).
- Validation/test inputs live under the same refined root.

## 2. Train a model

This preview release includes two baselines: **DBIM** and **CUT**, with flagship experiment configurations for each task.

### Flagship Experiments

| Task | Method | Why |
|------|--------|-----|
| sar2eo | DBIM | Best results among our experiments |
| sar2rgb | DBIM | Best results among our experiments |
| rgb2ir | DBIM | Best results among our experiments |
| sar2ir | CUT | Best results among our experiments |

### DBIM (Diffusion Bridge Implicit Models)

```bash
# Single GPU sar2eo training
python -m examples.dbim.train_sar2eo --output_dir ./outputs/dbim_sar2eo --train_batch_size 4

# Multi-GPU (accelerate)
accelerate launch -m examples.dbim.train_sar2eo --train_batch_size 8
```

### CUT (Contrastive Unpaired Translation)

```bash
# Single GPU sar2ir training (flagship task for CUT)
python -m examples.cut.train_sar2ir --output_dir ./outputs/cut_sar2ir --train_batch_size 4

# Multi-GPU (accelerate)
accelerate launch -m examples.cut.train_sar2ir --train_batch_size 8
```

Every config field can be overridden on the command line (see `examples/{dbim,cut}/config.py`). Checkpoints are written to the chosen `--output_dir`.

### Experiment tracking (SwanLab / TensorBoard / WandB)

All trainers use `accelerate.log_with` for experiment tracking. **SwanLab** is supported:

```bash
# SwanLab (pip install swanlab)
python -m examples.dbim.train_sar2eo --log_with swanlab

# Set SwanLab run metadata from CLI
python -m examples.cut.train_sar2ir \
  --log_with swanlab \
  --swanlab_experiment_name cut-sar2ir-exp01 \
  --swanlab_tags baseline,stage2 \
  --swanlab_description "CUT baseline on SAR->IR"
```

> [!IMPORTANT]
> **Negative Training Loss**
> When training with **Representation Alignment (REPA)** enabled (`--use_rep_alignment true`), it is normal and expected for the total loss to become **negative**. This happens because the alignment loss is calculated as **negative cosine similarity** (ranging from -1 to 1). As the model successfully aligns its features with the pre-trained encoder, this component will move toward -1.0, often pushing the total loss below zero. This indicates healthy convergence.

## 3. Run inference

After training, generate predictions for the val/test inputs:

```bash
# DBIM inference (sar2eo example)
python -m examples.dbim.sample \
  --task sar2eo \
  --checkpoint_path ./outputs/dbim_sar2eo \
  --split test \
  --output_dir ./samples/dbim_sar2eo

# CUT inference (sar2ir — flagship task for CUT)
python -m examples.cut.sample \
  --task sar2ir \
  --checkpoint_path ./outputs/cut_sar2ir \
  --split test \
  --output_dir ./samples/cut_sar2ir
```

The script reads the evaluation inputs, runs the model, and saves PNGs under `--output_dir`.

## 4. Evaluate locally

For a quick sanity check, compute LPIPS/L1 (and FID when `torchvision` is available):

```python
import torch
from torch.utils.data import DataLoader
from src.utils.metrics import MetricCalculator
from examples.dbim.dataset_wrapper import MavicTDBIMDataset
from examples.dbim.config import sar2eo_config

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = sar2eo_config()
calc = MetricCalculator(device=device, compute_fid=False)

# (run inference and update calc with your predictions and targets)
print(calc.compute())  # -> MetricResults(lpips=..., fid=None, l1=..., score=...)
```

## 5. Ready-made training scripts

Pre-configured flagship training scripts are available under `scripts/`:

```bash
# DBIM (sar2eo, sar2rgb, rgb2ir)
bash scripts/DBIM_Pixel_Medium-0216/train_sar2eo.sh
bash scripts/DBIM_Pixel_Medium-0216/train_sar2rgb.sh
bash scripts/DBIM_Pixel_Medium-0216/train_rgb2ir.sh

# CUT (sar2ir)
bash scripts/CUT_Scaled-0218/train_sar2ir.sh
```

See `scripts/README.md` for details on each experiment folder.
