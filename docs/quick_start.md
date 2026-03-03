# Quick Start

Use this short guide to get a baseline training run, export predictions, and measure MAVIC-T metrics.

## 1. Environment

- Create the provided conda environment:  
  `conda env create -f environment.yaml && conda activate rsgen`
- Place the refined dataset at `datasets/BiliSakura/MACIV-T-2025-Structure-Refined`.
- Note: the challenge is **MAVIC-T**, but the released folder is spelled **MACIV-T-2025-Structure-Refined**—keep that exact name (see `docs/dataset.md` for layout).
- Validation/test inputs live under the same refined root.

## 2. Train a model

This preview release includes three baselines: **DDBM**, **DBIM**, and **CUT**.

### DDBM (Denoising Diffusion Bridge Models)

```bash
# Single GPU sar2eo training
python -m examples.ddbm.train_sar2eo --output_dir ./outputs/ddbm_sar2eo --train_batch_size 4

# Multi-GPU (accelerate)
accelerate launch -m examples.ddbm.train_sar2eo --train_batch_size 8
```

### DBIM (Diffusion Bridge Image-to-Image Models)

```bash
# Single GPU sar2eo training
python -m examples.dbim.train_sar2eo --output_dir ./outputs/dbim_sar2eo --train_batch_size 4

# Multi-GPU (accelerate)
accelerate launch -m examples.dbim.train_sar2eo --train_batch_size 8
```

### CUT (Contrastive Unpaired Translation)

```bash
# Single GPU sar2eo training
python -m examples.cut.train_sar2eo --output_dir ./outputs/cut_sar2eo --train_batch_size 4

# Multi-GPU (accelerate)
accelerate launch -m examples.cut.train_sar2eo --train_batch_size 8
```

Every config field can be overridden on the command line (see `examples/{ddbm,dbim,cut}/config.py`). Checkpoints are written to the chosen `--output_dir`.

### Experiment tracking (SwanLab / TensorBoard / WandB)

All trainers use `accelerate.log_with` for experiment tracking. **SwanLab** is supported:

```bash
# SwanLab (pip install swanlab)
python -m examples.ddbm.train_sar2eo --log_with swanlab

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
# DDBM inference
python -m examples.ddbm.sample \
  --task sar2eo \
  --checkpoint_path ./outputs/ddbm_sar2eo \
  --split test \
  --output_dir ./samples/ddbm_sar2eo

# DBIM inference
python -m examples.dbim.sample \
  --task sar2eo \
  --checkpoint_path ./outputs/dbim_sar2eo \
  --split test \
  --output_dir ./samples/dbim_sar2eo

# CUT inference
python -m examples.cut.sample \
  --task sar2eo \
  --checkpoint_path ./outputs/cut_sar2eo \
  --split test \
  --output_dir ./samples/cut_sar2eo
```

The script reads the evaluation inputs, runs the model, and saves PNGs under `--output_dir`.

## 4. Evaluate locally

For a quick sanity check, compute LPIPS/L1 (and FID when `torchvision` is available):

```python
import torch
from torch.utils.data import DataLoader
from src.utils.metrics import MetricCalculator
from examples.ddbm.dataset_wrapper import MavicTDDBMDataset
from examples.ddbm.config import sar2eo_config

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = sar2eo_config()
calc = MetricCalculator(device=device, compute_fid=False)

# (run inference and update calc with your predictions and targets)
print(calc.compute())  # -> MetricResults(lpips=..., fid=None, l1=..., score=...)
```

## 5. Ready-made training scripts

Pre-configured training scripts are available under `scripts/`:

```bash
# DDBM
bash scripts/DDBM_Pixel_Medium-0213/train_sar2eo.sh

# DBIM
bash scripts/DBIM_Pixel_Medium-0216/train_sar2eo.sh

# CUT
bash scripts/CUT_Scaled-0218/train_sar2eo.sh
```

See `scripts/README.md` for details on each experiment folder.
