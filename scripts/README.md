# Scripts Directory

This directory contains training scripts and utilities for the EarthBridge preview release.

## Flagship Experiments

This preview release includes only the flagship experiment configurations where each method achieved its best results. Due to time limitations, comprehensive baselines, hyperparameter tuning, and scaling studies are not included — these are left for community exploration.

| Task | Method | Script Directory |
|------|--------|------------------|
| **sar2eo** | DBIM | `DBIM_Pixel_Medium-0216/` |
| **sar2rgb** | DBIM | `DBIM_Pixel_Medium-0216/` |
| **rgb2ir** | DBIM | `DBIM_Pixel_Medium-0216/` |
| **sar2ir** | CUT | `CUT_Scaled-0218/` |

### `DBIM_Pixel_Medium-0216/`

DBIM (Diffusion Bridge Implicit Models) training and inference scripts for the three tasks where DBIM achieves best results:

- `train_sar2eo.sh` — Train DBIM on SAR→EO (256×256)
- `train_sar2rgb.sh` — Train DBIM on SAR→RGB (1024×1024)
- `train_rgb2ir.sh` — Train DBIM on RGB→IR (1024×1024)
- `run_sar2eo.sh` — Inference for SAR→EO
- `run_sar2rgb.sh` — Inference for SAR→RGB
- `run_rgb2ir.sh` — Inference for RGB→IR
- `eval_all.sh` — Evaluate all three DBIM tasks on paired validation manifests

### `CUT_Scaled-0218/`

CUT (Contrastive Unpaired Translation) training script for SAR→IR where CUT achieves best results:

- `train_sar2ir.sh` — Train CUT on SAR→IR (1024×1024)

## Utility Scripts

### `document_model_configs.py`

**Purpose:** Generate comprehensive documentation of model architectures and parameter counts.

**Usage:**
```bash
python3 scripts/document_model_configs.py
```

**Output:**
- Creates/updates `docs/model_parameters.md` with detailed configuration and parameter count information.

### `create_paired_validation_set.py`

**Purpose:** Create a paired validation set by randomly sampling 500 pairs per task from the train set. Writes manifest files under `dataset_root/manifests/paired_val_<task>.txt` for use during training validation.

**Usage:**
```bash
python3 scripts/create_paired_validation_set.py
python3 scripts/create_paired_validation_set.py --dataset_root ./datasets/BiliSakura/MACIV-T-2025-Structure-Refined
python3 scripts/create_paired_validation_set.py --n_pairs 500 --seed 42
```

### `analyze_test_dataset.py`

Analyzes test dataset statistics and metadata.

### `filter_bad_samples.py`

Identifies and filters out problematic samples from the dataset.

### `prepare_refined_dataset.py`

Prepares the refined version of the MAVIC-T dataset.

### `prepare_refined_dataset_crop.py`

Prepares cropped versions of the refined dataset with augmentation.

### `test_vae_reconstruction.py`

Tests VAE reconstruction quality on sample images.

### Model Scaling Variants (`configs/model_scaling_variants.yaml`)

**Purpose:** Reference of model-size configurations (small / medium / large / huge) for all baselines and tasks, to support community model-size scaling experiments.

**Location:** `configs/model_scaling_variants.yaml`
