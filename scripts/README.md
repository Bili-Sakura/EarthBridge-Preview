# Scripts Directory

This directory contains utility scripts for the 4th-MAVIC-T project.

## Experiment Subfolders

### `EXP_0221_SAR2IR/`

Focused 8-GPU DBIM training for failing task `sar2ir`:

- `train_dbim_sar2ir_8gpu.sh`

Uses runtime random crop (`1024 -> 512`) during training and SAR-specific lighter UNet setting
(`num_channels=96`, `channel_mult="1,1,2,2,4,4"`).

### `EXP_0222_SAR2RGB_MULTIRES/`

8-GPU DBIM-only follow-up scripts:

- `rgb2ir` quick 1024 tuning from known good 512 checkpoint:  
  `train_dbim_rgb2ir_1024_quick_from_512_8gpu.sh`
- `sar2ir` 1024 tuning from EXP-0221 checkpoint:  
  `train_dbim_sar2ir_1024_from_0221_8gpu.sh`
- `sar2rgb` Stage A (crop 512):  
  `train_dbim_sar2rgb_512_8gpu.sh`
- `sar2rgb` Stage B (direct 1024 fine-tuning):  
  `train_dbim_sar2rgb_1024_8gpu.sh`

(`sar2eo` intentionally has no script for this experiment batch.)

## Available Scripts

### `document_model_configs.py`

**Purpose:** Generate comprehensive documentation of model architectures and parameter counts for all baselines.

**Usage:**
```bash
python3 scripts/document_model_configs.py
```

**Output:** 
- Creates/updates `docs/model_parameters.md` with detailed configuration and parameter count information for all 6 baselines across all 4 tasks (24 total configurations).

**Features:**
- Parses configuration files directly (no PyTorch dependency required)
- Estimates parameter counts based on model architecture
- Generates formatted markdown tables and detailed configuration sections
- Includes architecture comparisons and summaries

**Baselines covered:**
- DDBM (Denoising Diffusion Bridge Models)
- BiBBDM (Bidirectional Brownian Bridge Diffusion Models)
- I2SB (Image-to-Image Schrödinger Bridge)
- DDIB (Dual Diffusion Implicit Bridges)
- CUT (Contrastive Unpaired Translation)
- Img2Img-Turbo (Pix2Pix-Turbo)

**Tasks covered:**
- sar2eo (SAR to EO optical)
- rgb2ir (RGB to Infrared)
- sar2ir (SAR to Infrared)
- sar2rgb (SAR to RGB)

### Model Scaling Variants (`configs/model_scaling_variants.yaml`)

**Purpose:** Comprehensive reference of model-size configurations (small / medium / large / huge) for all baselines and tasks, to support future model-size scaling experiments.

**Location:** `configs/model_scaling_variants.yaml`

**Contents:**
- Four size tiers per baseline with approximate parameter counts and VRAM guidelines
- Per-task channel/resolution settings and recommended starting sizes
- Usage examples showing how to override config fields

**Baselines covered:** DDBM, BiBBDM, I2SB (shared diffusion UNet), DDIB, CUT, Img2Img-Turbo

### `analyze_test_dataset.py`

Analyzes test dataset statistics and metadata.

### `create_paired_validation_set.py`

**Purpose:** Create a paired validation set by randomly sampling 500 pairs per task from the train set. Writes manifest files under `dataset_root/manifests/paired_val_<task>.txt` for use during training validation.

**Usage:**
```bash
python3 scripts/create_paired_validation_set.py
python3 scripts/create_paired_validation_set.py --dataset_root ./datasets/BiliSakura/MACIV-T-2025-Structure-Refined
python3 scripts/create_paired_validation_set.py --n_pairs 500 --seed 42
```

**Output:** One manifest file per task (`paired_val_sar2eo.txt`, `paired_val_rgb2ir.txt`, etc.) with tab-separated `input_path\ttarget_path` lines.

**Training integration:** When `paired_val_manifest` is set in the DDBM config (default: `datasets/.../manifests/paired_val_<task>.txt`), validation runs MAVIC-T competition metrics (LPIPS, L1) on the paired val set and logs them to the experiment tracker.

### `filter_bad_samples.py`

Identifies and filters out problematic samples from the dataset.

### `prepare_refined_dataset.py`

Prepares the refined version of the MAVIC-T dataset.

### `prepare_refined_dataset_crop.py`

Prepares cropped versions of the refined dataset with augmentation.

### `test_vae_reconstruction.py`

Tests VAE reconstruction quality on sample images.
