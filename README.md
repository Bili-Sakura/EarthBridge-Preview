# EarthBridge-Preview

**EarthBridge: A Solution for the 4th Multi-modal Aerial View Image Challenge (MAVIC-T) — Translation Track**

This repository is a **preview release** of the EarthBridge codebase, containing the **DBIM**, **DDBM**, and **CUT** baselines used in our competition solution, along with their related training, inference, and evaluation code.

## Baselines Included

| Baseline | Reference | Description |
|----------|-----------|-------------|
| **DDBM** | [ICLR 2024](https://openreview.net/forum?id=FKksTayvGo) | Denoising Diffusion Bridge Models |
| **DBIM** | [OpenReview 2025](https://openreview.net/forum?id=eghAocvqBk) | Diffusion Bridge Implicit Models |
| **CUT** | [ECCV 2020](https://link.springer.com/chapter/10.1007/978-3-030-58545-7_19) | Contrastive Unpaired Translation |

## Installation

1. Install from `requirements.txt` (recommended)

```bash
conda create -n rsgen python=3.12
conda activate rsgen
# we are using PyTorch 2.8.0 torchaudio 2.8.0 torchvision 0.23.0 from https://download.pytorch.org/whl/cu126
# other version mostly would work as long installed follow https://pytorch.org/get-started/previous-versions/
pip install torch==2.8.0+cu126 torchaudio==2.8.0+cu126 torchvision==0.23.0+cu126 --index-url https://download.pytorch.org/whl/cu126
# install other packages
pip install -r requirements.txt
pip install swanlab
```

2. Install from `environment.yaml`

```bash
conda env create -f environment.yaml
conda activate rsgen
```

### Path configuration (optional)

If you clone the repo to a custom location, set `PROJECT_ROOT` to your project directory. Scripts will then resolve paths relative to it.

```bash
# Option 1: Source paths.env (auto-detects project root from file location)
source paths.env

# Option 2: Set manually before running scripts
export PROJECT_ROOT=/path/to/EarthBridge-Preview
```

### Project structure

| Directory | Purpose |
| :--- | :--- |
| **`datasets/`** | `BiliSakura/MACIV-T-2025-Structure-Refined`: `manifests/`, `{task}/train/{input,target}/`, `val/{task}/input/`, `test/{task}/`. See `docs/dataset.md`. |
| **`models/`** | Pre-trained model weights. |
| **`src/models/`** | Model implementations: `unet_ddbm`, `unet_dbim`, `cut_model`. |
| **`examples/`** | Trainer and sample scripts for ddbm, dbim, cut. |
| **`scripts/`** | Training launchers for DDBM, DBIM, and CUT experiments. |
| **`ckpt/`** | Checkpoints and SwanLab logs from training runs. |

### Pre-trained models (MaRS-Base)

Some scripts use pre-trained MaRS encoders for representation alignment or validation-set creation. Please pre-download them from [HuggingFace/BiliSakura](https://huggingface.co/BiliSakura) to your local `models/` folder:

| Model | HuggingFace ID | Local path |
| :--- | :--- | :--- |
| MaRS-Base-RGB | `BiliSakura/MaRS-Base-RGB` | `models/BiliSakura/MaRS-Base-RGB` |
| MaRS-Base-SAR | `BiliSakura/MaRS-Base-SAR` | `models/BiliSakura/MaRS-Base-SAR` |

```bash
# From project root
mkdir -p models/BiliSakura
huggingface-cli download BiliSakura/MaRS-Base-RGB --local-dir models/BiliSakura/MaRS-Base-RGB
huggingface-cli download BiliSakura/MaRS-Base-SAR --local-dir models/BiliSakura/MaRS-Base-SAR
```

### Experiment tracking with SwanLab

Training scripts support [SwanLab](https://swanlab.cn) for experiment tracking. Install with `pip install swanlab`.

**Enable SwanLab** — The DDBM scripts in `scripts/DDBM_Pixel_Medium-0213/` already use `--log_with swanlab`. For other trainers, add:

```bash
--log_with swanlab
```

**Log location** — SwanLab logs are stored under `./ckpt/swanlog`.

## Quick Start

See [docs/quick_start.md](docs/quick_start.md) for detailed training, inference, and evaluation instructions.

## Documentation

- [Dataset setup](docs/dataset.md)
- [Quick start guide](docs/quick_start.md)
- [Model parameters](docs/model_parameters.md)
- [Pre-trained models](docs/pre_trained_models.md)

## Credits

### Library credits

<a href="https://github.com/huggingface/diffusers">diffusers</a>.

### Reference papers

<a href="https://openreview.net/forum?id=FKksTayvGo">Denoising Diffusion Bridge Models (DDBM, ICLR 2024)</a>

<a href="https://openreview.net/forum?id=eghAocvqBk">Diffusion Bridge Implicit Models (DBIM, 2025)</a>

<a href="https://link.springer.com/chapter/10.1007/978-3-030-58545-7_19">Contrastive Unpaired Translation (CUT, ECCV 2020)</a>
