# Pre-trained Models

This document describes the pre-trained models used in the MAVIC-T project (excluding models trained within this repository).

## Model Paths

All pre-trained models are stored under the `models/` directory with the following structure:

### 1. SARCLIP

**Path:** `models/BiliSakura/SARCLIP-ViT-L-14`

Vision-language model (ViT-L/14) fine-tuned for SAR (Synthetic Aperture Radar) imagery understanding. Can be used as an alternative encoder for representation alignment in SAR2EO, SAR2IR, and SAR2RGB tasks.

### 2. MaRS-SAR

**Path:** `models/BiliSakura/MaRS-Base-SAR`

SwinV2-based image encoder (swinv2_base_window8_256) pre-trained for SAR (Synthetic Aperture Radar) imagery. Used as the default encoder for representation alignment in SAR2EO, SAR2IR, and SAR2RGB tasks. Loaded via `transformers`.

### 3. MaRS-RGB

**Path:** `models/BiliSakura/MaRS-Base-RGB`

SwinV2-based image encoder (swinv2_base_window8_256) pre-trained for RGB imagery. Used as the default encoder for representation alignment in the RGB2IR task. Loaded via `transformers`.

### 4. DINOv3-sat

**Path:** `models/facebook/dinov3-vitl16-pretrain-sat493m`

Self-supervised vision transformer (ViT-L) pre-trained on satellite imagery. Provides robust visual features for remote sensing tasks. Can be used as an alternative encoder for representation alignment in the RGB2IR task.

### 5. VAEs Collection

**Path:** `models/BiliSakura/VAEs`

A collection of Variational Autoencoders (VAEs) used for latent space encoding/decoding in various tasks. Contains multiple VAE checkpoints for different modalities and resolutions.

---

## Usage Notes

- These models are external dependencies and should be downloaded separately if not already present.
- Model paths are referenced relative to the repository root.
- Ensure proper model loading and initialization according to each model's specific requirements.

## Representation Alignment (REPA)

The representation alignment technique (inspired by [REPA](vendor/REPA/)) uses a
frozen pre-trained encoder to extract features from the **target (ground-truth)**
image, while a trainable projection head maps the translation model's output
features into the same embedding space.  A negative-cosine-similarity loss
encourages the model to produce outputs whose representations match the clean
target as seen by the encoder.

> **Important:** Following the original REPA formulation, the teacher encoder
> must process the **target** image (the clean ground-truth), *not* the source
> input.  This means REPA is only applicable when a pre-trained encoder exists
> for the **target domain**.

| Task | REPA Support | Default Encoder | Config field |
|------|:---:|-----------------|-------------|
| `sar2rgb` | ✅ | MaRS-RGB | `rep_alignment_model_path="./models/BiliSakura/MaRS-Base-RGB"` |
| `sar2eo` | ❌ | — | No pre-trained EO encoder available |
| `rgb2ir` | ❌ | — | No pre-trained IR encoder available |
| `sar2ir` | ❌ | — | No pre-trained IR encoder available |

Enable via `use_rep_alignment=True` in the task config.  The alignment loss
weight is controlled by `lambda_rep_alignment` (default 0.1).  For unsupported
tasks, enabling REPA will log a warning and skip the alignment loss.
