# Model Parameters Documentation

This document provides detailed information about model architectures and parameter counts
for all baselines in the 4th-MAVIC-T project. All models are initialized from their default
configurations for each task.

**Note:** Parameter counts are **exact**, verified by instantiating each model with PyTorch and
counting parameters via ``sum(p.numel() for p in model.parameters())``.
See also ``configs/model_scaling_variants.yaml`` for size-scaling tiers (small / medium / large / huge).

**1024px recommendation:** For direct 1024x1024 pixel-space diffusion modeling, prefer the standalone SID/SID2 baselines (`examples/sid`, `examples/sid2`) to keep comparisons decoupled from bridge-specific baselines.

**Tasks:**

- `sar2eo`: SAR to EO (optical) translation
- `rgb2ir`: RGB to Infrared translation
- `sar2ir`: SAR to Infrared translation
- `sar2rgb`: SAR to RGB translation

## DDBM Baseline

**Description:** Denoising Diffusion Bridge Models for image-to-image translation.
Uses a UNet architecture with conditioning via concatenation.
For direct 1024x1024 runs with SID/SID2, use the standalone ``examples/sid`` or ``examples/sid2`` baseline.

| Task | Exact Parameters | Resolution | Channels | Model Channels |
|------|---------------------|------------|----------|----------------|
| sar2eo | 120,246,401 | 256×256 | 1→1 | 1 |
| rgb2ir | 120,253,315 | 1024×1024 | 3→1 | 3 |
| sar2ir | 120,246,401 | 1024×1024 | 1→1 | 1 |
| sar2rgb | 120,253,315 | 1024×1024 | 1→3 | 3 |

#### sar2eo

```yaml
task_name: sar2eo
resolution: 256x256
channels: 1 → 1
model_channels: 1
unet_base_channels: 128
num_res_blocks: 2
attention_resolutions: 32,16,8
dropout: 0.0
condition_mode: concat
```

#### rgb2ir

```yaml
task_name: rgb2ir
resolution: 1024x1024
channels: 3 → 1
model_channels: 3
unet_base_channels: 128
num_res_blocks: 2
attention_resolutions: 32,16,8
dropout: 0.0
condition_mode: concat
```

#### sar2ir

```yaml
task_name: sar2ir
resolution: 1024x1024
channels: 1 → 1
model_channels: 1
unet_base_channels: 128
num_res_blocks: 2
attention_resolutions: 32,16,8
dropout: 0.0
condition_mode: concat
```

#### sar2rgb

```yaml
task_name: sar2rgb
resolution: 1024x1024
channels: 1 → 3
model_channels: 3
unet_base_channels: 128
num_res_blocks: 2
attention_resolutions: 32,16,8
dropout: 0.0
condition_mode: concat
```

## CUT Baseline

**Description:** Contrastive Unpaired Translation using contrastive learning.
Uses a ResNet-based generator and PatchGAN discriminator with contrastive loss.

| Task | Generator | Discriminator | Total Parameters | Resolution | Channels |
|------|-----------|---------------|------------------|------------|----------|
| sar2eo | 11,365,633 | 2,762,689 | 14,128,322 | 256×256 | 1→1 |
| rgb2ir | 11,371,905 | 2,762,689 | 14,134,594 | 1024×1024 | 3→1 |
| sar2ir | 11,365,633 | 2,762,689 | 14,128,322 | 1024×1024 | 1→1 |
| sar2rgb | 11,371,907 | 2,764,737 | 14,136,644 | 1024×1024 | 1→3 |

### CUT - Detailed Configuration

#### sar2eo

```yaml
task_name: sar2eo
resolution: 256x256
channels: 1 → 1
generator_base_filters_ngf: 64
discriminator_base_filters_ndf: 64
num_downsampling_layers: 2
num_residual_blocks: 9
discriminator_layers: 3
```

#### rgb2ir

```yaml
task_name: rgb2ir
resolution: 1024x1024
channels: 3 → 1
generator_base_filters_ngf: 64
discriminator_base_filters_ndf: 64
num_downsampling_layers: 2
num_residual_blocks: 9
discriminator_layers: 3
```

#### sar2ir

```yaml
task_name: sar2ir
resolution: 1024x1024
channels: 1 → 1
generator_base_filters_ngf: 64
discriminator_base_filters_ndf: 64
num_downsampling_layers: 2
num_residual_blocks: 9
discriminator_layers: 3
```

#### sar2rgb

```yaml
task_name: sar2rgb
resolution: 1024x1024
channels: 1 → 3
generator_base_filters_ngf: 64
discriminator_base_filters_ndf: 64
num_downsampling_layers: 2
num_residual_blocks: 9
discriminator_layers: 3
```


## DBIM Baseline

**Description:** Diffusion Bridge Implicit Models (DBIM). DBIM shares the same UNet architecture
as DDBM and differs only in the sampling algorithm. Parameter counts are identical to DDBM above.

See `docs/model_parameters.md` DDBM section for exact parameter counts per task.

## Summary

### Parameter Count Comparison (sar2eo task)

| Baseline | Exact Parameters | Architecture Type | Notes |
|----------|---------------------|-------------------|-------|
| DDBM | 120,246,401 | Conditional UNet | Diffusion-based bridge |
| DBIM | 120,246,401 | Conditional UNet | Same UNet as DDBM, different sampler |
| CUT | 14,128,322 | ResNet + PatchGAN | Generator + Discriminator |

### Key Architecture Differences

1. **Diffusion Models (DDBM, DBIM):** Use iterative denoising process
   - DDBM: Bridge diffusion with VP/VE noise schedules, DDPM-style sampler
   - DBIM: Same bridge training as DDBM; uses DBIM/Heun deterministic sampler at inference

2. **CUT:** GAN-based translation
   - PatchGAN + ResNet generator with contrastive patch loss

### Resolution Recommendations

| Task | Default Resolution | Notes |
|------|--------------------|-------|
| sar2eo | 256×256 | Smaller EO images |
| rgb2ir | 1024×1024 | High-res IR translation |
| sar2ir | 1024×1024 | High-res IR translation |
| sar2rgb | 1024×1024 | Full-resolution RGB output |
