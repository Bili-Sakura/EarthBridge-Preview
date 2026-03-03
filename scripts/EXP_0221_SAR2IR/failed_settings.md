# Arch-Search Settings (EXP-0221)

## Run 1 (USE_AUGMENTED=true, crop_aug dataset)

All 8 configs failed (loss stuck ~0.15, target < 0.1).

## Run 2 (USE_AUGMENTED=false, no crop_aug)

**Worked well:** cuda0, cuda1, cuda2, cuda4, cuda5, cuda6  
**Still failed:** cuda3, cuda7

### Working configs (Run 2)

| Script | NUM_CHANNELS | NUM_RES_BLOCKS | ATTENTION_RESOLUTIONS | CHANNEL_MULT |
|--------|--------------|----------------|------------------------|--------------|
| train_dbim_sar2ir_arch_cuda0_4st_noattn.sh | 64 | 2 | "" | 1,2,3,4 |
| train_dbim_sar2ir_arch_cuda1_5st_nc96.sh | 96 | 2 | "" | 1,2,4,4,8 |
| train_dbim_sar2ir_arch_cuda2_6st_noattn.sh | 64 | 2 | "" | 1,1,2,2,4,4 |
| train_dbim_sar2ir_arch_cuda4_nrb3.sh | 64 | 3 | "" | 1,2,4,4,8 |
| train_dbim_sar2ir_arch_cuda5_nc48.sh | 64 | 2 | "" | 1,2,4,4,8 |
| train_dbim_sar2ir_arch_cuda6_nc128.sh | 128 | 2 | "" | 1,2,3,4 |

### Failed configs (avoid): cuda3, cuda7

| Script | NUM_CHANNELS | NUM_RES_BLOCKS | ATTENTION_RESOLUTIONS | CHANNEL_MULT |
|--------|--------------|----------------|------------------------|--------------|
| train_dbim_sar2ir_arch_cuda3_attn32.sh | 64 | 2 | "32" | 1,2,4,4,8 |
| train_dbim_sar2ir_arch_cuda7_nc32.sh | 32 | 2 | "" | 1,2,4,4,8 |

### Assumptions (why cuda3 and cuda7 failed)

**cuda3 (attn at 32px):** Only config with attention. All working configs use no attention. Possible causes: (a) attention at 32px hurts optimization or doesn't fit SAR→IR inductive bias; (b) attention placement/scale is wrong for this task; (c) attention consumes capacity that convolutions use better. **→ Try:** drop attention, or test other resolutions (16, 64).

**cuda7 (nc32 minimal):** Only config with nc32. Working configs use nc64, nc96, nc128. Likely underfitting — nc32 is below minimum capacity for SAR→IR at 512px. **→ Try:** use nc48 or nc64 as minimum; avoid nc32.

| Config | Likely cause | Suggested direction |
|--------|--------------|---------------------|
| cuda3 (attn32) | Attention at 32px hurts or is poorly suited | Drop attention or try other resolutions |
| cuda7 (nc32) | nc32 too small | Use nc48+ as minimum capacity |

### Chosen arch for 8gpu v4

**cuda1 (5st nc96)** — `train_dbim_sar2ir_8gpu_0221_v4.sh`

- `NUM_CHANNELS=96`, `NUM_RES_BLOCKS=2`, `ATTENTION_RESOLUTIONS=""`, `CHANNEL_MULT="1,2,4,4,8"`
- `USE_AUGMENTED=false` (no crop_aug)

---

## Shared (all configs)

- `RESOLUTION=512`
- `OUTPUT_RESOLUTION=1024`
- `OPTIMIZER_TYPE=prodigy`
- `MAX_TRAIN_STEPS=20000`
- `TRAIN_BATCH_SIZE=8` (except cuda6)
- `NUM_EPOCHS=0`
- `GRADIENT_ACCUMULATION_STEPS=1`
- `USE_EMA=true`
- `MIXED_PRECISION=bf16`
- `USE_LATENT_TARGET=false`
- `USE_REP_ALIGNMENT=false`
- `USE_MAVIC_LOSS=false`
- `SAMPLER=dbim`

## Per-config (model arch only)

| Script | NUM_CHANNELS | NUM_RES_BLOCKS | ATTENTION_RESOLUTIONS | CHANNEL_MULT | TRAIN_BATCH_SIZE |
|--------|--------------|----------------|------------------------|--------------|------------------|
| train_dbim_sar2ir_arch_cuda0_4st_noattn.sh | 64 | 2 | "" | 1,2,3,4 | 8 |
| train_dbim_sar2ir_arch_cuda1_5st_nc96.sh | 96 | 2 | "" | 1,2,4,4,8 | 8 |
| train_dbim_sar2ir_arch_cuda2_6st_noattn.sh | 64 | 2 | "" | 1,1,2,2,4,4 | 8 |
| train_dbim_sar2ir_arch_cuda3_attn32.sh | 64 | 2 | "32" | 1,2,4,4,8 | 8 |
| train_dbim_sar2ir_arch_cuda4_nrb3.sh | 64 | 3 | "" | 1,2,4,4,8 | 8 |
| train_dbim_sar2ir_arch_cuda5_nc48.sh | 64 | 2 | "" | 1,2,4,4,8 | 8 |
| train_dbim_sar2ir_arch_cuda6_nc128.sh | 128 | 2 | "" | 1,2,3,4 | 4 |
| train_dbim_sar2ir_arch_cuda7_nc32.sh | 32 | 2 | "" | 1,2,4,4,8 | 8 |
