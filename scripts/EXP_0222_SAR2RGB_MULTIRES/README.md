# EXP_0222_SAR2RGB_MULTIRES

DBIM-only follow-up experiment set.

- `sar2eo`: no action in this experiment.
- `rgb2ir`: use known good 512 checkpoint, then quick 1024 tuning.
- `sar2ir`: use EXP-0221 512 checkpoint, then 1024 tuning.
- `sar2rgb`: 512 stage first, then 1024 stage.

All scripts default to `NGPU=8`.

## RGB→IR (quick 1024 tuning from existing 512 checkpoint)

- `train_dbim_rgb2ir_1024_quick_from_512_8gpu.sh`
- Resume is required; script auto-searches known checkpoint paths.
- Further-training steps are counted as additional steps:
  - default `FURTHER_TRAIN_STEPS=40000`
  - effective `max_train_steps = resume_step + FURTHER_TRAIN_STEPS`
  - optional absolute override: `MAX_TRAIN_STEPS=<total_step_cap>`

## SAR→IR (1024 tuning from EXP-0221)

- `train_dbim_sar2ir_1024_from_0221_8gpu.sh`
- Resume is required; script auto-searches:
  - `./ckpt/EXP_0221_SAR2IR/dbim/sar2ir_512`
- Further-training steps are counted as additional steps:
  - default `FURTHER_TRAIN_STEPS=80000`
  - effective `max_train_steps = resume_step + FURTHER_TRAIN_STEPS`
  - optional absolute override: `MAX_TRAIN_STEPS=<total_step_cap>`

## SAR→RGB multi-resolution

### Stage A (512 crop training)

- `train_dbim_sar2rgb_512_8gpu.sh`
- Runtime crop `1024 -> 512`
- SAR-lite architecture:
  - `num_channels=96`
  - `channel_mult="1,1,2,2,4,4"`

### Stage B (direct 1024 fine-tune from Stage A)

- `train_dbim_sar2rgb_1024_8gpu.sh`
- Resume is required; script auto-searches:
  - `./ckpt/EXP_0222_SAR2RGB_MULTIRES/dbim/sar2rgb_512`
- SAR-lite architecture is kept compatible with Stage A for checkpoint loading:
  - `num_channels=96`
  - `channel_mult="1,1,2,2,4,4"`
- Further-training steps are counted as additional steps:
  - default `FURTHER_TRAIN_STEPS=100000`
  - effective `max_train_steps = resume_step + FURTHER_TRAIN_STEPS`
  - optional absolute override: `MAX_TRAIN_STEPS=<total_step_cap>`

For any 1024 tuning script, you can override checkpoint explicitly:

```bash
RESUME_FROM_CHECKPOINT=/path/to/checkpoint-XXXX bash scripts/EXP_0222_SAR2RGB_MULTIRES/train_dbim_sar2rgb_1024_8gpu.sh
```
