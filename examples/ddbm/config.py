# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Task-specific configurations for DDBM baseline training.

Each task (sar2eo, rgb2ir, sar2ir, sar2rgb) defines its own config with
resolution, channel layout, model architecture, and training hyper-parameters.
Configs are plain dataclasses so per-task scripts can override any field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class TaskConfig:
    """Configuration for a single DDBM image-to-image translation task."""

    # ---- task identity ----
    task_name: str = ""

    # ---- data ----
    source_channels: int = 1
    target_channels: int = 1
    model_channels: int = 1  # channels the DDBM UNet operates in
    resolution: int = 512
    use_augmented: bool = True  # also load *_crop_aug training split
    use_random_crop: bool = True  # random runtime crop to resolution during train
    use_horizontal_flip: bool = False  # random horizontal flip augmentation
    use_vertical_flip: bool = False  # random vertical flip augmentation

    # ---- sample filtering ----
    exclude_file: Optional[str] = "datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"  # path to txt of bad image paths to skip

    # ---- model ----
    # Backbone: adm (default) | edm | vdm | pixnerd
    # Note: edm2 disabled (pipeline incompatible)
    # pixnerd: PixNerd DiT + NerfBlock (pixel-space transformer with neural field decoder)
    unet_type: str = "adm"
    num_channels: int = 128
    num_res_blocks: int = 2
    attention_resolutions: str = "64,32"  # ADM-style; 64px and below
    dropout: float = 0.0
    condition_mode: str = "concat"
    channel_mult: str = ""  # auto-detected from resolution when empty
    attention_head_dim: Optional[int] = 64  # ADM-style; stabilizes training

    # ---- scheduler ----
    pred_mode: str = "vp"
    sigma_max: float = 1.0
    sigma_min: float = 0.002
    sigma_data: float = 0.5
    beta_d: float = 2.0
    beta_min: float = 0.1
    # ---- training ----
    output_dir: str = "./ckpt"
    train_batch_size: int = 8
    num_epochs: int = 100
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    optimizer_type: str = "prodigy"  # "prodigy" | "adamw" | "muon"
    learning_rate: float = 1.0  # Prodigy adapts lr; set to 1.0 by default
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    weight_decay: float = 0.01
    prodigy_d0: float = 1e-5  # Prodigy d0 parameter (initial estimate of D)
    use_ema: bool = True
    ema_decay: float = 0.9999

    # ---- logging / checkpointing ----
    # Accelerate log_with: "tensorboard" | "wandb" | "swanlab" | "all" | comma-separated
    # SwanLab: one-line integration via log_with="swanlab" (pip install swanlab)
    log_with: str = "tensorboard"
    # SwanLab init kwargs (used only when log_with contains "swanlab")
    swanlab_experiment_name: Optional[str] = None
    swanlab_description: Optional[str] = None
    swanlab_tags: Optional[str] = None  # comma-separated tags
    swanlab_init_kwargs_json: Optional[str] = None  # JSON object merged into init_kwargs["swanlab"]
    save_model_epochs: Optional[int] = 1
    checkpointing_steps: Optional[int] = None
    checkpoints_total_limit: int = 1
    resume_from_checkpoint: Optional[str] = None

    # ---- validation ----
    validation_epochs: Optional[int] = None  # run validation every N epochs
    validation_steps: Optional[int] = None   # run validation every N steps
    paired_val_manifest: Optional[str] = None  # path to paired_val_<task>.txt for metric evaluation
    sar2rgb_sup_manifest: Optional[str] = None  # path to paired_sar2rgb_sup.txt (extra supervised SAR→RGB train data)
    use_sar2rgb_sup: bool = False  # when True and task is sar2rgb, add sar2rgb_sup pairs (OpenEarthMap-SAR, SpaceNet6, FUSAR-Map)

    # ---- hub ----
    push_to_hub: bool = True
    hub_model_id: Optional[str] = None

    # ---- hardware ----
    mixed_precision: str = "bf16"
    dataloader_num_workers: int = 4
    seed: int = 42

    # ---- metric-based loss (MAVIC-T evaluation objective) ----
    use_mavic_loss: bool = False   # add LPIPS + L1 loss alongside denoising loss
    mavic_lpips_weight: float = 1.0
    mavic_l1_weight: float = 1.0
    mavic_loss_weight: float = 0.1  # relative weight vs. the denoising loss

    # ---- sampling (evaluation) ----
    num_inference_steps: int = 40
    guidance: float = 1.0
    # Classifier-Free Guidance (CFG) scale. 1.0 disables CFG and preserves legacy behavior.
    cfg_scale: float = 1.0
    churn_step_ratio: float = 0.33
    output_resolution: Optional[int] = None  # if set, load & infer at this resolution (e.g. 1024 when trained at 512)

    # ---- CFG fine-tuning ----
    # Per-sample probability to drop conditioning during training (replace with zeros).
    # Typical range for quick CFG enablement: 0.10 ~ 0.20.
    conditioning_dropout_prob: float = 0.0

    # ---- SAR-specific preprocessing ----
    # Light despeckling on SAR conditioning image to reduce overfitting to
    # high-frequency speckle "fingerprints" in weakly aligned SAR->X pairs.
    use_sar_despeckle: bool = False
    sar_despeckle_kernel_size: int = 5
    sar_despeckle_strength: float = 0.6

    # ---- latent modeling ablation ----
    use_latent_target: bool = False
    latent_vae_path: Optional[str] = None  # path to pre-trained VAE checkpoint
    lambda_latent: float = 1.0  # weight for latent-space L2 loss
    latent_channels: int = 32  # VAE latent dimension (overrides model_channels when use_latent_target=True)

    # ---- representation alignment ----
    use_rep_alignment: bool = False
    rep_alignment_model_path: Optional[str] = None  # path to encoder checkpoint
    lambda_rep_alignment: float = 0.1  # weight for alignment loss (constant, or start when schedule used)
    lambda_rep_alignment_decay_steps: int = 0  # 0 = constant; >0 = cosine decay from lambda_rep_alignment to lambda_rep_alignment_end
    lambda_rep_alignment_end: float = 0.0  # end value when using cosine decay


# ---------------------------------------------------------------------------
# Pre-built configs for the four core tasks
# ---------------------------------------------------------------------------

def _default_paired_val_manifest(task_name: str) -> str:
    return f"datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_{task_name}.txt"


def sar2eo_config(**overrides) -> TaskConfig:
    """SAR-to-EO: native 1024×1024, runtime random-crop to 512×512."""
    cfg = TaskConfig(
        task_name="sar2eo",
        paired_val_manifest=_default_paired_val_manifest("sar2eo"),
        source_channels=1,
        target_channels=1,
        model_channels=1,
        resolution=512,
        output_dir="./ckpt",
        train_batch_size=32,
        # latent modeling (VAE encoder from BiliSakura/VAEs)
        latent_vae_path="./models/BiliSakura/VAEs/FLUX2-VAE",
        # representation alignment not applicable – no pre-trained EO encoder
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def rgb2ir_config(**overrides) -> TaskConfig:
    """RGB-to-IR: 3-band → 1-band (native 1024×1024, train at 512×512 crop)."""
    cfg = TaskConfig(
        task_name="rgb2ir",
        paired_val_manifest=_default_paired_val_manifest("rgb2ir"),
        source_channels=3,
        target_channels=1,
        model_channels=3,  # operate in 3-ch space; 1-ch target is expanded
        resolution=512,
        num_channels=160,  # huge tier for 1024px
        attention_resolutions="128,64,32",
        channel_mult="1,1,2,2,4,8",
        use_augmented=True,
        output_dir="./ckpt",
        train_batch_size=8,
        # latent modeling ablation (VAE encoder from BiliSakura/VAEs)
        latent_vae_path="./models/BiliSakura/VAEs/FLUX2-VAE",
        # representation alignment not applicable – no pre-trained IR encoder
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2ir_config(**overrides) -> TaskConfig:
    """SAR-to-IR: 1-band → 1-band (native 1024×1024, train at 512×512 crop)."""
    cfg = TaskConfig(
        task_name="sar2ir",
        paired_val_manifest=_default_paired_val_manifest("sar2ir"),
        source_channels=1,
        target_channels=1,
        model_channels=1,
        resolution=512,
        num_channels=160,  # huge tier for 1024px
        attention_resolutions="128,64,32",
        channel_mult="1,1,2,2,4,8",
        use_augmented=True,
        use_sar_despeckle=True,
        output_dir="./ckpt",
        train_batch_size=8,
        # latent modeling (VAE encoder from BiliSakura/VAEs)
        latent_vae_path="./models/BiliSakura/VAEs/FLUX2-VAE",
        # representation alignment not applicable – no pre-trained IR encoder
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _default_sar2rgb_sup_manifest() -> str:
    return "datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_sar2rgb_sup.txt"


def sar2rgb_config(**overrides) -> TaskConfig:
    """SAR-to-RGB: 1-band → 3-band (native 1024×1024, train at 512×512 crop)."""
    cfg = TaskConfig(
        task_name="sar2rgb",
        paired_val_manifest=_default_paired_val_manifest("sar2rgb"),
        sar2rgb_sup_manifest=_default_sar2rgb_sup_manifest(),
        use_sar2rgb_sup=True,  # include OpenEarthMap-SAR, SpaceNet6, FUSAR-Map; set False to disable
        source_channels=1,
        target_channels=3,
        model_channels=3,  # operate in 3-ch space; 1-ch source is expanded
        resolution=512,
        num_channels=160,  # huge tier for 1024px
        attention_resolutions="128,64,32",
        channel_mult="1,1,2,2,4,8",
        use_augmented=True,
        use_sar_despeckle=True,
        output_dir="./ckpt",
        train_batch_size=8,
        # latent modeling (VAE encoder from BiliSakura/VAEs)
        latent_vae_path="./models/BiliSakura/VAEs/FLUX2-VAE",
        # representation alignment via MaRS-RGB (encode the RGB target)
        rep_alignment_model_path="./models/BiliSakura/MaRS-Base-RGB",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
