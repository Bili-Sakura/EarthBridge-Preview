# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Task-specific configurations for CUT baseline training.

Each task (sar2eo, rgb2ir, sar2ir, sar2rgb) defines its own config with
resolution, channel layout, model architecture, and training hyper-parameters.
Configs are plain dataclasses so per-task scripts can override any field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TaskConfig:
    """Configuration for a single CUT image-to-image translation task."""

    # ---- task identity ----
    task_name: str = ""

    # ---- data ----
    source_channels: int = 1
    target_channels: int = 1
    model_channels: int = 3  # channels the CUT generator operates in
    resolution: int = 512
    load_size: Optional[int] = None  # if set: resize to load_size, then crop (if >resolution) or resize (if <resolution)
    use_augmented: bool = True  # also load *_crop_aug training split
    use_random_crop: bool = True  # random runtime crop to resolution during train
    use_horizontal_flip: bool = True
    use_vertical_flip: bool = False

    # ---- sample filtering ----
    exclude_file: Optional[str] = "datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/bad_samples.txt"  # path to txt of bad image paths to skip

    # ---- generator ----
    netG: str = "resnet_9blocks"  # resnet_9blocks | resnet_6blocks
    ngf: int = 64
    normG: str = "instance"
    no_dropout: bool = True
    no_antialias: bool = False
    no_antialias_up: bool = False
    init_type: str = "xavier"
    init_gain: float = 0.02

    # ---- discriminator ----
    netD: str = "basic"  # basic (PatchGAN 70x70)
    ndf: int = 64
    n_layers_D: int = 3
    normD: str = "instance"

    # ---- feature network (PatchSampleF) ----
    netF: str = "mlp_sample"
    netF_nc: int = 256

    # ---- CUT loss ----
    CUT_mode: str = "CUT"  # CUT | FastCUT
    lambda_GAN: float = 1.0
    lambda_NCE: float = 1.0
    nce_idt: bool = True  # identity NCE loss
    nce_layers: str = "0,4,8,12,16"  # layers for NCE feature extraction
    nce_T: float = 0.07  # temperature for contrastive loss
    num_patches: int = 256
    nce_includes_all_negatives_from_minibatch: bool = False
    flip_equivariance: bool = False
    gan_mode: str = "lsgan"  # lsgan | vanilla | wgangp

    # ---- training ----
    output_dir: str = "./outputs/cut"
    train_batch_size: int = 1
    max_grad_norm: float = 1.0  # gradient clipping (G, D, F)
    eval_batch_size: int = 4
    n_epochs: int = 100  # epochs with initial lr
    n_epochs_decay: int = 0  # epochs to linearly decay lr to zero
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    optimizer_type: str = "prodigy"  # "prodigy" | "adam" | "muon"
    learning_rate: float = 1.0  # Prodigy adapts lr; set to 1.0 by default
    beta1: float = 0.5
    beta2: float = 0.999
    lr_policy: str = "linear"  # linear | step | cosine

    # ---- logging / checkpointing ----
    # Accelerate log_with: "tensorboard" | "wandb" | "swanlab" | "all" | comma-separated
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
    validation_resolution: Optional[int] = None  # inference resolution for validation (default: same as resolution)
    paired_val_manifest: Optional[str] = None  # path to paired_val_<task>.txt for golden val (log validation)
    sar2rgb_sup_manifest: Optional[str] = None  # path to paired_sar2rgb_sup.txt (extra supervised SAR→RGB train data)
    use_sar2rgb_sup: bool = False  # when True and task is sar2rgb, add sar2rgb_sup pairs

    # ---- hub ----
    push_to_hub: bool = True
    hub_model_id: Optional[str] = None
    hub_path_tier: Optional[str] = None  # e.g. huge/large/medium -> path cut/{task_name}/{tier}/checkpoint-{step}

    # ---- hardware ----
    mixed_precision: str = "bf16"
    dataloader_num_workers: int = 4
    seed: int = 42

    # ---- attention backend ----
    enable_xformers: bool = False  # use xformers memory-efficient attention (requires xformers)
    enable_flash_attn: bool = False  # use PyTorch 2.0 SDPA / flash-attention backend

    # ---- metric-based loss (MAVIC-T evaluation objective) ----
    use_mavic_loss: bool = False
    mavic_lpips_weight: float = 1.0
    mavic_l1_weight: float = 1.0
    mavic_loss_weight: float = 0.1

    # ---- sampling (evaluation) ----
    num_inference_steps: int = 1  # CUT is single-pass (no iterative denoising)

    # ---- latent modeling ablation ----
    use_latent_target: bool = False
    latent_vae_path: Optional[str] = None  # path to pre-trained VAE checkpoint
    lambda_latent: float = 1.0  # weight for latent-space L2 loss
    latent_channels: int = 32  # VAE latent dimension (overrides model_channels when use_latent_target=True)

    # ---- representation alignment ----
    use_rep_alignment: bool = False
    rep_alignment_model_path: Optional[str] = None  # path to encoder checkpoint
    lambda_rep_alignment: float = 0.1  # weight for alignment loss (constant, or start when schedule used)
    lambda_rep_alignment_decay_steps: int = 0
    lambda_rep_alignment_end: float = 0.0

    # ---- SAR-specific preprocessing ----
    use_sar_despeckle: bool = False
    sar_despeckle_kernel_size: int = 5
    sar_despeckle_strength: float = 0.6


# ---------------------------------------------------------------------------
# Pre-built configs for the four core tasks
# ---------------------------------------------------------------------------

def _default_paired_val_manifest(task_name: str) -> str:
    return f"datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_val_{task_name}.txt"


def _default_sar2rgb_sup_manifest() -> str:
    return "datasets/BiliSakura/MACIV-T-2025-Structure-Refined/manifests/paired_sar2rgb_sup.txt"


def sar2eo_config(**overrides) -> TaskConfig:
    """SAR-to-EO: native 1024×1024, runtime random-crop to 512×512."""
    cfg = TaskConfig(
        task_name="sar2eo",
        paired_val_manifest=_default_paired_val_manifest("sar2eo"),
        source_channels=1,
        target_channels=1,
        model_channels=1,
        resolution=512,
        output_dir="./outputs/cut_sar2eo",
        train_batch_size=4,
        eval_batch_size=16,
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
        model_channels=3,
        resolution=512,
        use_augmented=True,
        output_dir="./outputs/cut_rgb2ir",
        train_batch_size=4,
        eval_batch_size=4,
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
        use_augmented=True,
        use_sar_despeckle=True,
        output_dir="./outputs/cut_sar2ir",
        train_batch_size=4,
        eval_batch_size=4,
        # latent modeling (VAE encoder from BiliSakura/VAEs)
        latent_vae_path="./models/BiliSakura/VAEs/FLUX2-VAE",
        # representation alignment not applicable – no pre-trained IR encoder
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2rgb_config(**overrides) -> TaskConfig:
    """SAR-to-RGB: 1-band → 3-band (native 1024×1024, train at 512×512 crop)."""
    cfg = TaskConfig(
        task_name="sar2rgb",
        paired_val_manifest=_default_paired_val_manifest("sar2rgb"),
        sar2rgb_sup_manifest=_default_sar2rgb_sup_manifest(),
        use_sar2rgb_sup=True,
        source_channels=1,
        target_channels=3,
        model_channels=3,
        resolution=512,
        use_augmented=True,
        use_sar_despeckle=True,
        output_dir="./outputs/cut_sar2rgb",
        train_batch_size=4,
        eval_batch_size=4,
        # latent modeling (VAE encoder from BiliSakura/VAEs)
        latent_vae_path="./models/BiliSakura/VAEs/FLUX2-VAE",
        # representation alignment via MaRS-RGB (encode the RGB target)
        rep_alignment_model_path="./models/BiliSakura/MaRS-Base-RGB",
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
