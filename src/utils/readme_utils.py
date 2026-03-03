# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Shared utilities for writing submission readme.txt with detailed descriptions.

Used by all MAVIC-T baseline sample scripts (DBIM, DDBM, BiBBDM, I2SB, DDIB, CUT, Turbo)
to produce consistent, extremely detailed 'Other description' fields including
model training, configurations, and sampling details.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Common training/config keys to include when present in checkpoint config.yaml
_TRAIN_KEYS = [
    "train_batch_size", "num_epochs", "max_train_steps", "gradient_accumulation_steps",
    "optimizer_type", "learning_rate", "lr_scheduler", "lr_warmup_steps", "weight_decay",
    "prodigy_d0", "use_ema", "ema_decay", "mixed_precision", "seed",
]
_MODEL_KEYS = [
    "unet_type", "num_channels", "num_res_blocks", "attention_resolutions",
    "dropout", "condition_mode", "channel_mult", "attention_head_dim",
]
_SCHED_KEYS = [
    "sigma_min", "sigma_max", "sigma_data", "beta_d", "beta_min", "pred_mode",
]


def load_checkpoint_config(checkpoint_path: str) -> dict:
    """Load config.yaml from checkpoint directory if present."""
    config_yaml = Path(checkpoint_path) / "config.yaml"
    if Path(checkpoint_path).is_dir() and config_yaml.exists():
        with config_yaml.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def build_detailed_description(
    *,
    model_name: str,
    model_description: str,
    checkpoint_path: str,
    args: Any,
    cfg: Any,
    checkpoint_config: dict,
    runtime_per_image: float,
    extra_sampling_lines: list[str] | None = None,
) -> str:
    """Build an extremely detailed 'Other description' for readme.txt.

    Args:
        model_name: Short model name (e.g. "DBIM", "DDBM", "BiBBDM").
        model_description: One-line model description.
        checkpoint_path: Path to checkpoint (for config loading and display).
        args: Parsed argparse namespace with task, device, batch_size, etc.
        cfg: TaskConfig with source_channels, target_channels, resolution, etc.
        checkpoint_config: Dict from load_checkpoint_config (may be empty).
        runtime_per_image: Measured runtime per image in seconds.
        extra_sampling_lines: Optional list of model-specific sampling param lines.

    Returns:
        Single-line description string (sections joined with " | ").
    """
    lines = []

    # ---- Model & framework ----
    lines.append("=== MODEL & FRAMEWORK ===")
    lines.append(model_description)
    lines.append(f"Checkpoint path: {checkpoint_path}")
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_dir():
        unet_subfolder = "ema_unet" if (ckpt_path / "ema_unet").is_dir() else "unet"
        lines.append(f"UNet subfolder: {unet_subfolder}")

    # ---- Task configuration ----
    lines.append("")
    lines.append("=== TASK CONFIGURATION ===")
    task = getattr(args, "task", "N/A")
    lines.append(f"Task: {task}")
    for key in ("source_channels", "target_channels", "model_channels"):
        if hasattr(cfg, key):
            lines.append(f"{key}: {getattr(cfg, key)}")
    res = getattr(args, "resolution", None)
    if res is not None:
        lines.append(f"Resolution: {res}")
    elif hasattr(cfg, "resolution"):
        lines.append(f"Resolution: {cfg.resolution}")

    # ---- Training configuration (from checkpoint config.yaml if available) ----
    lines.append("")
    lines.append("=== TRAINING CONFIGURATION ===")
    if checkpoint_config:
        for key in _TRAIN_KEYS + _MODEL_KEYS + _SCHED_KEYS:
            if key in checkpoint_config:
                lines.append(f"  {key}: {checkpoint_config[key]}")
        if checkpoint_config.get("use_mavic_loss"):
            lines.append(
                f"  use_mavic_loss: True (lpips_weight={checkpoint_config.get('mavic_lpips_weight', 1.0)}, "
                f"l1_weight={checkpoint_config.get('mavic_l1_weight', 1.0)}, "
                f"loss_weight={checkpoint_config.get('mavic_loss_weight', 0.1)})"
            )
        if checkpoint_config.get("use_latent_target"):
            lines.append(
                f"  use_latent_target: True (latent_vae_path={checkpoint_config.get('latent_vae_path', 'N/A')})"
            )
        if checkpoint_config.get("use_rep_alignment"):
            lines.append(
                f"  use_rep_alignment: True (rep_alignment_model_path={checkpoint_config.get('rep_alignment_model_path', 'N/A')})"
            )
    else:
        lines.append("  (No config.yaml found in checkpoint; training config unknown)")

    # ---- Sampling / inference configuration ----
    lines.append("")
    lines.append("=== SAMPLING / INFERENCE CONFIGURATION ===")
    if extra_sampling_lines:
        for line in extra_sampling_lines:
            lines.append(line)
    # Common params (use getattr with defaults)
    lines.append(f"Batch size: {getattr(args, 'batch_size', 'N/A')}")
    lines.append(f"Split: {getattr(args, 'split', 'N/A')}")
    lines.append(f"Seed: {getattr(args, 'seed', 'N/A')}")
    if hasattr(args, "deterministic"):
        lines.append(f"Deterministic: {args.deterministic}")

    # ---- Runtime ----
    lines.append("")
    lines.append("=== RUNTIME ===")
    lines.append(f"Runtime per image: {runtime_per_image:.2f}s")
    primary_device = getattr(args, "primary_device", "cpu")
    lines.append(f"Device: {'GPU' if str(primary_device).startswith('cuda') else 'CPU'} ({primary_device})")
    if getattr(args, "use_multi_gpu", False):
        lines.append(f"Multi-GPU: {getattr(args, 'device', [])}")

    # Optional custom description
    readme_desc = getattr(args, "readme_description", None)
    if readme_desc:
        lines.append("")
        lines.append("=== ADDITIONAL NOTES ===")
        lines.append(readme_desc)

    return " | ".join(lines)


def write_readme(
    output_dir: Path,
    *,
    runtime_per_image: float,
    use_gpu: bool,
    extra_data: bool = False,
    description: str,
) -> None:
    """Write readme.txt per official MAVIC-T submission format."""
    readme_path = output_dir / "readme.txt"
    content = f"""runtime per image [s] : {runtime_per_image:.2f}
CPU[1] / GPU[0] : {0 if use_gpu else 1}
Extra Data [1] / No Extra Data [0] : {1 if extra_data else 0}
Other description : {description}
"""
    readme_path.write_text(content, encoding="utf-8")
    logger.info("Wrote %s", readme_path)
