# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Task-specific configurations for DBIM baseline training.

DBIM reuses the DDBM bridge training objective while exposing improved
sampling options (`dbim`, `dbim_high_order`, `heun`) at validation/inference.
"""

from __future__ import annotations

from dataclasses import dataclass

from examples.ddbm.config import (
    TaskConfig as _DDBMTaskConfig,
    sar2eo_config as _ddbm_sar2eo_config,
    rgb2ir_config as _ddbm_rgb2ir_config,
    sar2ir_config as _ddbm_sar2ir_config,
    sar2rgb_config as _ddbm_sar2rgb_config,
)


@dataclass
class TaskConfig(_DDBMTaskConfig):
    """Configuration for a single DBIM image-to-image translation task."""

    sampler: str = "dbim"  # options: "dbim", "dbim_high_order", "heun"
    eta: float = 1.0
    order: int = 2  # used by dbim_high_order
    lower_order_final: bool = True
    clip_denoised: bool = False


def _to_dbim_config(base_cfg: _DDBMTaskConfig) -> TaskConfig:
    cfg = TaskConfig(**vars(base_cfg))
    cfg.sampler = "dbim"
    cfg.eta = 1.0
    cfg.order = 2
    cfg.lower_order_final = True
    cfg.clip_denoised = False
    return cfg


def sar2eo_config(**overrides) -> TaskConfig:
    cfg = _to_dbim_config(_ddbm_sar2eo_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def rgb2ir_config(**overrides) -> TaskConfig:
    cfg = _to_dbim_config(_ddbm_rgb2ir_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2ir_config(**overrides) -> TaskConfig:
    cfg = _to_dbim_config(_ddbm_sar2ir_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def sar2rgb_config(**overrides) -> TaskConfig:
    cfg = _to_dbim_config(_ddbm_sar2rgb_config())
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
