# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""DBIM UNet wrappers.

DBIM shares the same bridge parameterization as DDBM but is defined as its own
namespace so that DBIM pipelines do not rely on re-exported DDBM symbols.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

if TYPE_CHECKING:
    from .pixnerd_backbone import PixNerdBackbone

from .unet_ddbm import (
    DDBMUNet,
    EDM2UNet,
    EDMUNet,
    VDMUNet,
    _channel_mult_for_resolution,
    _parse_create_model_args,
    _parse_layers_per_block,
    get_unet_type_config,
    SUPPORTED_UNET_TYPES,
    UNET_TYPE_ADM,
    UNET_TYPE_EDM,
    UNET_TYPE_EDM2,
    UNET_TYPE_PIXNERD,
    UNET_TYPE_VDM,
)


class DBIMUNet(DDBMUNet):
    """DBIM ADM-style UNet with DBIM namespace."""


class DBIMEDMUNet(EDMUNet):
    """DBIM EDM-style UNet with DBIM namespace."""


class DBIMEDM2UNet(EDM2UNet):
    """DBIM EDM2-style UNet (disabled; see create_dbim_model)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ValueError(
            f"DBIM EDM2 UNet is disabled. Use one of {SUPPORTED_UNET_TYPES} via create_dbim_model."
        )


class DBIMVDMUNet(VDMUNet):
    """DBIM VDM-style UNet with DBIM namespace."""


_DBIM_UNET_CLASS_MAP = {
    UNET_TYPE_ADM: DBIMUNet,
    UNET_TYPE_EDM: DBIMEDMUNet,
    UNET_TYPE_VDM: DBIMVDMUNet,
}
# PixNerd uses a different parameter set and is handled explicitly in create_dbim_model.


def create_dbim_model(
    image_size: int = 256,
    in_channels: int = 3,
    num_channels: int = 128,
    num_res_blocks: Union[int, str, Tuple[int, ...]] = 2,
    attention_resolutions: str = "32,16,8",
    dropout: float = 0.0,
    condition_mode: Optional[str] = "concat",
    channel_mult: str = "",
    unet_type: str = UNET_TYPE_ADM,
    attention_head_dim: Optional[int] = 64,
    **kwargs: Any,
) -> Union[DBIMUNet, DBIMEDMUNet, DBIMVDMUNet, PixNerdBackbone]:
    """Factory for DBIM backbone models.

    Mirrors :func:`src.models.unet_ddbm.create_model` but instantiates DBIM
    namespaced classes so DBIM code paths remain self-contained.
    """
    # EDM2 is explicitly disabled for DBIM, so handle it before the general validation.
    if unet_type == UNET_TYPE_EDM2:
        cfg = get_unet_type_config(UNET_TYPE_EDM2)
        issue = cfg.get(
            "issue", "EDM2 preconditioning is incompatible with DBIM pipelines."
        )
        raise ValueError(
            f"unet_type 'edm2' is disabled for DBIM pipelines: {issue}"
        )
    if unet_type not in SUPPORTED_UNET_TYPES:
        raise ValueError(
            f"unet_type '{unet_type}' not supported. Use one of: {SUPPORTED_UNET_TYPES}"
        )

    if unet_type == UNET_TYPE_PIXNERD:
        from .pixnerd_backbone import PixNerdBackbone

        return PixNerdBackbone(
            image_size=image_size,
            in_channels=in_channels,
            hidden_size=kwargs.get("pixnerd_hidden_size", 1152),
            hidden_size_x=kwargs.get("pixnerd_hidden_size_x", 64),
            nerf_mlp_ratio=kwargs.get("pixnerd_nerf_mlp_ratio", 4),
            num_blocks=kwargs.get("pixnerd_num_blocks", 18),
            num_cond_blocks=kwargs.get("pixnerd_num_cond_blocks", 4),
            patch_size=kwargs.get("pixnerd_patch_size", 2),
            num_groups=kwargs.get("pixnerd_num_groups", 12),
            condition_mode=condition_mode,
            dropout=dropout,
        )

    attn_indices, cm_tuple = _parse_create_model_args(
        image_size, attention_resolutions, channel_mult
    )

    channel_mult_resolved = (
        cm_tuple if cm_tuple is not None else _channel_mult_for_resolution(image_size)
    )
    num_levels = len(channel_mult_resolved)
    parsed_num_res_blocks = _parse_layers_per_block(
        num_res_blocks,
        num_levels=num_levels,
        allow_variable=False,
    )

    common_kwargs = dict(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        num_res_blocks=parsed_num_res_blocks,
        attention_resolutions=attn_indices,
        dropout=dropout,
        condition_mode=condition_mode,
        channel_mult=channel_mult_resolved,
        attention_head_dim=attention_head_dim,
    )

    cls = _DBIM_UNET_CLASS_MAP[unet_type]

    if unet_type == UNET_TYPE_VDM:
        if "gamma_min" in kwargs:
            common_kwargs["gamma_min"] = kwargs["gamma_min"]
        if "gamma_max" in kwargs:
            common_kwargs["gamma_max"] = kwargs["gamma_max"]

    return cls(**common_kwargs)


__all__ = [
    "DBIMUNet",
    "DBIMEDMUNet",
    "DBIMEDM2UNet",
    "DBIMVDMUNet",
    "create_dbim_model",
    "get_unet_type_config",
    "SUPPORTED_UNET_TYPES",
    "UNET_TYPE_ADM",
    "UNET_TYPE_EDM",
    "UNET_TYPE_EDM2",
    "UNET_TYPE_VDM",
]
