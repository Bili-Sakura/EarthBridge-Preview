# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""DBIM UNet wrappers.

DBIM is an improved sampler family over the same bridge model parameterization
used by DDBM, so DBIM reuses the exact UNet contracts from ``unet_ddbm``.
"""

from .unet_ddbm import (
    DDBMUNet as DBIMUNet,
    EDMUNet as DBIMEDMUNet,
    EDM2UNet as DBIMEDM2UNet,
    VDMUNet as DBIMVDMUNet,
    create_model as create_dbim_model,
    get_unet_type_config,
    SUPPORTED_UNET_TYPES,
    UNET_TYPE_ADM,
    UNET_TYPE_EDM,
    UNET_TYPE_EDM2,
    UNET_TYPE_VDM,
)

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
