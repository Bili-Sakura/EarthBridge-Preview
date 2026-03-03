# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Model architectures for MAVIC-T baselines (preview: DBIM, DDBM, CUT)."""

from .unet_ddbm import (
    DDBMUNet,
    EDMUNet,
    EDM2UNet,
    VDMUNet,
    create_model as create_ddbm_model,
    get_unet_type_config,
    SUPPORTED_UNET_TYPES,
    UNET_TYPE_ADM,
    UNET_TYPE_EDM,
    UNET_TYPE_EDM2,
    UNET_TYPE_VDM,
)
from .unet_dbim import (
    DBIMUNet,
    DBIMEDMUNet,
    DBIMEDM2UNet,
    DBIMVDMUNet,
    create_dbim_model,
)
from .cut_model import (
    CUTGenerator,
    PatchGANDiscriminator,
    PatchSampleMLP,
    GANLoss,
    PatchNCELoss,
    create_generator,
    create_discriminator,
    create_patch_sample_mlp,
)

__all__ = [
    "DDBMUNet", "EDMUNet", "EDM2UNet", "VDMUNet",
    "create_ddbm_model",
    "get_unet_type_config",
    "SUPPORTED_UNET_TYPES",
    "UNET_TYPE_ADM",
    "UNET_TYPE_EDM",
    "UNET_TYPE_EDM2",
    "UNET_TYPE_VDM",
    "DBIMUNet", "DBIMEDMUNet", "DBIMEDM2UNet", "DBIMVDMUNet",
    "create_dbim_model",
    "CUTGenerator", "PatchGANDiscriminator", "PatchSampleMLP",
    "GANLoss", "PatchNCELoss",
    "create_generator", "create_discriminator", "create_patch_sample_mlp",
]
