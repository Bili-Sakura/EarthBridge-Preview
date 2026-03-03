# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""DBIM model entry points for examples."""

from src.models.unet_dbim import DBIMUNet, create_dbim_model  # noqa: F401

__all__ = ["DBIMUNet", "create_dbim_model"]
