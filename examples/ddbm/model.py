# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Backward-compatibility shim — imports from ``models.unet``.

New code should import directly from :mod:`src.models`.
"""

from src.models.unet_ddbm import DDBMUNet, create_model, _channel_mult_for_resolution  # noqa: F401

__all__ = ["DDBMUNet", "create_model", "_channel_mult_for_resolution"]
