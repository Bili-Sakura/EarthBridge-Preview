# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Latent-space target encoder for ablation studies.

Loads a pre-trained VAE (encoder only) from a local checkpoint and uses it
to encode target images into latent space.  A latent-space L2 loss between
the model's internal latent and the pre-trained VAE's latent encourages the
translation pipeline to produce outputs that are consistent in latent
space with targets encoded by the reference VAE.

This module is shared across all baselines (Pix2Pix-Turbo, CUT, DDBM).

The pre-trained VAE checkpoints are expected at
``models/BiliSakura/VAEs`` (a HuggingFace ``diffusers`` style directory).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import AutoencoderDC, AutoencoderKL, AutoencoderKLFlux2, AutoencoderKLQwenImage
from diffusers.configuration_utils import ConfigMixin

# Map of VAE class names to their classes
# This follows the same pattern as diffusers' AutoModel.from_pretrained
_VAE_CLASSES = {
    "AutoencoderKL": AutoencoderKL,
    "AutoencoderKLFlux2": AutoencoderKLFlux2,
    "AutoencoderKLQwenImage": AutoencoderKLQwenImage,
    "AutoencoderDC": AutoencoderDC,
}


def _detect_vae_class(vae_path: str) -> type:
    """Detect the correct VAE class from config.json.

    This function reads the ``_class_name`` field from the VAE's config.json
    file (similar to how diffusers' AutoModel.from_pretrained works) and
    returns the appropriate VAE class.

    Parameters
    ----------
    vae_path : str
        Path to the VAE checkpoint directory.

    Returns
    -------
    type
        The VAE class to use (defaults to AutoencoderKL if config not found
        or class name not recognized).
    """
    try:
        # Use diffusers' load_config to handle both local and remote paths
        config = ConfigMixin.load_config(vae_path)
        class_name = config.get("_class_name", "AutoencoderKL")
        return _VAE_CLASSES.get(class_name, AutoencoderKL)
    except Exception:
        # Fallback to AutoencoderKL if config loading fails
        return AutoencoderKL


class LatentTargetEncoder(nn.Module):
    """Frozen VAE encoder used to produce latent targets.

    Parameters
    ----------
    vae_path : str
        Local path (or HuggingFace hub id) to a ``diffusers``-style VAE
        checkpoint directory (e.g. ``./models/BiliSakura/VAEs``).
    """

    def __init__(self, vae_path: str) -> None:
        super().__init__()
        vae_class = _detect_vae_class(vae_path)
        self.vae = vae_class.from_pretrained(vae_path)
        self.vae.requires_grad_(False)
        self.vae.eval()
        # Store scaling factor for consistency
        self.scaling_factor: float = self.vae.config.scaling_factor

    @staticmethod
    def _adapt_channels(images: torch.Tensor) -> torch.Tensor:
        """Expand 1-channel images to 3 channels for the VAE.

        When the input has a single channel it is repeated three times so
        that the standard 3-channel VAE can process it.  3-channel inputs
        are returned unchanged.
        """
        if images.shape[1] == 1:
            return images.repeat(1, 3, 1, 1)
        return images

    @staticmethod
    def _restore_channels(images: torch.Tensor, target_channels: int) -> torch.Tensor:
        """Restore channel count after VAE decode."""
        if target_channels == 1 and images.shape[1] == 3:
            return images.mean(dim=1, keepdim=True)
        if target_channels == 3 and images.shape[1] == 1:
            return images.repeat(1, 3, 1, 1)
        return images

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent means (no sampling noise).

        Parameters
        ----------
        images : Tensor (B, C, H, W)
            Images in ``[-1, 1]``.  *C* may be 1 or 3; single-channel
            inputs are automatically expanded to 3 channels.

        Returns
        -------
        Tensor (B, C_latent, H_lat, W_lat)
            Scaled latent representation (deterministic mean).
        """
        images = self._adapt_channels(images)
        posterior = self.vae.encode(images).latent_dist
        return posterior.mean * self.scaling_factor

    @torch.enable_grad()
    def encode_with_grad(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent means **with** gradient flow.

        Same as :meth:`encode` but allows gradients to propagate back
        through the VAE encoder.  Use this to compute latent predictions
        whose loss should update the upstream model.

        Parameters
        ----------
        images : Tensor (B, C, H, W)
            Images in ``[-1, 1]``.  *C* may be 1 or 3; single-channel
            inputs are automatically expanded to 3 channels.

        Returns
        -------
        Tensor (B, C_latent, H_lat, W_lat)
            Scaled latent representation (deterministic mean).
        """
        images = self._adapt_channels(images)
        posterior = self.vae.encode(images).latent_dist
        return posterior.mean * self.scaling_factor

    def decode(self, latents: torch.Tensor, target_channels: int | None = None) -> torch.Tensor:
        """Decode latents back to pixel space (optionally restoring channels)."""
        scaled = latents / self.scaling_factor
        images = self.vae.decode(scaled).sample
        if target_channels is not None:
            images = self._restore_channels(images, target_channels)
        return images
