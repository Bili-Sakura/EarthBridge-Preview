# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Latent-space inference pipeline for CUT.

Wraps the CUT generator with a frozen VAE so that the generator
operates entirely in latent space while the pipeline accepts and
produces pixel-space images.

This pipeline follows the classic latent modeling pattern established
in diffusers pipelines like StableDiffusionPipeline, where pixel-space
images are encoded into VAE latents, processed in latent space, and
then decoded back to pixel space.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.utils import BaseOutput

from src.models.cut_model import CUTGenerator


@dataclass
class CUTLatentPipelineOutput(BaseOutput):
    """Output class for the CUT latent pipeline.

    Attributes
    ----------
    images : list of PIL.Image.Image, np.ndarray, or torch.Tensor
        Generated images in pixel space.
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]


class CUTLatentPipeline(DiffusionPipeline):
    """CUT pipeline that operates in VAE latent space.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation
    for the generic methods implemented for all pipelines (downloading, saving, running
    on a particular device, etc.).

    The pipeline encodes pixel-space source images into the latent space of a frozen VAE,
    runs the CUT generator in that latent space, and decodes the result back to pixel space.
    This allows the generator to operate entirely in latent space while maintaining a
    pixel-space API.

    Args:
        generator (`CUTGenerator`):
            A CUT generator trained on VAE latents.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from
            latent representations.
    """

    def __init__(self, generator: CUTGenerator, vae: AutoencoderKL) -> None:
        super().__init__()
        self.register_modules(generator=generator, vae=vae)

    @property
    def device(self) -> torch.device:
        """Get the device of the pipeline."""
        return next(self.generator.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the pipeline."""
        return next(self.generator.parameters()).dtype

    # ------------------------------------------------------------------
    # VAE encoding/decoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adapt_channels(images: torch.Tensor) -> torch.Tensor:
        """Adapt single-channel images to 3-channel for VAE encoding.

        Args:
            images: Input tensor of shape (B, C, H, W).

        Returns:
            Tensor with 3 channels if input was 1 channel, otherwise unchanged.
        """
        if images.shape[1] == 1:
            return images.repeat(1, 3, 1, 1)
        return images

    @staticmethod
    def _restore_channels(images: torch.Tensor, target_channels: int) -> torch.Tensor:
        """Restore original channel count after VAE decoding.

        Args:
            images: Decoded tensor of shape (B, C, H, W).
            target_channels: Original number of channels before encoding.

        Returns:
            Tensor with restored channel count.
        """
        if target_channels == 1 and images.shape[1] == 3:
            return images.mean(dim=1, keepdim=True)
        return images

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode pixel-space images to VAE latent space.

        Args:
            images: Pixel-space images in [-1, 1] range, shape (B, C, H, W).

        Returns:
            Latent representations scaled by VAE scaling factor.
        """
        adapted = self._adapt_channels(images)
        posterior = self.vae.encode(adapted).latent_dist
        return posterior.mean * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents to pixel-space images.

        Args:
            latents: Latent representations scaled by VAE scaling factor.

        Returns:
            Pixel-space images in [-1, 1] range.
        """
        scaled = latents / self.vae.config.scaling_factor
        return self.vae.decode(scaled).sample

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Prepare input images for the pipeline.

        Converts PIL images or numpy arrays to normalized tensors in [-1, 1] range.

        Args:
            image: Input image(s) as PIL Image, list of PIL Images, numpy array, or tensor.
            device: Target device. If None, uses pipeline device.
            dtype: Target dtype. If None, uses pipeline dtype.

        Returns:
            Normalized tensor in [-1, 1] range, shape (B, C, H, W).
        """
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        if isinstance(image, Image.Image):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], Image.Image):
            images = []
            for img in image:
                img_array = np.array(img, dtype=np.float32)
                if img_array.max() > 1.0:
                    img_array = img_array / 255.0
                if img_array.ndim == 2:
                    img_array = img_array[:, :, np.newaxis]
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                images.append(img_tensor)
            image = torch.stack(images)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        # Ensure image is in [-1, 1] range
        if image.min() >= 0 and image.max() <= 1.0:
            image = image * 2 - 1  # [0, 1] → [-1, 1]
        elif image.max() > 1.0:
            image = image / 255.0 * 2 - 1  # [0, 255] → [-1, 1]

        return image.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Output conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_to_pil(images: torch.Tensor) -> List[Image.Image]:
        """Convert tensor in [-1, 1] to PIL images.

        Args:
            images: Tensor of shape (B, C, H, W) in [-1, 1] range.

        Returns:
            List of PIL Images.
        """
        images = (images + 1) / 2  # [-1, 1] → [0, 1]
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        pil_images = []
        for img in images:
            if img.shape[2] == 1:
                pil_images.append(Image.fromarray(img.squeeze(2), mode="L"))
            else:
                pil_images.append(Image.fromarray(img))
        return pil_images

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        """Convert tensor in [-1, 1] to numpy array in [0, 1].

        Args:
            images: Tensor of shape (B, C, H, W) in [-1, 1] range.

        Returns:
            Numpy array of shape (B, H, W, C) in [0, 1] range.
        """
        images = (images + 1) / 2  # [-1, 1] → [0, 1]
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        return images

    # ------------------------------------------------------------------
    # Main inference method
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        output_type: str = "pil",
        return_dict: bool = True,
        target_channels: Optional[int] = None,
    ) -> Union[CUTLatentPipelineOutput, tuple]:
        """Translate a source image via CUT in VAE latent space.

        Args:
            source_image (`torch.Tensor`, `PIL.Image.Image`, or `List[PIL.Image.Image]`):
                Source images for translation. Tensors should be in `[-1, 1]` or `[0, 1]` range.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `"pil"`, `"np"`, or `"pt"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.cut_latent_pipeline.CUTLatentPipelineOutput`]
                instead of a plain tuple.
            target_channels (`int`, *optional*):
                Number of channels for the output image. If not provided, 
                defaults to the number of channels in source_image.

        Returns:
            [`~pipelines.cut_latent_pipeline.CUTLatentPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.cut_latent_pipeline.CUTLatentPipelineOutput`]
                is returned, otherwise a `tuple` is returned where the first element is a list with
                the generated images.

        Example:
            ```py
            >>> from src.pipelines.cut import CUTLatentPipeline
            >>> import torch
            >>> from PIL import Image

            >>> pipe = CUTLatentPipeline.from_pretrained("path/to/checkpoint")
            >>> pipe = pipe.to("cuda")

            >>> source_image = Image.open("path/to/image.jpg")
            >>> output = pipe(source_image=source_image)
            >>> image = output.images[0]
            ```
        """
        device = self.device
        dtype = self.dtype

        # Prepare pixel inputs
        x_pixel = self.prepare_inputs(source_image, device, dtype)
        orig_channels = x_pixel.shape[1]

        # Encode to latent space
        z = self.encode_image(x_pixel)

        # Single forward pass in latent space
        z_out = self.generator(z)

        # Decode from latent to pixel space
        images = self.decode_latents(z_out)
        images = self._restore_channels(images, target_channels or orig_channels)
        images = images.clamp(-1, 1)

        # Convert to requested output format
        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)
        # else: output_type == "pt", return tensor as-is

        if not return_dict:
            return (images,)

        return CUTLatentPipelineOutput(images=images)
