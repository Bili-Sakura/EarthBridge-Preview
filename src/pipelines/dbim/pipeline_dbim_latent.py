# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Latent-space DBIM inference pipeline.

Runs DBIM sampling in the latent space of a frozen VAE while exposing the
same pixel-space input/output interface as other MAVIC-T pipelines.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import BaseOutput

from src.models.unet_dbim import DBIMUNet
from src.schedulers.scheduling_dbim import DBIMScheduler
from src.utils.multidiffusion import DEFAULT_LATENT_WINDOW_SIZE
from .pipeline_dbim import DBIMSamplingMixin


@dataclass
class DBIMLatentPipelineOutput(BaseOutput):
    """Output class for DBIM latent pipeline."""

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int = 0
    sampler: str = "dbim"


class DBIMLatentPipeline(DiffusionPipeline, DBIMSamplingMixin):
    """DBIM pipeline operating in VAE latent space."""

    model_cpu_offload_seq = "vae->unet"

    def __init__(
        self,
        unet: DBIMUNet,
        scheduler: DBIMScheduler,
        vae: AutoencoderKL,
    ):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, vae=vae)
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def prepare_inputs(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert PIL/numpy/tensor inputs to `[-1, 1]` tensors."""
        if isinstance(image, Image.Image):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], Image.Image):
            images = []
            for img in image:
                img = img.convert("RGB")
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                images.append(img_tensor)
            image = torch.stack(images)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        if image.max() > 1.0:
            image = image / 255.0
        if image.min() >= 0:
            image = image * 2 - 1

        return image.to(device=device, dtype=dtype)

    def _resize_to_output_size(
        self,
        image: torch.Tensor,
        output_size: Optional[Tuple[int, int]],
    ) -> torch.Tensor:
        """Resize image to (height, width) for MultiDiffusion upsampling (e.g. 512→1024)."""
        if output_size is None:
            return image
        h, w = output_size
        if image.shape[-2] == h and image.shape[-1] == w:
            return image
        return torch.nn.functional.interpolate(
            image, size=(h, w), mode="bilinear", align_corners=False
        )

    @staticmethod
    def _adapt_channels(images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] == 1:
            return images.repeat(1, 3, 1, 1)
        return images

    @staticmethod
    def _restore_channels(images: torch.Tensor, target_channels: int) -> torch.Tensor:
        if target_channels == 1 and images.shape[1] == 3:
            return images.mean(dim=1, keepdim=True)
        return images

    @torch.no_grad()
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        adapted = self._adapt_channels(images)
        posterior = self.vae.encode(adapted).latent_dist
        return posterior.mean * self.vae.config.scaling_factor

    @torch.no_grad()
    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        scaled = latents / self.vae.config.scaling_factor
        return self.vae.decode(scaled).sample

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        num_inference_steps: int = 40,
        sampler: str = "dbim",
        guidance: float = 1.0,
        churn_step_ratio: float = 0.33,
        eta: Optional[float] = None,
        order: Optional[int] = None,
        lower_order_final: Optional[bool] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        target_channels: Optional[int] = None,
        clip_denoised: bool = False,
        output_size: Optional[Tuple[int, int]] = None,
        view_batch_size: int = 1,
        latent_window_size: int = DEFAULT_LATENT_WINDOW_SIZE,
    ):
        """Run DBIM sampling in latent space and decode back to pixel space.

        When the latent grid is larger than the model's training size (e.g. 64 for 512px),
        MultiDiffusion tiling is used automatically so the UNet only sees 64x64 crops.
        Use output_size=(1024, 1024) to upsample a 512px input to 1024px.
        """
        if output_type not in ("pil", "np", "pt"):
            raise ValueError(
                f"Unsupported output_type '{output_type}'. Use one of: pil, np, pt."
            )

        eta_val = self.scheduler.config.eta if eta is None else eta
        order_val = self.scheduler.config.order if order is None else order
        lof_val = (
            self.scheduler.config.lower_order_final
            if lower_order_final is None
            else lower_order_final
        )

        x_pixel = self.prepare_inputs(source_image, self._get_device(), self._get_dtype())
        x_pixel = self._resize_to_output_size(x_pixel, output_size)
        orig_channels = x_pixel.shape[1]

        z_T = self._encode(x_pixel)
        _, _, lh, lw = z_T.shape
        views = None
        # Only use MultiDiffusion tiling when output_size is explicitly set; otherwise run full-res
        if output_size is not None:
            views = self.get_views(lh, lw, window_size=latent_window_size)

        z, nfe = self._run_sampler(
            x_T=z_T,
            num_inference_steps=num_inference_steps,
            sampler=sampler,
            guidance=guidance,
            cfg_scale=1.0,
            churn_step_ratio=churn_step_ratio,
            eta=eta_val,
            order=order_val,
            lower_order_final=lof_val,
            generator=generator,
            callback=callback,
            callback_steps=callback_steps,
            clip_denoised=clip_denoised,
            views=views,
            view_batch_size=view_batch_size,
        )

        images = self._decode(z)
        images = self._restore_channels(images, target_channels or orig_channels)
        images = images.clamp(-1, 1)

        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)

        if not return_dict:
            return images, nfe
        return DBIMLatentPipelineOutput(images=images, nfe=nfe, sampler=sampler)
