# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Latent-space inference pipeline for DDBM.

Wraps the DDBM diffusion process with a frozen VAE so that the UNet
operates entirely in latent space while the pipeline accepts and
produces pixel-space images.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from src.schedulers.scheduling_ddbm import DDBMScheduler
from src.models.unet_ddbm import DDBMUNet
from src.utils.multidiffusion import (
    DEFAULT_LATENT_WINDOW_SIZE,
    get_views,
)


@dataclass
class DDBMLatentPipelineOutput(BaseOutput):
    """Output class for the DDBM latent pipeline.

    Attributes
    ----------
    images : list of PIL.Image or ndarray or Tensor
        Generated images in pixel space.
    nfe : int
        Number of function evaluations during sampling.
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int = 0


class DDBMLatentPipeline(DiffusionPipeline):
    """DDBM pipeline that operates in VAE latent space.

    The pipeline encodes pixel-space source images into the latent space
    of a frozen VAE, runs the DDBM bridge diffusion process in that
    latent space, and decodes the result back to pixel space.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters
    ----------
    unet : DDBMUNet
        A DDBM UNet trained on VAE latents.
    scheduler : DDBMScheduler
        The DDBM noise scheduler.
    vae : AutoencoderKL
        A frozen pre-trained VAE for encoding/decoding.
    """

    model_cpu_offload_seq = "vae->unet"

    def __init__(
        self,
        unet: DDBMUNet,
        scheduler: DDBMScheduler,
        vae: AutoencoderKL,
    ):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, vae=vae)

        # Store diffusion parameters for the denoiser
        self.sigma_data = scheduler.config.sigma_data
        self.sigma_max = scheduler.config.sigma_max
        self.sigma_min = scheduler.config.sigma_min
        self.beta_d = scheduler.config.beta_d
        self.beta_min = scheduler.config.beta_min
        self.pred_mode = scheduler.config.pred_mode
        self.cov_xy = 0.0  # Default covariance

        # VAE scale factor for image processing
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @property
    def device(self) -> torch.device:
        """Get the device of the pipeline."""
        return next(self.unet.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the pipeline."""
        return next(self.unet.parameters()).dtype

    # ------------------------------------------------------------------
    # DDBM helper methods
    # ------------------------------------------------------------------

    def _vp_logsnr(self, t):
        """Compute log SNR for VP schedule."""
        t = torch.as_tensor(t)
        return -torch.log((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1)

    def _vp_logs(self, t):
        """Compute log scale for VP schedule."""
        t = torch.as_tensor(t)
        return -0.25 * t ** 2 * self.beta_d - 0.5 * t * self.beta_min

    def _append_dims(self, x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(
                f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
            )
        return x[(...,) + (None,) * dims_to_append]

    def _get_bridge_scalings(self, sigma):
        """Get the bridge scalings (c_skip, c_out, c_in) for the denoiser."""
        sigma_data = self.sigma_data
        sigma_data_end = sigma_data
        cov_xy = self.cov_xy
        c = 1

        if self.pred_mode == 've':
            A = (sigma**4 / self.sigma_max**4 * sigma_data_end**2 + 
                 (1 - sigma**2 / self.sigma_max**2)**2 * sigma_data**2 + 
                 2 * sigma**2 / self.sigma_max**2 * (1 - sigma**2 / self.sigma_max**2) * cov_xy + 
                 c**2 * sigma**2 * (1 - sigma**2 / self.sigma_max**2))
            c_in = 1 / A ** 0.5
            c_skip = ((1 - sigma**2 / self.sigma_max**2) * sigma_data**2 + 
                      sigma**2 / self.sigma_max**2 * cov_xy) / A
            c_out = ((sigma / self.sigma_max)**4 * (sigma_data_end**2 * sigma_data**2 - cov_xy**2) + 
                     sigma_data**2 * c**2 * sigma**2 * (1 - sigma**2 / self.sigma_max**2))**0.5 * c_in
            return c_skip, c_out, c_in

        elif self.pred_mode == 'vp':
            logsnr_t = self._vp_logsnr(sigma)
            logsnr_T = self._vp_logsnr(1)
            logs_t = self._vp_logs(sigma)
            logs_T = self._vp_logs(1)

            a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
            b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
            c_t = -torch.expm1(logsnr_T - logsnr_t) * (2 * logs_t - logsnr_t).exp()

            A = a_t**2 * sigma_data_end**2 + b_t**2 * sigma_data**2 + 2 * a_t * b_t * cov_xy + c**2 * c_t

            c_in = 1 / A ** 0.5
            c_skip = (b_t * sigma_data**2 + a_t * cov_xy) / A
            c_out = (a_t**2 * (sigma_data_end**2 * sigma_data**2 - cov_xy**2) + 
                     sigma_data**2 * c**2 * c_t)**0.5 * c_in
            return c_skip, c_out, c_in

        elif self.pred_mode in ['ve_simple', 'vp_simple']:
            c_in = torch.ones_like(sigma)
            c_out = torch.ones_like(sigma)
            c_skip = torch.zeros_like(sigma)
            return c_skip, c_out, c_in
        
        raise ValueError(f"Unknown pred_mode: {self.pred_mode}")

    def denoise(self, x_t, sigmas, x_T, clip_denoised=True):
        """
        Denoise the sample using the UNet model.
        
        Args:
            x_t: Noisy samples.
            sigmas: Sigma values for each sample.
            x_T: Target/condition images.
            clip_denoised: Whether to clip the output to [-1, 1].
        """
        model_device = self.device
        model_dtype = self.dtype

        c_skip, c_out, c_in = [
            self._append_dims(x, x_t.ndim) for x in self._get_bridge_scalings(sigmas)
        ]

        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)
        model_output = self.unet(
            (c_in * x_t).to(device=model_device, dtype=model_dtype),
            rescaled_t.to(device=model_device, dtype=model_dtype),
            xT=x_T.to(device=model_device, dtype=model_dtype),
        )
        model_output = model_output.to(device=x_t.device, dtype=x_t.dtype)
        denoised = c_out * model_output + c_skip * x_t

        if clip_denoised:
            denoised = denoised.clamp(-1, 1)

        return denoised

    def denoise_tiled(
        self,
        x_t: torch.Tensor,
        sigmas: torch.Tensor,
        x_T: torch.Tensor,
        views: List[Tuple[int, int, int, int]],
        view_batch_size: int = 1,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """MultiDiffusion: denoise on overlapping crops and merge by averaging."""
        value = torch.zeros_like(x_t)
        count = torch.zeros_like(x_t)
        batch_size = x_t.shape[0]
        view_batches = [
            views[i : i + view_batch_size]
            for i in range(0, len(views), view_batch_size)
        ]
        for batch_view in view_batches:
            vb_size = len(batch_view)
            crops_x = torch.cat(
                [
                    x_t[:, :, h_start:h_end, w_start:w_end]
                    for h_start, h_end, w_start, w_end in batch_view
                ],
                dim=0,
            )
            crops_x_T = torch.cat(
                [
                    x_T[:, :, h_start:h_end, w_start:w_end]
                    for h_start, h_end, w_start, w_end in batch_view
                ],
                dim=0,
            )
            sigmas_crops = sigmas.repeat_interleave(vb_size, dim=0)
            denoised_crops = self.denoise(crops_x, sigmas_crops, crops_x_T, clip_denoised=clip_denoised)
            for b in range(batch_size):
                for k, (h_start, h_end, w_start, w_end) in enumerate(batch_view):
                    idx = b * vb_size + k
                    value[b : b + 1, :, h_start:h_end, w_start:w_end] += denoised_crops[idx : idx + 1]
                    count[b : b + 1, :, h_start:h_end, w_start:w_end] += 1
        return torch.where(count > 0, value / count, value)

    def _get_d_stochastic(self, x, sigma, denoised, x_T, guidance):
        """Get stochastic derivative for churn step."""
        if self.pred_mode == 've':
            return self._get_d_ve(x, sigma, denoised, x_T, guidance, stochastic=True)
        elif self.pred_mode.startswith('vp'):
            return self._get_d_vp(x, sigma, denoised, x_T, guidance, stochastic=True)
        raise ValueError(f"Unknown pred_mode: {self.pred_mode}")

    def _get_d(self, x, sigma, denoised, x_T, guidance):
        """Get derivative for deterministic step."""
        if self.pred_mode == 've':
            return self._get_d_ve(x, sigma, denoised, x_T, guidance, stochastic=False)
        elif self.pred_mode.startswith('vp'):
            return self._get_d_vp(x, sigma, denoised, x_T, guidance, stochastic=False)
        raise ValueError(f"Unknown pred_mode: {self.pred_mode}")

    def _get_d_ve(self, x, sigma, denoised, x_T, w, stochastic=False):
        """Get derivative for VE mode."""
        grad_pxtlx0 = (denoised - x) / self._append_dims(sigma**2, x.ndim)
        grad_pxTlxt = (x_T - x) / (
            self._append_dims(torch.ones_like(sigma) * self.sigma_max**2, x.ndim) - 
            self._append_dims(sigma**2, x.ndim)
        )
        gt2 = 2 * sigma
        d = -(0.5 if not stochastic else 1) * gt2 * (
            grad_pxtlx0 - w * grad_pxTlxt * (0 if stochastic else 1)
        )
        if stochastic:
            return d, self._append_dims(gt2, x.ndim)
        return d

    def _get_d_vp(self, x, sigma, denoised, x_T, w, stochastic=False):
        """Get derivative for VP mode."""
        # Use scheduler's VP helper functions
        vp_snr_sqrt_reciprocal = self.scheduler._vp_snr_sqrt_reciprocal
        vp_snr_sqrt_reciprocal_deriv = self.scheduler._vp_snr_sqrt_reciprocal_deriv
        s_deriv = self.scheduler._s_deriv
        logs = self.scheduler._logs
        std = self.scheduler._std
        logsnr = self.scheduler._logsnr
        
        logsnr_T = logsnr(torch.as_tensor(self.sigma_max))
        logs_T = logs(torch.as_tensor(self.sigma_max))
        
        std_t = std(sigma)
        logsnr_t = logsnr(sigma)
        logs_t = logs(sigma)
        s_t_deriv = s_deriv(sigma)
        sigma_t = vp_snr_sqrt_reciprocal(sigma)
        sigma_t_deriv = vp_snr_sqrt_reciprocal_deriv(sigma)
        
        a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
        b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
        
        mu_t = a_t * x_T + b_t * denoised
        
        grad_logq = -(x - mu_t) / std_t**2 / (-torch.expm1(logsnr_T - logsnr_t))
        grad_logpxTlxt = -(x - torch.exp(logs_t - logs_T) * x_T) / std_t**2 / torch.expm1(logsnr_t - logsnr_T)
        
        f = s_t_deriv * (-logs_t).exp() * x
        gt2 = 2 * logs_t.exp()**2 * sigma_t * sigma_t_deriv
        
        d = f - gt2 * ((0.5 if not stochastic else 1) * grad_logq - w * grad_logpxTlxt)
        
        if stochastic:
            return d, self._append_dims(gt2, x.ndim)
        return d

    # ------------------------------------------------------------------
    # Image preprocessing and postprocessing
    # ------------------------------------------------------------------

    def prepare_inputs(
        self,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Prepare input images for the pipeline.
        
        Converts PIL images or numpy arrays to normalized tensors in [-1, 1] range.
        """
        if isinstance(image, Image.Image):
            image = [image]
        
        if isinstance(image, list) and isinstance(image[0], Image.Image):
            # Convert PIL images to tensor
            images = []
            for img in image:
                img = img.convert("RGB")
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                images.append(img_tensor)
            image = torch.stack(images)
        
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        
        # Ensure image is in [-1, 1] range
        if image.max() > 1.0:
            image = image / 255.0
        
        if image.min() >= 0:
            image = image * 2 - 1  # Convert [0, 1] to [-1, 1]
        
        return image.to(device=device, dtype=dtype)

    def _resize_to_output_size(
        self,
        image: torch.Tensor,
        output_size: Optional[Tuple[int, int]],
    ) -> torch.Tensor:
        """Resize image for MultiDiffusion upsampling (e.g. 512→1024)."""
        if output_size is None:
            return image
        h, w = output_size
        if image.shape[-2] == h and image.shape[-1] == w:
            return image
        return torch.nn.functional.interpolate(
            image, size=(h, w), mode="bilinear", align_corners=False
        )

    def _convert_to_pil(self, images: torch.Tensor) -> List[Image.Image]:
        """Convert tensor to PIL images."""
        images = (images + 1) / 2  # [-1, 1] -> [0, 1]
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        return [Image.fromarray(img) for img in images]

    def _convert_to_numpy(self, images: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        images = (images + 1) / 2  # [-1, 1] -> [0, 1]
        images = images.clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        return images

    # ------------------------------------------------------------------
    # VAE helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adapt_channels(images: torch.Tensor) -> torch.Tensor:
        """Repeat single-channel images to 3 channels for the VAE."""
        if images.shape[1] == 1:
            return images.repeat(1, 3, 1, 1)
        return images

    @staticmethod
    def _restore_channels(images: torch.Tensor, target_channels: int) -> torch.Tensor:
        """Average 3-channel VAE output back to 1 channel if needed."""
        if target_channels == 1 and images.shape[1] == 3:
            return images.mean(dim=1, keepdim=True)
        return images

    @torch.no_grad()
    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to VAE latent space."""
        adapted = self._adapt_channels(images)
        posterior = self.vae.encode(adapted).latent_dist
        return posterior.mean * self.vae.config.scaling_factor

    @torch.no_grad()
    def _decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents to pixel space."""
        scaled = latents / self.vae.config.scaling_factor
        return self.vae.decode(scaled).sample

    # ------------------------------------------------------------------
    # Main call method
    # ------------------------------------------------------------------

    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        num_inference_steps: int = 40,
        guidance: float = 1.0,
        churn_step_ratio: float = 0.33,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        target_channels: Optional[int] = None,
        output_size: Optional[Tuple[int, int]] = None,
        view_batch_size: int = 1,
        latent_window_size: int = DEFAULT_LATENT_WINDOW_SIZE,
    ):
        """
        Translate a source image via DDBM in VAE latent space.
        When latent grid exceeds trained size (e.g. 64 for 512px), MultiDiffusion
        tiling is used automatically. Use output_size=(1024, 1024) to upsample.

        Args:
            source_image: The source/condition image(s) for the bridge.
                Can be a tensor of shape (B, C, H, W) in [-1, 1] range,
                or PIL Image(s).
            num_inference_steps: Number of diffusion steps (default: 40).
            guidance: Guidance weight for sampling (default: 1.0).
            churn_step_ratio: Ratio of stochastic churn steps (default: 0.33).
            generator: Random number generator for reproducibility.
            output_type: Output format - "pil", "np", or "pt" (default: "pil").
            return_dict: Whether to return a dict with the output (default: True).
            callback: Callback function for progress updates.
            callback_steps: Frequency of callback calls.
            target_channels: Number of channels for the output image. If not 
                provided, defaults to the number of channels in source_image.

        Returns:
            Images generated through the bridge diffusion process in latent space.
        """
        device = self.device
        dtype = self.dtype

        # Prepare pixel inputs
        x_pixel = self.prepare_inputs(source_image, device, dtype)
        x_pixel = self._resize_to_output_size(x_pixel, output_size)
        orig_channels = x_pixel.shape[1]

        # Encode to latent space
        z_T = self._encode(x_pixel)
        batch_size = z_T.shape[0]
        _, _, lh, lw = z_T.shape
        views = None
        # Only use MultiDiffusion tiling when output_size is explicitly set; otherwise run full-res
        if output_size is not None:
            views = get_views(lh, lw, window_size=latent_window_size)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        sigmas = self.scheduler.sigmas

        # Start from z_T (encoded source image)
        z = z_T.clone()
        s_in = z.new_ones([batch_size])

        # Main sampling loop
        nfe = 0
        progress_bar = tqdm(range(len(sigmas) - 1), desc="DDBM Latent Sampling")

        for i in progress_bar:
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            # Stochastic churn step
            if churn_step_ratio > 0 and sigma_next != 0:
                sigma_hat = (sigma_next - sigma) * churn_step_ratio + sigma

                # Denoise at current sigma
                if views is not None:
                    denoised = self.denoise_tiled(z, sigma * s_in, z_T, views, view_batch_size=view_batch_size, clip_denoised=False)
                else:
                    denoised = self.denoise(z, sigma * s_in, z_T, clip_denoised=False)
                nfe += 1

                # Get stochastic derivative
                d_1, gt2 = self._get_d_stochastic(z, sigma, denoised, z_T, guidance)
                dt = sigma_hat - sigma
                noise = randn_tensor(z.shape, generator=generator, device=device, dtype=dtype)
                z = z + d_1 * dt + noise * (dt.abs() ** 0.5) * gt2.sqrt()
            else:
                sigma_hat = sigma

            # Denoise at sigma_hat
            if views is not None:
                denoised = self.denoise_tiled(z, sigma_hat * s_in, z_T, views, view_batch_size=view_batch_size, clip_denoised=False)
            else:
                denoised = self.denoise(z, sigma_hat * s_in, z_T, clip_denoised=False)
            nfe += 1

            # Get derivative
            d = self._get_d(z, sigma_hat, denoised, z_T, guidance)
            dt = sigma_next - sigma_hat

            if sigma_next == 0:
                # Final step
                z = z + d * dt
            else:
                # Heun's method
                z_2 = z + d * dt
                if views is not None:
                    denoised_2 = self.denoise_tiled(z_2, sigma_next * s_in, z_T, views, view_batch_size=view_batch_size, clip_denoised=False)
                else:
                    denoised_2 = self.denoise(z_2, sigma_next * s_in, z_T, clip_denoised=False)
                nfe += 1
                d_2 = self._get_d(z_2, sigma_next, denoised_2, z_T, guidance)
                d_prime = (d + d_2) / 2
                z = z + d_prime * dt

            # Callback
            if callback is not None and i % callback_steps == 0:
                callback(i, num_inference_steps, z)

        # Decode from latent to pixel space
        images = self._decode(z)
        images = self._restore_channels(images, target_channels or orig_channels)
        images = images.clamp(-1, 1)

        # Post-process output
        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)
        # else: output_type == "pt", return tensor as-is

        if not return_dict:
            return (images, nfe)

        return DDBMLatentPipelineOutput(images=images, nfe=nfe)
