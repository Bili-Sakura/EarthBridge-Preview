# Copyright (c) 2026 EarthBridge Team.
# Credits: See upstream/paper attribution below and README.md citations.

# Copyright 2024 The DBIM Authors and The Hugging Face Team.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Pipeline for Diffusion Bridge Implicit Models (DBIM) compatible with
# the Hugging Face diffusers library.

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

from src.utils.multidiffusion import (
    DEFAULT_LATENT_STRIDE,
    DEFAULT_LATENT_WINDOW_SIZE,
    DEFAULT_PIXEL_STRIDE,
    DEFAULT_PIXEL_WINDOW_SIZE,
    get_views as _get_views_impl,
)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from src.models.unet_dbim import DBIMUNet
from src.schedulers.scheduling_dbim import DBIMScheduler


@dataclass
class DBIMPipelineOutput(BaseOutput):
    """Output class for DBIM pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray` or `torch.Tensor`):
            Generated images.
        nfe (`int`):
            Number of function evaluations (UNet forward passes).
        sampler (`str`):
            Sampler used (`"dbim"`, `"dbim_high_order"`, `"heun"`).
    """

    images: Union[List[Image.Image], np.ndarray, torch.Tensor]
    nfe: int = 0
    sampler: str = "dbim"


class DBIMSamplingMixin:
    """Shared DBIM sampling logic for pixel and latent pipelines."""

    def _get_device(self) -> torch.device:
        """Get execution device; works with DataParallel-wrapped unet."""
        unet = self.unet.module if hasattr(self.unet, "module") else self.unet
        return next(unet.parameters()).device

    def _get_dtype(self) -> torch.dtype:
        """Get execution dtype; works with DataParallel-wrapped unet."""
        unet = self.unet.module if hasattr(self.unet, "module") else self.unet
        return next(unet.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return self._get_device()

    @property
    def dtype(self) -> torch.dtype:
        return self._get_dtype()

    @staticmethod
    def _append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(
                f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
            )
        return x[(...,) + (None,) * dims_to_append]

    @staticmethod
    def _safe_div(
        numer: torch.Tensor,
        denom: torch.Tensor,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        sign = torch.where(denom >= 0, torch.ones_like(denom), -torch.ones_like(denom))
        safe = torch.where(denom.abs() < eps, sign * eps, denom)
        return numer / safe

    @staticmethod
    def _safe_log_ratio(
        numer: torch.Tensor,
        denom: torch.Tensor,
        eps: float = 1e-20,
    ) -> torch.Tensor:
        return torch.log(torch.clamp(numer, min=eps) / torch.clamp(denom, min=eps))

    def _bridge_scalings(self, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Return DBIM bridge preconditioning terms `(c_skip, c_out, c_in, c_noise)`."""
        a_t, b_t, c_t = self.scheduler.get_abc(t)
        sigma_data = self.scheduler.config.sigma_data
        sigma_data_end = sigma_data
        cov_xy = 0.0

        A = (
            a_t**2 * sigma_data_end**2
            + b_t**2 * sigma_data**2
            + 2 * a_t * b_t * cov_xy
            + c_t**2
        )
        c_in = torch.rsqrt(torch.clamp(A, min=1e-20))
        c_skip = (b_t * sigma_data**2 + a_t * cov_xy) / torch.clamp(A, min=1e-20)
        c_out = (
            torch.sqrt(
                torch.clamp(
                    a_t**2 * (sigma_data_end**2 * sigma_data**2 - cov_xy**2)
                    + sigma_data**2 * c_t**2,
                    min=1e-20,
                )
            )
            * c_in
        )
        c_noise = 1000.0 * 0.25 * torch.log(torch.clamp(t, min=1e-44))
        return c_skip, c_out, c_in, c_noise

    def denoise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_T: torch.Tensor,
        clip_denoised: bool = True,
        cfg_scale: float = 1.0,
        null_condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict denoised bridge state from noisy sample at time `t`."""
        model_device = self._get_device()
        model_dtype = self._get_dtype()

        c_skip, c_out, c_in, c_noise = self._bridge_scalings(t)
        c_skip = self._append_dims(c_skip, x_t.ndim).to(dtype=x_t.dtype, device=x_t.device)
        c_out = self._append_dims(c_out, x_t.ndim).to(dtype=x_t.dtype, device=x_t.device)
        c_in = self._append_dims(c_in, x_t.ndim).to(dtype=x_t.dtype, device=x_t.device)

        use_cfg = abs(float(cfg_scale) - 1.0) > 1e-6
        if use_cfg:
            if null_condition is None:
                null_condition = torch.zeros_like(x_T)
            model_input = torch.cat([c_in * x_t, c_in * x_t], dim=0).to(
                dtype=model_dtype, device=model_device
            )
            timestep_input = torch.cat(
                [
                    c_noise.to(dtype=model_dtype, device=model_device),
                    c_noise.to(dtype=model_dtype, device=model_device),
                ],
                dim=0,
            )
            cond_input = torch.cat([x_T, null_condition], dim=0).to(
                dtype=model_dtype, device=model_device
            )
            model_output_batched = self.unet(model_input, timestep_input, xT=cond_input)
            model_output_cond, model_output_uncond = model_output_batched.chunk(2, dim=0)
            model_output = model_output_uncond + cfg_scale * (model_output_cond - model_output_uncond)
        else:
            model_output = self.unet(
                (c_in * x_t).to(dtype=model_dtype, device=model_device),
                c_noise.to(dtype=model_dtype, device=model_device),
                xT=x_T.to(dtype=model_dtype, device=model_device),
            )
        model_output = model_output.to(dtype=x_t.dtype, device=x_t.device)
        denoised = c_out * model_output + c_skip * x_t
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    @staticmethod
    def get_views(
        latent_height: int,
        latent_width: int,
        window_size: int = DEFAULT_LATENT_WINDOW_SIZE,
        stride: int = DEFAULT_LATENT_STRIDE,
    ) -> List[Tuple[int, int, int, int]]:
        """MultiDiffusion view layout; delegates to shared util."""
        return _get_views_impl(latent_height, latent_width, window_size, stride)

    def denoise_tiled(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_T: torch.Tensor,
        views: List[Tuple[int, int, int, int]],
        view_batch_size: int = 1,
        clip_denoised: bool = True,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Predict denoised bridge state using MultiDiffusion: run UNet on overlapping
        crops (views) and merge by averaging. Use when latent size > trained size
        (e.g. 1024px → 128 latent vs 512px → 64).
        """
        value = torch.zeros_like(x_t)
        count = torch.zeros_like(x_t)
        view_batches = [
            views[i : i + view_batch_size]
            for i in range(0, len(views), view_batch_size)
        ]
        batch_size = x_t.shape[0]
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
            # t shape (batch_size,); repeat per view so each crop gets correct timestep
            t_crops = t.repeat_interleave(vb_size, dim=0).to(crops_x.dtype)
            denoised_crops = self.denoise(
                crops_x,
                t_crops,
                crops_x_T,
                clip_denoised=clip_denoised,
                cfg_scale=cfg_scale,
            )
            for b in range(batch_size):
                for k, (h_start, h_end, w_start, w_end) in enumerate(batch_view):
                    idx = b * vb_size + k
                    value[b : b + 1, :, h_start:h_end, w_start:w_end] += denoised_crops[idx : idx + 1]
                    count[b : b + 1, :, h_start:h_end, w_start:w_end] += 1
        denoised = torch.where(count > 0, value / count, value)
        return denoised

    def _get_d(
        self,
        x: torch.Tensor,
        x_T: torch.Tensor,
        t: torch.Tensor,
        stochastic: bool,
        guidance: float,
        cfg_scale: float,
        clip_denoised: bool,
        views: Optional[List[Tuple[int, int, int, int]]] = None,
        view_batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute DBIM/DDBM drift term and diffusion coefficient."""
        ones = x.new_ones([x.shape[0]])
        t_batch = t * ones

        f_t, g2_t = self.scheduler.get_f_g2(t_batch)
        alpha_t, alpha_bar_t, _rho_t, rho_bar_t = self.scheduler.get_alpha_rho(t_batch)
        a_t, b_t, c_t = self.scheduler.get_abc(t_batch)

        f_t = self._append_dims(f_t, x.ndim).to(dtype=x.dtype, device=x.device)
        g2_t = self._append_dims(g2_t, x.ndim).to(dtype=x.dtype, device=x.device)
        alpha_t = self._append_dims(alpha_t, x.ndim).to(dtype=x.dtype, device=x.device)
        alpha_bar_t = self._append_dims(alpha_bar_t, x.ndim).to(dtype=x.dtype, device=x.device)
        rho_bar_t = self._append_dims(rho_bar_t, x.ndim).to(dtype=x.dtype, device=x.device)
        a_t = self._append_dims(a_t, x.ndim).to(dtype=x.dtype, device=x.device)
        b_t = self._append_dims(b_t, x.ndim).to(dtype=x.dtype, device=x.device)
        c_t = self._append_dims(c_t, x.ndim).to(dtype=x.dtype, device=x.device)

        if views is not None:
            denoised = self.denoise_tiled(
                x,
                t_batch,
                x_T,
                views,
                view_batch_size=view_batch_size,
                clip_denoised=clip_denoised,
                cfg_scale=cfg_scale,
            )
        else:
            denoised = self.denoise(
                x,
                t_batch,
                x_T,
                clip_denoised=clip_denoised,
                cfg_scale=cfg_scale,
            )

        grad_logq = -self._safe_div(x - (a_t * x_T + b_t * denoised), c_t**2)
        grad_logpxTlxt = -self._safe_div(
            x - alpha_bar_t * x_T,
            alpha_t**2 * rho_bar_t**2,
        )

        prefactor = 0.5 if not stochastic else 1.0
        d = f_t * x - g2_t * (prefactor * grad_logq - guidance * grad_logpxTlxt)
        return d, g2_t, denoised

    def _ddbm_simulate(
        self,
        x: torch.Tensor,
        x_T: torch.Tensor,
        t_cur: torch.Tensor,
        t_next: torch.Tensor,
        stochastic: bool,
        second_order: bool,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        guidance: float,
        cfg_scale: float,
        clip_denoised: bool,
        views: Optional[List[Tuple[int, int, int, int]]] = None,
        view_batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate one DDBM/DBIM sub-step."""
        dt = t_next - t_cur
        d, g2_t, pred_x0 = self._get_d(
            x=x,
            x_T=x_T,
            t=t_cur,
            stochastic=stochastic,
            guidance=guidance,
            cfg_scale=cfg_scale,
            clip_denoised=clip_denoised,
            views=views,
            view_batch_size=view_batch_size,
        )

        if stochastic:
            noise = randn_tensor(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            x_new = x + d * dt + noise * torch.sqrt(dt.abs()) * torch.sqrt(
                torch.clamp(g2_t, min=0.0)
            )
        else:
            x_new = x + d * dt

        if second_order:
            d_2, _g2_t_2, pred_x0 = self._get_d(
                x=x_new,
                x_T=x_T,
                t=t_next,
                stochastic=stochastic,
                guidance=guidance,
                cfg_scale=cfg_scale,
                clip_denoised=clip_denoised,
                views=views,
                view_batch_size=view_batch_size,
            )
            d_prime = (d + d_2) / 2
            if stochastic:
                noise = randn_tensor(x.shape, generator=generator, device=x.device, dtype=x.dtype)
                x_new = x + d_prime * dt + noise * torch.sqrt(dt.abs()) * torch.sqrt(
                    torch.clamp(g2_t, min=0.0)
                )
            else:
                x_new = x + d_prime * dt

        return x_new, pred_x0

    def _sample_heun(
        self,
        x_T: torch.Tensor,
        num_inference_steps: int,
        guidance: float,
        cfg_scale: float,
        churn_step_ratio: float,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        callback: Optional[Callable[[int, int, torch.Tensor], None]],
        callback_steps: int,
        clip_denoised: bool,
        views: Optional[List[Tuple[int, int, int, int]]] = None,
        view_batch_size: int = 1,
    ) -> Tuple[torch.Tensor, int]:
        self.scheduler.set_timesteps(num_inference_steps, device=x_T.device, sampler="heun")
        ts = self.scheduler.sigmas

        x = x_T
        nfe = 0
        nfe_per_denoise = 2 if abs(float(cfg_scale) - 1.0) > 1e-6 else 1
        sim_kw = dict(
            views=views,
            view_batch_size=view_batch_size,
        )

        for i in tqdm(range(len(ts) - 1), desc="DBIM Heun Sampling"):
            if churn_step_ratio > 0:
                t_hat = (ts[i + 1] - ts[i]) * churn_step_ratio + ts[i]
                x, _ = self._ddbm_simulate(
                    x=x,
                    x_T=x_T,
                    t_cur=ts[i],
                    t_next=t_hat,
                    stochastic=True,
                    second_order=False,
                    generator=generator,
                    guidance=guidance,
                    cfg_scale=cfg_scale,
                    clip_denoised=clip_denoised,
                    **sim_kw,
                )
                nfe += nfe_per_denoise
            else:
                t_hat = ts[i]

            if ts[i + 1] == 0:
                x, _ = self._ddbm_simulate(
                    x=x,
                    x_T=x_T,
                    t_cur=t_hat,
                    t_next=ts[i + 1],
                    stochastic=False,
                    second_order=False,
                    generator=generator,
                    guidance=guidance,
                    cfg_scale=cfg_scale,
                    clip_denoised=clip_denoised,
                    **sim_kw,
                )
                nfe += nfe_per_denoise
            else:
                x, _ = self._ddbm_simulate(
                    x=x,
                    x_T=x_T,
                    t_cur=t_hat,
                    t_next=ts[i + 1],
                    stochastic=False,
                    second_order=True,
                    generator=generator,
                    guidance=guidance,
                    cfg_scale=cfg_scale,
                    clip_denoised=clip_denoised,
                    **sim_kw,
                )
                nfe += 2 * nfe_per_denoise

            if callback is not None and i % callback_steps == 0:
                callback(i, num_inference_steps, x)

        return x, nfe

    def _sample_dbim(
        self,
        x_T: torch.Tensor,
        num_inference_steps: int,
        cfg_scale: float,
        eta: float,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        callback: Optional[Callable[[int, int, torch.Tensor], None]],
        callback_steps: int,
        clip_denoised: bool,
        views: Optional[List[Tuple[int, int, int, int]]] = None,
        view_batch_size: int = 1,
    ) -> Tuple[torch.Tensor, int]:
        self.scheduler.set_timesteps(num_inference_steps, device=x_T.device, sampler="dbim")
        ts = self.scheduler.sigmas

        x = x_T
        ones = x.new_ones([x.shape[0]])
        nfe = 0
        nfe_per_denoise = 2 if abs(float(cfg_scale) - 1.0) > 1e-6 else 1
        t_max = torch.as_tensor(self.scheduler.config.sigma_max, device=x.device, dtype=x.dtype) * ones

        if views is not None:
            x0_hat = self.denoise_tiled(
                x,
                t_max,
                x_T,
                views,
                view_batch_size=view_batch_size,
                clip_denoised=clip_denoised,
                cfg_scale=cfg_scale,
            )
        else:
            x0_hat = self.denoise(x, t_max, x_T, clip_denoised=clip_denoised, cfg_scale=cfg_scale)
        nfe += nfe_per_denoise

        noise = randn_tensor(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        x = self.scheduler.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)

        for i in tqdm(range(len(ts) - 1), desc="DBIM Sampling"):
            s = ts[i]
            t = ts[i + 1]

            if views is not None:
                x0_hat = self.denoise_tiled(
                    x,
                    s * ones,
                    x_T,
                    views,
                    view_batch_size=view_batch_size,
                    clip_denoised=clip_denoised,
                    cfg_scale=cfg_scale,
                )
            else:
                x0_hat = self.denoise(x, s * ones, x_T, clip_denoised=clip_denoised, cfg_scale=cfg_scale)
            nfe += nfe_per_denoise

            a_s, b_s, c_s = [
                self._append_dims(item, x.ndim).to(dtype=x.dtype, device=x.device)
                for item in self.scheduler.get_abc(s * ones)
            ]
            a_t, b_t, c_t = [
                self._append_dims(item, x.ndim).to(dtype=x.dtype, device=x.device)
                for item in self.scheduler.get_abc(t * ones)
            ]
            _alpha_s, _alpha_bar_s, rho_s, _rho_bar_s = [
                self._append_dims(item, x.ndim).to(dtype=x.dtype, device=x.device)
                for item in self.scheduler.get_alpha_rho(s * ones)
            ]
            alpha_t, _alpha_bar_t, rho_t, _rho_bar_t = [
                self._append_dims(item, x.ndim).to(dtype=x.dtype, device=x.device)
                for item in self.scheduler.get_alpha_rho(t * ones)
            ]

            ratio = torch.clamp(1.0 - (rho_t**2 / torch.clamp(rho_s**2, min=1e-20)), min=0.0)
            omega_st = eta * (alpha_t * rho_t) * torch.sqrt(ratio)
            tmp_var = torch.sqrt(torch.clamp(c_t**2 - omega_st**2, min=0.0)) / torch.clamp(
                c_s, min=1e-20
            )

            coeff_xs = tmp_var
            coeff_x0_hat = b_t - tmp_var * b_s
            coeff_xT = a_t - tmp_var * a_s

            noise = randn_tensor(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            if i == len(ts) - 2:
                x = coeff_x0_hat * x0_hat + coeff_xT * x_T + coeff_xs * x
            else:
                x = coeff_x0_hat * x0_hat + coeff_xT * x_T + coeff_xs * x + omega_st * noise

            if callback is not None and i % callback_steps == 0:
                callback(i, num_inference_steps, x)

        return x, nfe

    def _sample_dbim_high_order(
        self,
        x_T: torch.Tensor,
        num_inference_steps: int,
        cfg_scale: float,
        order: int,
        lower_order_final: bool,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        callback: Optional[Callable[[int, int, torch.Tensor], None]],
        callback_steps: int,
        clip_denoised: bool,
        views: Optional[List[Tuple[int, int, int, int]]] = None,
        view_batch_size: int = 1,
    ) -> Tuple[torch.Tensor, int]:
        if order not in (2, 3):
            raise ValueError("DBIM high-order sampler currently supports order in {2, 3}.")

        self.scheduler.set_timesteps(
            num_inference_steps,
            device=x_T.device,
            sampler="dbim_high_order",
        )
        ts = self.scheduler.sigmas

        x = x_T
        ones = x.new_ones([x.shape[0]])
        nfe = 0
        nfe_per_denoise = 2 if abs(float(cfg_scale) - 1.0) > 1e-6 else 1
        t_max = torch.as_tensor(self.scheduler.config.sigma_max, device=x.device, dtype=x.dtype) * ones

        if views is not None:
            x0_hat = self.denoise_tiled(
                x,
                t_max,
                x_T,
                views,
                view_batch_size=view_batch_size,
                clip_denoised=clip_denoised,
                cfg_scale=cfg_scale,
            )
        else:
            x0_hat = self.denoise(x, t_max, x_T, clip_denoised=clip_denoised, cfg_scale=cfg_scale)
        nfe += nfe_per_denoise
        noise = randn_tensor(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        x = self.scheduler.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)

        u0 = torch.as_tensor(self.scheduler.config.sigma_max, device=x.device, dtype=x.dtype)
        if float(u0) == 1.0:
            u0 = u0 - 5e-5
        u_hist: List[torch.Tensor] = [u0.clone() for _ in range(order - 1)]
        x0_hist: List[torch.Tensor] = [x0_hat.detach().clone() for _ in range(order - 1)]

        for i in tqdm(range(len(ts) - 1), desc=f"DBIM High-Order (order={order})"):
            s = ts[i]
            t = ts[i + 1]

            if (lower_order_final and i + 1 == len(ts) - 1) or (i == 0):
                a_s, b_s, c_s = [
                    self._append_dims(item, x.ndim).to(dtype=x.dtype, device=x.device)
                    for item in self.scheduler.get_abc(s * ones)
                ]
                a_t, b_t, c_t = [
                    self._append_dims(item, x.ndim).to(dtype=x.dtype, device=x.device)
                    for item in self.scheduler.get_abc(t * ones)
                ]

                tmp_var = self._safe_div(c_t, c_s)
                coeff_xs = tmp_var
                coeff_x0_hat = b_t - tmp_var * b_s
                coeff_xT = a_t - tmp_var * a_s

                if views is not None:
                    x0_hat = self.denoise_tiled(
                        x,
                        s * ones,
                        x_T,
                        views,
                        view_batch_size=view_batch_size,
                        clip_denoised=clip_denoised,
                        cfg_scale=cfg_scale,
                    )
                else:
                    x0_hat = self.denoise(x, s * ones, x_T, clip_denoised=clip_denoised, cfg_scale=cfg_scale)
                nfe += nfe_per_denoise
                x = coeff_xs * x + coeff_x0_hat * x0_hat + coeff_xT * x_T

            elif order == 2 or i == 1:
                a_u, b_u, c_u = [
                    self._append_dims(item, x.ndim).to(dtype=x.dtype, device=x.device)
                    for item in self.scheduler.get_abc(u_hist[-1] * ones)
                ]
                a_s, b_s, c_s = [
                    self._append_dims(item, x.ndim).to(dtype=x.dtype, device=x.device)
                    for item in self.scheduler.get_abc(s * ones)
                ]
                a_t, b_t, c_t = [
                    self._append_dims(item, x.ndim).to(dtype=x.dtype, device=x.device)
                    for item in self.scheduler.get_abc(t * ones)
                ]

                lambda_u = self._safe_log_ratio(b_u, c_u)
                lambda_s = self._safe_log_ratio(b_s, c_s)
                lambda_t = self._safe_log_ratio(b_t, c_t)

                if views is not None:
                    x0_hat = self.denoise_tiled(
                        x,
                        s * ones,
                        x_T,
                        views,
                        view_batch_size=view_batch_size,
                        clip_denoised=clip_denoised,
                        cfg_scale=cfg_scale,
                    )
                else:
                    x0_hat = self.denoise(x, s * ones, x_T, clip_denoised=clip_denoised, cfg_scale=cfg_scale)
                nfe += nfe_per_denoise

                h = lambda_t - lambda_s
                h2 = lambda_s - lambda_u
                term = self._safe_div(x0_hat - x0_hist[-1], h2)
                integral = torch.exp(lambda_t) * (
                    (1 - torch.exp(-h)) * x0_hat
                    + (torch.exp(-h) + h - 1) * term
                )
                x = x * self._safe_div(c_t, c_s) + x_T * (
                    a_t - a_s * self._safe_div(c_t, c_s)
                ) + c_t * integral

            else:
                a_u1, b_u1, c_u1 = [
                    self._append_dims(item, x.ndim).to(dtype=x.dtype, device=x.device)
                    for item in self.scheduler.get_abc(u_hist[-1] * ones)
                ]
                a_u2, b_u2, c_u2 = [
                    self._append_dims(item, x.ndim).to(dtype=x.dtype, device=x.device)
                    for item in self.scheduler.get_abc(u_hist[-2] * ones)
                ]
                a_s, b_s, c_s = [
                    self._append_dims(item, x.ndim).to(dtype=x.dtype, device=x.device)
                    for item in self.scheduler.get_abc(s * ones)
                ]
                a_t, b_t, c_t = [
                    self._append_dims(item, x.ndim).to(dtype=x.dtype, device=x.device)
                    for item in self.scheduler.get_abc(t * ones)
                ]

                lambda_u2 = self._safe_log_ratio(b_u2, c_u2)
                lambda_u1 = self._safe_log_ratio(b_u1, c_u1)
                lambda_s = self._safe_log_ratio(b_s, c_s)
                lambda_t = self._safe_log_ratio(b_t, c_t)

                if views is not None:
                    x0_hat = self.denoise_tiled(
                        x,
                        s * ones,
                        x_T,
                        views,
                        view_batch_size=view_batch_size,
                        clip_denoised=clip_denoised,
                        cfg_scale=cfg_scale,
                    )
                else:
                    x0_hat = self.denoise(x, s * ones, x_T, clip_denoised=clip_denoised, cfg_scale=cfg_scale)
                nfe += nfe_per_denoise

                h = lambda_t - lambda_s
                h1 = lambda_s - lambda_u1
                h2 = lambda_u1 - lambda_u2

                dx0_hat = self._safe_div(
                    self._safe_div(
                        (x0_hat - x0_hist[-1]) * (2 * h1 + h2),
                        h1,
                    )
                    - self._safe_div(
                        (x0_hist[-1] - x0_hist[-2]) * h1,
                        h2,
                    ),
                    h1 + h2,
                )
                d2x0_hat = 2 * self._safe_div(
                    self._safe_div(x0_hat - x0_hist[-1], h1)
                    - self._safe_div(x0_hist[-1] - x0_hist[-2], h2),
                    h1 + h2,
                )
                integral = torch.exp(lambda_t) * (
                    (1 - torch.exp(-h)) * x0_hat
                    + (torch.exp(-h) + h - 1) * dx0_hat
                    + (h**2 / 2 - h + 1 - torch.exp(-h)) * d2x0_hat
                )
                x = x * self._safe_div(c_t, c_s) + x_T * (
                    a_t - a_s * self._safe_div(c_t, c_s)
                ) + c_t * integral

            u_hist.append(s.detach().clone())
            u_hist.pop(0)
            x0_hist.append(x0_hat.detach().clone())
            x0_hist.pop(0)

            if callback is not None and i % callback_steps == 0:
                callback(i, num_inference_steps, x)

        return x, nfe

    def _run_sampler(
        self,
        x_T: torch.Tensor,
        num_inference_steps: int,
        sampler: str,
        guidance: float,
        cfg_scale: float,
        churn_step_ratio: float,
        eta: float,
        order: int,
        lower_order_final: bool,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        callback: Optional[Callable[[int, int, torch.Tensor], None]],
        callback_steps: int,
        clip_denoised: bool,
        views: Optional[List[Tuple[int, int, int, int]]] = None,
        view_batch_size: int = 1,
    ) -> Tuple[torch.Tensor, int]:
        samp_kw = dict(
            views=views,
            view_batch_size=view_batch_size,
        )
        if sampler == "heun":
            return self._sample_heun(
                x_T=x_T,
                num_inference_steps=num_inference_steps,
                guidance=guidance,
                cfg_scale=cfg_scale,
                churn_step_ratio=churn_step_ratio,
                generator=generator,
                callback=callback,
                callback_steps=callback_steps,
                clip_denoised=clip_denoised,
                **samp_kw,
            )
        if sampler == "dbim":
            return self._sample_dbim(
                x_T=x_T,
                num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale,
                eta=eta,
                generator=generator,
                callback=callback,
                callback_steps=callback_steps,
                clip_denoised=clip_denoised,
                **samp_kw,
            )
        if sampler == "dbim_high_order":
            return self._sample_dbim_high_order(
                x_T=x_T,
                num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale,
                order=order,
                lower_order_final=lower_order_final,
                generator=generator,
                callback=callback,
                callback_steps=callback_steps,
                clip_denoised=clip_denoised,
                **samp_kw,
            )
        raise ValueError(
            f"Unknown sampler '{sampler}'. Expected one of: heun, dbim, dbim_high_order."
        )

    @staticmethod
    def _convert_to_pil(images: torch.Tensor) -> List[Image.Image]:
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype(np.uint8)
        out = []
        for img in images:
            if img.shape[2] == 1:
                img = img.squeeze(2)
            out.append(Image.fromarray(img))
        return out

    @staticmethod
    def _convert_to_numpy(images: torch.Tensor) -> np.ndarray:
        images = (images + 1) / 2
        images = images.clamp(0, 1)
        return images.detach().cpu().permute(0, 2, 3, 1).numpy()


class DBIMPipeline(DiffusionPipeline, DBIMSamplingMixin):
    r"""Pixel-space DBIM image-to-image translation pipeline."""

    model_cpu_offload_seq = "unet"

    def __init__(
        self,
        unet: DBIMUNet,
        scheduler: DBIMScheduler,
    ):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

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

    @torch.no_grad()
    def __call__(
        self,
        source_image: Union[torch.Tensor, Image.Image, List[Image.Image]],
        num_inference_steps: int = 40,
        sampler: str = "dbim",
        guidance: float = 1.0,
        cfg_scale: float = 1.0,
        churn_step_ratio: float = 0.33,
        eta: Optional[float] = None,
        order: Optional[int] = None,
        lower_order_final: Optional[bool] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        clip_denoised: bool = True,
        output_size: Optional[Tuple[int, int]] = None,
        view_batch_size: int = 1,
        multidiffusion_window_size: Optional[int] = None,
        multidiffusion_stride: Optional[int] = None,
    ):
        """Run DBIM sampling from the provided source image(s).

        MultiDiffusion-style: set output_size=(1024, 1024) to resize source to 1024
        before sampling; tiled denoising runs on 512px windows. Use view_batch_size
        to batch views for speed (e.g. view_batch_size=4).
        multidiffusion_window_size / multidiffusion_stride override defaults (512 / 64) when set.
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

        x_T = self.prepare_inputs(source_image, self._get_device(), self._get_dtype())
        if output_size is not None:
            h_out, w_out = output_size
            x_T = F.interpolate(
                x_T, size=(h_out, w_out), mode="bilinear", align_corners=False
            )
        _, _, h, w = x_T.shape
        pixel_window = multidiffusion_window_size if multidiffusion_window_size is not None else DEFAULT_PIXEL_WINDOW_SIZE
        pixel_stride = multidiffusion_stride if multidiffusion_stride is not None else DEFAULT_PIXEL_STRIDE
        views = None
        # Only use MultiDiffusion tiling when output_size is explicitly set; otherwise run full-res
        if output_size is not None:
            views = self.get_views(
                h, w,
                window_size=pixel_window,
                stride=pixel_stride,
            )
        images, nfe = self._run_sampler(
            x_T=x_T,
            num_inference_steps=num_inference_steps,
            sampler=sampler,
            guidance=guidance,
            cfg_scale=cfg_scale,
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
        images = images.clamp(-1, 1)

        if output_type == "pil":
            images = self._convert_to_pil(images)
        elif output_type == "np":
            images = self._convert_to_numpy(images)

        if not return_dict:
            return images, nfe
        return DBIMPipelineOutput(images=images, nfe=nfe, sampler=sampler)
