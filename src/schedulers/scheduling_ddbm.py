# Copyright (c) 2026 EarthBridge Team.
# Credits: See upstream/paper attribution below and README.md citations.

# Copyright 2024 The DDBM Authors and The Hugging Face Team.
# Licensed under the Apache License, Version 2.0 (the "License");
#
# This scheduler implements the Denoising Diffusion Bridge Models (DDBM) sampling
# algorithm compatible with the Hugging Face diffusers library.
# Based on: https://arxiv.org/abs/2309.16948

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerMixin


@dataclass
class DDBMSchedulerOutput(BaseOutput):
    """
    Output class for the DDBM scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class DDBMScheduler(SchedulerMixin, ConfigMixin):
    """
    Scheduler for Denoising Diffusion Bridge Models (DDBM).

    This scheduler implements the Heun sampler from the DDBM paper for image-to-image translation tasks using
    diffusion bridges. Unlike standard diffusion models that map noise to data, bridge models learn to transform
    between two data distributions (source -> target).

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        sigma_min (`float`, defaults to `0.002`):
            Minimum sigma value for the noise schedule.
        sigma_max (`float`, defaults to `80.0`):
            Maximum sigma value (T in the diffusion bridge). For VP schedules with normalized time, use 1.0.
        sigma_data (`float`, defaults to `0.5`):
            Standard deviation of the data distribution.
        beta_d (`float`, defaults to `2.0`):
            Beta_d parameter for VP (variance-preserving) schedule.
        beta_min (`float`, defaults to `0.1`):
            Beta_min parameter for VP schedule.
        rho (`float`, defaults to `7.0`):
            Rho parameter for Karras noise schedule discretization.
        pred_mode (`str`, defaults to `"vp"`):
            Prediction mode for the diffusion process. Choose from:
            - `"ve"`: Variance-exploding schedule
            - `"vp"`: Variance-preserving schedule
            - `"ve_simple"`: Simplified VE schedule
            - `"vp_simple"`: Simplified VP schedule
        num_train_timesteps (`int`, defaults to `40`):
            Number of diffusion steps used during training.
    """

    _compatibles = []
    order = 2  # Heun is a 2nd order method

    @register_to_config
    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        beta_d: float = 2.0,
        beta_min: float = 0.1,
        rho: float = 7.0,
        pred_mode: str = "vp",
        num_train_timesteps: int = 40,
    ):
        # Store config values
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.rho = rho
        self.pred_mode = pred_mode
        self.num_train_timesteps = num_train_timesteps

        # Initialize state
        self.sigmas: Optional[torch.Tensor] = None
        self.timesteps: Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int] = None
        
        # Standard deviation of the initial noise distribution
        self.init_noise_sigma = sigma_max

        # Initialize VP helper functions
        self._init_vp_functions()
    
    def _init_vp_functions(self):
        """Initialize VP schedule helper functions."""
        beta_d = self.beta_d
        beta_min = self.beta_min
        
        def vp_snr_sqrt_reciprocal(t):
            """Compute SNR^(-0.5) for VP schedule."""
            t_tensor = torch.as_tensor(t)
            return (torch.exp(0.5 * beta_d * (t_tensor ** 2) + beta_min * t_tensor) - 1) ** 0.5
        
        def vp_snr_sqrt_reciprocal_deriv(t):
            """Derivative of SNR^(-0.5)."""
            snr_sqrt_recip = vp_snr_sqrt_reciprocal(t)
            return 0.5 * (beta_min + beta_d * t) * (snr_sqrt_recip + 1 / snr_sqrt_recip)
        
        def s(t):
            """Scale factor s(t)."""
            snr_sqrt_recip = vp_snr_sqrt_reciprocal(t)
            return (1 + snr_sqrt_recip ** 2).rsqrt()
        
        def s_deriv(t):
            """Derivative of s(t)."""
            return -vp_snr_sqrt_reciprocal(t) * vp_snr_sqrt_reciprocal_deriv(t) * (s(t) ** 3)
        
        def logs(t):
            """Log scale function."""
            t_tensor = torch.as_tensor(t)
            return -0.25 * t_tensor ** 2 * beta_d - 0.5 * t_tensor * beta_min
        
        def std(t):
            """Standard deviation function."""
            return vp_snr_sqrt_reciprocal(t) * s(t)
        
        def logsnr(t):
            """Log SNR function."""
            return -2 * torch.log(vp_snr_sqrt_reciprocal(t))
        
        self._vp_snr_sqrt_reciprocal = vp_snr_sqrt_reciprocal
        self._vp_snr_sqrt_reciprocal_deriv = vp_snr_sqrt_reciprocal_deriv
        self._s = s
        self._s_deriv = s_deriv
        self._logs = logs
        self._std = std
        self._logsnr = logsnr

    def _vp_logsnr(self, t):
        """Compute log SNR for VP schedule."""
        t = torch.as_tensor(t)
        return -torch.log((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1)

    def _vp_logs(self, t):
        """Compute log scale for VP schedule."""
        t = torch.as_tensor(t)
        return -0.25 * t ** 2 * self.beta_d - 0.5 * t * self.beta_min

    def set_timesteps(
        self, 
        num_inference_steps: int, 
        device: Union[str, torch.device] = None
    ):
        """
        Sets the discrete timesteps used for the diffusion chain.

        Args:
            num_inference_steps (`int`):
                Number of diffusion steps.
            device (`str` or `torch.device`, *optional*):
                Device to place tensors on.
        """
        self.num_inference_steps = num_inference_steps
        
        # Generate Karras sigmas schedule
        ramp = torch.linspace(0, 1, num_inference_steps)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = (self.sigma_max - 1e-4) ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        sigmas = torch.cat([sigmas, torch.zeros(1)])  # Append zero
        
        self.sigmas = sigmas.to(device)
        self.timesteps = torch.arange(num_inference_steps, device=device)
    
    def _get_d_ve(
        self, 
        x: torch.Tensor, 
        sigma: torch.Tensor, 
        denoised: torch.Tensor, 
        x_T: torch.Tensor, 
        w: float = 1.0, 
        stochastic: bool = False
    ):
        """Compute derivative for VE mode."""
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
            return d, gt2
        return d

    def _get_d_vp(
        self, 
        x: torch.Tensor, 
        denoised: torch.Tensor, 
        x_T: torch.Tensor, 
        sigma: torch.Tensor,
        w: float = 1.0, 
        stochastic: bool = False
    ):
        """Compute derivative for VP mode."""
        std_t = self._std(sigma)
        logsnr_t = self._logsnr(sigma)
        logsnr_T = self._logsnr(torch.as_tensor(self.sigma_max))
        logs_t = self._logs(sigma)
        logs_T = self._logs(torch.as_tensor(self.sigma_max))
        s_t_deriv = self._s_deriv(sigma)
        sigma_t = self._vp_snr_sqrt_reciprocal(sigma)
        sigma_t_deriv = self._vp_snr_sqrt_reciprocal_deriv(sigma)
        
        a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
        b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
        
        mu_t = a_t * x_T + b_t * denoised
        
        grad_logq = -(x - mu_t) / std_t**2 / (-torch.expm1(logsnr_T - logsnr_t))
        grad_logpxTlxt = -(x - torch.exp(logs_t - logs_T) * x_T) / std_t**2 / torch.expm1(logsnr_t - logsnr_T)
        
        f = s_t_deriv * (-logs_t).exp() * x
        gt2 = 2 * logs_t.exp()**2 * sigma_t * sigma_t_deriv
        
        d = f - gt2 * ((0.5 if not stochastic else 1) * grad_logq - w * grad_logpxTlxt)
        
        if stochastic:
            return d, gt2
        return d

    def _append_dims(self, x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(
                f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
            )
        return x[(...,) + (None,) * dims_to_append]

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        x_T: torch.Tensor,
        churn_step_ratio: float = 0.0,
        guidance: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[DDBMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE.

        Args:
            model_output (`torch.Tensor`):
                Direct output from the learned diffusion model (denoised prediction).
            timestep (`int`):
                Current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                Current instance of sample being created by diffusion process.
            x_T (`torch.Tensor`):
                Target/condition image for the bridge process.
            churn_step_ratio (`float`, defaults to 0.0):
                Ratio of churn steps for stochastic sampling.
            guidance (`float`, defaults to 1.0):
                Guidance weight for the sampling.
            generator (`torch.Generator`, *optional*):
                Random number generator.
            return_dict (`bool`, defaults to `True`):
                Whether to return a `DDBMSchedulerOutput` or tuple.

        Returns:
            [`DDBMSchedulerOutput`] or `tuple`:
                The predicted sample and optionally the predicted denoised sample.
        """
        if self.sigmas is None:
            raise ValueError("Sigmas not initialized. Call `set_timesteps` first.")
        
        i = timestep
        sigma = self.sigmas[i]
        sigma_next = self.sigmas[i + 1]
        
        s_in = sample.new_ones([sample.shape[0]])
        denoised = model_output  # Assume model outputs denoised prediction
        
        x = sample
        
        # Stochastic churn step
        if churn_step_ratio > 0 and sigma_next != 0:
            sigma_hat = (sigma_next - sigma) * churn_step_ratio + sigma
            
            if self.pred_mode == 've':
                d_1, gt2 = self._get_d_ve(x, sigma, denoised, x_T, w=guidance, stochastic=True)
            elif self.pred_mode.startswith('vp'):
                d_1, gt2 = self._get_d_vp(x, denoised, x_T, sigma, w=guidance, stochastic=True)
            
            dt = sigma_hat - sigma
            noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
            x = x + d_1 * dt + noise * (dt.abs() ** 0.5) * gt2.sqrt()
        else:
            sigma_hat = sigma
        
        # Heun step - first stage
        if self.pred_mode == 've':
            d = self._get_d_ve(x, sigma_hat, denoised, x_T, w=guidance)
        elif self.pred_mode.startswith('vp'):
            d = self._get_d_vp(x, denoised, x_T, sigma_hat, w=guidance)
        
        dt = sigma_next - sigma_hat
        
        if sigma_next == 0:
            # Final step
            prev_sample = x + d * dt
        else:
            # Heun's method - second stage
            x_2 = x + d * dt
            
            # Note: In actual usage, model needs to be called again for x_2
            # This scheduler assumes the caller handles the second model call
            # For simplicity, we use single-step Euler here
            # The full Heun implementation should be done in the pipeline
            prev_sample = x_2
        
        pred_original_sample = denoised

        if not return_dict:
            return (prev_sample, pred_original_sample)

        return DDBMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
    
    def step_heun(
        self,
        denoised_1: torch.Tensor,
        denoised_2: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        x_T: torch.Tensor,
        guidance: float = 1.0,
        return_dict: bool = True,
    ) -> Union[DDBMSchedulerOutput, Tuple]:
        """
        Perform a full Heun step with two model evaluations.
        
        Args:
            denoised_1: Denoised prediction at current sigma.
            denoised_2: Denoised prediction at next sigma.
            timestep: Current timestep index.
            sample: Current sample.
            x_T: Target/condition image.
            guidance: Guidance weight.
            return_dict: Whether to return a dict.
        """
        i = timestep
        sigma = self.sigmas[i]
        sigma_next = self.sigmas[i + 1]
        
        x = sample
        
        # First derivative
        if self.pred_mode == 've':
            d = self._get_d_ve(x, sigma, denoised_1, x_T, w=guidance)
        elif self.pred_mode.startswith('vp'):
            d = self._get_d_vp(x, denoised_1, x_T, sigma, w=guidance)
        
        dt = sigma_next - sigma
        
        if sigma_next == 0:
            prev_sample = x + d * dt
        else:
            x_2 = x + d * dt
            
            # Second derivative
            if self.pred_mode == 've':
                d_2 = self._get_d_ve(x_2, sigma_next, denoised_2, x_T, w=guidance)
            elif self.pred_mode.startswith('vp'):
                d_2 = self._get_d_vp(x_2, denoised_2, x_T, sigma_next, w=guidance)
            
            d_prime = (d + d_2) / 2
            prev_sample = x + d_prime * dt
        
        if not return_dict:
            return (prev_sample, denoised_1)
        
        return DDBMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=denoised_1)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        x_T: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to the original samples for the bridge diffusion process.

        Args:
            original_samples: Clean samples (x_0).
            noise: Random noise.
            timesteps: Timestep values (sigma values for DDBM).
            x_T: Target samples.
        """
        sigmas = timesteps.float()
        dims = original_samples.ndim
        sigmas = self._append_dims(sigmas, dims)
        
        if self.pred_mode.startswith('ve'):
            std_t = sigmas * torch.sqrt(1 - sigmas**2 / self.sigma_max**2)
            mu_t = sigmas**2 / self.sigma_max**2 * x_T + (1 - sigmas**2 / self.sigma_max**2) * original_samples
            noisy_samples = mu_t + std_t * noise
        elif self.pred_mode.startswith('vp'):
            logsnr_t = self._vp_logsnr(sigmas)
            logsnr_T = self._vp_logsnr(torch.as_tensor(self.sigma_max))
            logs_t = self._vp_logs(sigmas)
            logs_T = self._vp_logs(torch.as_tensor(self.sigma_max))
            
            a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
            b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
            std_t = (-torch.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t / 2).exp()
            
            noisy_samples = a_t * x_T + b_t * original_samples + std_t * noise
        else:
            raise ValueError(f"Unknown pred_mode: {self.pred_mode}")
        
        return noisy_samples

    def scale_model_input(
        self, 
        sample: torch.Tensor, 
        timestep: Optional[int] = None
    ) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input.
        """
        return sample
