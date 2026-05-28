"""DDPM utilities for CIFAR-10 training and sampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _warmup_beta(beta_start: float, beta_end: float, num_timesteps: int, warmup_frac: float):
    betas = beta_end * torch.ones(num_timesteps, dtype=torch.float64)
    warmup_time = int(num_timesteps * warmup_frac)
    if warmup_time > 0:
        betas[:warmup_time] = torch.linspace(beta_start, beta_end, warmup_time, dtype=torch.float64)
    return betas


def get_beta_schedule(
    beta_schedule: str,
    *,
    beta_start: float,
    beta_end: float,
    num_diffusion_timesteps: int,
):
    if beta_schedule == "quad":
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_diffusion_timesteps, dtype=torch.float64) ** 2
    elif beta_schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=torch.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * torch.ones(num_diffusion_timesteps, dtype=torch.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / torch.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=torch.float64)
    else:
        raise NotImplementedError(beta_schedule)
    return betas


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    bs = t.shape[0]
    out = a.gather(0, t)
    return out.reshape(bs, *([1] * (len(x_shape) - 1)))


class GaussianDiffusion:
    def __init__(self, *, betas: torch.Tensor):
        betas = betas.to(dtype=torch.float64)
        if betas.ndim != 1:
            raise ValueError("betas must be a 1D tensor")
        if not torch.all((betas > 0) & (betas <= 1)):
            raise ValueError("betas must be in (0, 1]")

        self.betas = betas
        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]])

        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return extract(self.sqrt_alphas_cumprod.to(x_start.device).to(x_start.dtype), t, x_start.shape) * x_start + extract(
            self.sqrt_one_minus_alphas_cumprod.to(x_start.device).to(x_start.dtype), t, x_start.shape
        ) * noise

    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        coef1 = extract(self.posterior_mean_coef1.to(x_t.device).to(x_t.dtype), t, x_t.shape)
        coef2 = extract(self.posterior_mean_coef2.to(x_t.device).to(x_t.dtype), t, x_t.shape)
        posterior_mean = coef1 * x_start + coef2 * x_t
        posterior_variance = extract(self.posterior_variance.to(x_t.device).to(x_t.dtype), t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped.to(x_t.device).to(x_t.dtype), t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_xstart_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        return extract(self.sqrt_recip_alphas_cumprod.to(x_t.device).to(x_t.dtype), t, x_t.shape) * x_t - extract(
            self.sqrt_recipm1_alphas_cumprod.to(x_t.device).to(x_t.dtype), t, x_t.shape
        ) * eps

    def p_mean_variance(
        self,
        denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ):
        if labels is not None and guidance_scale != 1.0:
            uncond_output = denoise_fn(x, t, labels=None)
            cond_output = denoise_fn(x, t, labels=labels)
            model_output = uncond_output + guidance_scale * (cond_output - uncond_output)
        else:
            model_output = denoise_fn(x, t, labels=labels)
        pred_xstart = self.predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        pred_xstart = pred_xstart.clamp(-1.0, 1.0)
        model_mean, _, model_log_variance = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        return model_mean, model_log_variance, pred_xstart

    @torch.no_grad()
    def p_sample(
        self,
        denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ):
        model_mean, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn,
            x=x,
            t=t,
            labels=labels,
            guidance_scale=guidance_scale,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().reshape(x.shape[0], *([1] * (x.dim() - 1)))
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return sample, pred_xstart

    @torch.no_grad()
    def p_sample_loop(
        self,
        denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        shape: Tuple[int, ...],
        device: torch.device,
        labels: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ):
        img = torch.randn(shape, device=device)
        if labels is not None:
            labels = labels.to(device=device, dtype=torch.long)
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img, _ = self.p_sample(denoise_fn, x=img, t=t, labels=labels, guidance_scale=guidance_scale)
        return img

    def training_losses(
        self,
        denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        cond_drop_prob: float = 0.0,
    ):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        target = noise
        model_output = denoise_fn(x_t, t, labels=labels, cond_drop_prob=cond_drop_prob)
        return F.mse_loss(model_output, target, reduction="none").mean(dim=(1, 2, 3))
