from __future__ import annotations

from pathlib import Path

import torch
from torchvision import utils as tv_utils


@torch.no_grad()
def save_sample_grid(
    model,
    diffusion,
    device: torch.device,
    output_dir: Path,
    step: int,
    image_size: int,
    num_images: int = 64,
    labels: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    nrow: int | None = None,
):
    model.eval()
    if labels is not None:
        num_images = labels.shape[0]
    samples = diffusion.p_sample_loop(
        model.forward,
        shape=(num_images, 3, image_size, image_size),
        device=device,
        labels=labels,
        guidance_scale=guidance_scale,
    )
    samples = (samples.clamp(-1, 1) + 1.0) / 2.0
    grid = tv_utils.make_grid(samples, nrow=nrow or int(num_images**0.5)).cpu()
    output_dir.mkdir(parents=True, exist_ok=True)
    tv_utils.save_image(grid, output_dir / f"sample_{step:06d}.png")
    model.train()
    return grid


@torch.no_grad()
def save_real_fake_panel(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    output_dir: Path,
    step: int,
    num_images: int = 8,
):
    """Save a 2-row panel: real images on top, generated images on bottom."""
    real = ((real_images[:num_images].clamp(-1, 1) + 1.0) / 2.0).cpu()
    fake = ((fake_images[:num_images].clamp(-1, 1) + 1.0) / 2.0).cpu()
    panel = tv_utils.make_grid(torch.cat([real, fake], dim=0), nrow=num_images)
    output_dir.mkdir(parents=True, exist_ok=True)
    tv_utils.save_image(panel, output_dir / f"real_fake_{step:06d}.png")
    return panel
