"""Evaluation metrics for CIFAR-10 diffusion training."""

from __future__ import annotations

from typing import Optional

import torch


def _get_fid_metric(device: torch.device):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError as exc:
        raise ImportError(
            "FID evaluation requires torchmetrics. Install it with `pip install torchmetrics`."
        ) from exc

    # normalize=True lets us feed float images in [0, 1].
    metric = FrechetInceptionDistance(normalize=True).to(device)
    return metric


@torch.no_grad()
def compute_fid(
    model,
    diffusion,
    loader,
    device: torch.device,
    image_size: int,
    num_samples: int = 1000,
):
    """Compute FID between samples from the model and a real loader.

    Images from the loader are expected to be in [-1, 1]. Generated samples are
    also converted to [0, 1] before being passed to the metric.
    """
    metric = _get_fid_metric(device)
    model.eval()

    real_seen = 0
    fake_seen = 0

    for images, _ in loader:
        images = images.to(device)
        real = ((images + 1.0) / 2.0).clamp(0.0, 1.0)
        batch = min(real.shape[0], num_samples - real_seen)
        if batch <= 0:
            break
        metric.update(real[:batch], real=True)
        real_seen += batch
        if real_seen >= num_samples:
            break

    while fake_seen < num_samples:
        batch = min(loader.batch_size or 1, num_samples - fake_seen)
        fake = diffusion.p_sample_loop(model.forward, shape=(batch, 3, image_size, image_size), device=device)
        fake = ((fake + 1.0) / 2.0).clamp(0.0, 1.0)
        metric.update(fake, real=False)
        fake_seen += batch

    score = metric.compute().item()
    model.train()
    return score
