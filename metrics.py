"""Evaluation metrics for CIFAR-10 diffusion training."""

from __future__ import annotations

import torch


def _get_fid_metric(device: torch.device):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError as exc:
        raise ImportError(
            "FID evaluation requires torchmetrics. Install it with `pip install torchmetrics`."
        ) from exc

    try:
        # FrechetInceptionDistance also needs the torch-fidelity backend at runtime.
        metric = FrechetInceptionDistance(normalize=True).to(device)
    except ModuleNotFoundError as exc:
        if "torch-fidelity" in str(exc).lower():
            return None
        raise

    return metric


@torch.no_grad()
def compute_fid(
    model,
    diffusion,
    loader,
    device: torch.device,
    image_size: int,
    num_samples: int = 1000,
    guidance_scale: float = 1.0,
    num_classes: int = 10,
):
    """Compute FID between samples from the model and a real loader.

    Images from the loader are expected to be in [-1, 1]. Generated samples are
    also converted to [0, 1] before being passed to the metric.
    """
    metric = _get_fid_metric(device)
    if metric is None:
        return None
    was_training = model.training
    model.eval()
    print(f"starting FID computation: num_samples={num_samples} guidance_scale={guidance_scale}", flush=True)

    try:
        real_seen = 0
        fake_seen = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            real = ((images + 1.0) / 2.0).clamp(0.0, 1.0)
            batch = min(real.shape[0], num_samples - real_seen, num_samples - fake_seen)
            if batch <= 0:
                break
            metric.update(real[:batch], real=True)
            real_seen += batch

            fake_labels = labels[:batch] % num_classes
            fake = diffusion.p_sample_loop(
                model.forward,
                shape=(batch, 3, image_size, image_size),
                device=device,
                labels=fake_labels,
                guidance_scale=guidance_scale,
            )
            fake = ((fake + 1.0) / 2.0).clamp(0.0, 1.0)
            metric.update(fake, real=False)
            fake_seen += batch

            if real_seen >= num_samples and fake_seen >= num_samples:
                break

        fid = metric.compute().item()
        print(f"finished FID computation: fid={fid:.4f}", flush=True)
        return fid
    finally:
        model.train(was_training)
