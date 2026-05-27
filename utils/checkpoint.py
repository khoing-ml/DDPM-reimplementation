from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch


def save_checkpoint(model, optimizer, output_dir: Path, epoch: int, cfg, global_step: int = 0):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = getattr(model, "module", model)
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model_to_save.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": asdict(cfg),
    }
    path = output_dir / f"checkpoint_{epoch:06d}.pt"
    torch.save(ckpt, path)
    return path


def load_checkpoint(model, optimizer, checkpoint_path: str | Path, map_location: str | torch.device = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model_to_load = getattr(model, "module", model)
    model_to_load.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint
