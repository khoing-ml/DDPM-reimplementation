from __future__ import annotations

from dataclasses import asdict
import os


def setup_wandb(cfg):
    project = getattr(cfg, "wandb_project", "")
    if not project:
        return None

    try:
        import wandb
    except ImportError:
        raise RuntimeError("W&B logging was requested but `wandb` is not installed. Install it with `pip install wandb`.")

    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "W&B logging was requested but no login was found. Run `wandb login` or set `WANDB_API_KEY` before training."
        )

    run = wandb.init(
        project=project,
        entity=getattr(cfg, "wandb_entity", None) or None,
        name=getattr(cfg, "wandb_name", None) or None,
        mode=getattr(cfg, "wandb_mode", "online") or "online",
        config=asdict(cfg),
    )
    return run


def wandb_image_from_grid(grid):
    try:
        import wandb
    except ImportError:
        return None
    return wandb.Image(grid)
