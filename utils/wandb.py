from __future__ import annotations

from dataclasses import asdict
from pathlib import Path


def setup_wandb(cfg):
    project = getattr(cfg, "wandb_project", "")
    if not project:
        return None

    try:
        import wandb
    except ImportError:
        raise RuntimeError("W&B logging was requested but `wandb` is not installed. Install it with `pip install wandb`.")

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

    if hasattr(grid, "detach"):
        grid = grid.detach().cpu()

    return wandb.Image(grid)


def log_wandb_artifact(run, name: str, artifact_type: str, file_path: str | Path, metadata=None):
    if run is None:
        return None

    try:
        import wandb
    except ImportError:
        return None

    artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
    artifact.add_file(str(file_path))
    run.log_artifact(artifact)
    return artifact
