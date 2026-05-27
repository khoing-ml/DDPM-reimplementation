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

    init_kwargs = {
        "project": project,
        "entity": getattr(cfg, "wandb_entity", None) or None,
        "name": getattr(cfg, "wandb_name", None) or None,
        "mode": getattr(cfg, "wandb_mode", "online") or "online",
        "config": asdict(cfg),
    }

    try:
        return wandb.init(**init_kwargs)
    except Exception as exc:
        # Permission/network failures in online mode should not kill training.
        if init_kwargs["mode"] == "online":
            print(f"[wandb] online init failed ({exc}); retrying in offline mode.")
            try:
                init_kwargs["mode"] = "offline"
                return wandb.init(**init_kwargs)
            except Exception as offline_exc:
                print(f"[wandb] offline init failed ({offline_exc}); disabling W&B logging.")
                return None

        print(f"[wandb] init failed ({exc}); disabling W&B logging.")
        return None


def wandb_image_from_grid(grid):
    try:
        import wandb
    except ImportError:
        return None

    if hasattr(grid, "detach"):
        grid = grid.detach().cpu()

    try:
        return wandb.Image(grid)
    except Exception as exc:
        print(f"[wandb] failed to convert grid to image ({exc}); skipping image log.")
        return None


def log_wandb_artifact(run, name: str, artifact_type: str, file_path: str | Path, metadata=None):
    if run is None:
        return None

    try:
        import wandb
    except ImportError:
        return None

    try:
        artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
        artifact.add_file(str(file_path))
        run.log_artifact(artifact)
        return artifact
    except Exception as exc:
        print(f"[wandb] failed to log artifact '{name}' ({exc}); continuing without artifact upload.")
        return None
