from __future__ import annotations

from datetime import timedelta
import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed():
    if not is_distributed():
        return False
    if not dist.is_initialized():
        timeout_minutes = int(os.environ.get("DDP_TIMEOUT_MINUTES", "120"))
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=timeout_minutes),
        )
    if torch.cuda.is_available():
        local_rank = get_local_rank()
        device_count = torch.cuda.device_count()
        if local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {device_count} CUDA device(s) are visible. "
                "Lower NUM_GPUS / --nproc_per_node or expose more GPUs via CUDA_VISIBLE_DEVICES."
            )
        torch.cuda.set_device(local_rank)
    return True


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def unwrap_model(model):
    return getattr(model, "module", model)
