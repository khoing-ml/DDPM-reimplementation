from __future__ import annotations

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
        dist.init_process_group(backend="nccl", init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())
    return True


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def unwrap_model(model):
    return getattr(model, "module", model)
