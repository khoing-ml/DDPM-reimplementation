"""
a small set of utilities used by the UNet port:
- `get_timestep_embedding`
- `Nin` (1x1 conv)
- `Dense` (linear)
- `conv2d` (conv factory)
- simple flatten / sumflat / meanflat helpers
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def flatten(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.shape[0], -1)


def sumflat(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.shape[0], -1).sum(dim=1)


def meanflat(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.shape[0], -1).mean(dim=1)


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Sinusoidal embeddings for timesteps.

    timesteps: 1-D tensor of shape (B,) with integer timestep values.
    Returns: float tensor of shape (B, embedding_dim).
    """
    assert timesteps.dim() == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / max(1, (half_dim - 1))
    freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Nin(nn.Module):
    """Network-in-network (1x1 convolution).

    Usage:
        nin = Nin(in_ch, out_ch)
        y = nin(x)
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Dense(nn.Module):
    """Simple linear layer wrapper.

    Note: TF code often used functional dense(x, name=..., num_units=...)
    that created variables on the fly; in PyTorch we provide a module.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def conv2d(in_ch: int, out_ch: int, filter_size: int = 3, stride: int = 1, padding: Optional[int] = None, bias: bool = True) -> nn.Module:
    """Factory that returns a Conv2d with sane padding default.

    This mirrors the TF helper `conv2d(..., filter_size=3, stride=1, pad='SAME')`.
    """
    if isinstance(filter_size, int):
        k = filter_size
    else:
        raise ValueError("filter_size must be int")
    if padding is None:
        padding = k // 2
    return nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=padding, bias=bias)


# Small initializer helper (optional)
def default_init(scale: float = 1.0):
    """Return an initializer function for module weights.

    The returned function accepts an nn.Module and in-place initializes its
    weights (and biases where applicable).
    """

    def _init(module: nn.Module):
        if hasattr(module, 'weight') and module.weight is not None:
            # Use kaiming uniform as a reasonable default, then scale.
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if scale != 1.0:
                with torch.no_grad():
                    module.weight.mul_(scale)
        if hasattr(module, 'bias') and module.bias is not None:
            fan_in = module.weight.size(1) if module.weight is not None and module.weight.dim() > 1 else None
            if fan_in is not None:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(module.bias, -bound, bound)
            else:
                nn.init.zeros_(module.bias)

    return _init
