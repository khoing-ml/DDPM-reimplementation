from typing import Sequence, Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# reuse helpers from models/nn.py
from models import nn as mnn


def swish(x):
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, *, temb_dim=None, dropout=0., conv_shortcut=False):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.temb_proj = nn.Linear(temb_dim, out_ch) if temb_dim is not None else None
        self.norm2 = nn.GroupNorm(32, out_ch)
        
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        if in_ch != out_ch:
            if conv_shortcut:
                self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            else:
                self.shortcut = mnn.Nin(in_ch, out_ch)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb=None):
        h = self.norm1(x)
        h = swish(h)
        h = self.conv1(h)
        if self.temb_proj is not None and temb is not None:
            h = h + self.temb_proj(swish(temb)).unsqueeze(-1).unsqueeze(-1)
        h = self.norm2(h)
        h = swish(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return self.shortcut(x) + h


class AttnBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.q = nn.Conv2d(ch, ch, kernel_size=1)
        self.k = nn.Conv2d(ch, ch, kernel_size=1)
        self.v = nn.Conv2d(ch, ch, kernel_size=1)
        self.proj_out = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).reshape(B, C, H * W).permute(0, 2, 1)  # B, HW, C
        k = self.k(h).reshape(B, C, H * W)  # B, C, HW
        v = self.v(h).reshape(B, C, H * W).permute(0, 2, 1)  # B, HW, C

        w = torch.bmm(q, k) * (C ** (-0.5))  # B, HW, HW
        w = F.softmax(w, dim=-1)
        h_attn = torch.bmm(w, v)  # B, HW, C
        h_attn = h_attn.permute(0, 2, 1).reshape(B, C, H, W)
        h_out = self.proj_out(h_attn)
        return x + h_out


class Downsample(nn.Module):
    def __init__(self, ch, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.op = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(2)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.op = nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)
        else:
            self.op = nn.Upsample(scale_factor=2.0, mode='nearest')

    def forward(self, x):
        return self.op(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_ch=3,
        ch=128,
        out_ch=3,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.,
        resolution=32,
        resamp_with_conv=True,
    ):
        super().__init__()
        self.ch = ch
        self.temb_dim = ch * 4
        self.num_resolutions = len(ch_mult)
        self.resolution = resolution
        self.attn_resolutions = set(attn_resolutions)
        # timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(ch, self.temb_dim),
            nn.SiLU(),
            nn.Linear(self.temb_dim, self.temb_dim),
        )

        # input conv
        self.conv_in = nn.Conv2d(in_ch, ch, kernel_size=3, padding=1)

        # build down
        self.down_resblocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsample = nn.ModuleList()
        in_channels = ch
        levels = [ch * m for m in ch_mult]
        curr_res = resolution
        skip_channels = [in_channels]
        for i_level, out_ch_level in enumerate(levels):
            resblocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(num_res_blocks):
                resblocks.append(ResnetBlock(in_channels, out_ch_level, temb_dim=self.temb_dim, dropout=dropout))
                in_channels = out_ch_level
                attns.append(AttnBlock(in_channels) if curr_res in self.attn_resolutions else nn.Identity())
                skip_channels.append(in_channels)
            self.down_resblocks.append(resblocks)
            self.down_attn.append(attns)
            if i_level != len(levels) - 1:
                self.downsample.append(Downsample(in_channels, with_conv=resamp_with_conv))
                skip_channels.append(in_channels)
                curr_res //= 2

        # middle
        self.mid = nn.ModuleList([
            ResnetBlock(in_channels, in_channels, temb_dim=self.temb_dim, dropout=dropout),
            AttnBlock(in_channels),
            ResnetBlock(in_channels, in_channels, temb_dim=self.temb_dim, dropout=dropout),
        ])

        # build up
        self.up_resblocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsample = nn.ModuleList()
        curr_res = resolution // (2 ** (self.num_resolutions - 1))
        skip_channels = list(reversed(skip_channels))
        for i_level, out_ch_level in list(enumerate(levels))[::-1]:
            resblocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop(0)
                resblocks.append(ResnetBlock(in_channels + skip_ch, out_ch_level, temb_dim=self.temb_dim, dropout=dropout))
                in_channels = out_ch_level
                attns.append(AttnBlock(in_channels) if curr_res in self.attn_resolutions else nn.Identity())
            self.up_resblocks.append(resblocks)
            self.up_attn.append(attns)
            if i_level != 0:
                self.upsample.append(Upsample(in_channels, with_conv=resamp_with_conv))
                curr_res *= 2

        # end
        self.norm_out = nn.GroupNorm(32, in_channels)
        self.conv_out = nn.Conv2d(in_channels, out_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        B = x.shape[0]
        temb = mnn.get_timestep_embedding(t, self.ch)
        temb = self.time_embed(temb)

        h = self.conv_in(x)
        hs = [h]

        # down
        for i_level, (resblocks, attns) in enumerate(zip(self.down_resblocks, self.down_attn)):
            for resblock, attn in zip(resblocks, attns):
                h = resblock(h, temb=temb)
                h = attn(h)
                hs.append(h)
            if i_level < len(self.downsample):
                h = self.downsample[i_level](h)
                hs.append(h)

        # middle
        for layer in self.mid:
            if isinstance(layer, ResnetBlock):
                h = layer(h, temb=temb)
            else:
                h = layer(h)

        # up
        for i_level, (resblocks, attns) in enumerate(zip(self.up_resblocks, self.up_attn)):
            for resblock, attn in zip(resblocks, attns):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = resblock(h, temb=temb)
                h = attn(h)
            if i_level < len(self.upsample):
                h = self.upsample[i_level](h)

        h = swish(self.norm_out(h))
        h = self.conv_out(h)
        return h


if __name__ == '__main__':
    # smoke test
    model = UNet(in_ch=3, ch=64, out_ch=3, ch_mult=(1, 2, 4), num_res_blocks=2, attn_resolutions=(16,), resolution=32)
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))
    y = model(x, t)
    print('output shape:', y.shape)
