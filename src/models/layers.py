"""
Resnet block layers, normalization layers
"""

import logging

import torch
from einops import rearrange
from torch import nn
from util import default, exists

log = logging.getLogger(__name__)


class Residual(nn.Module):
    def __init__(self, fct):
        super().__init__()
        self.fct = fct

    def forward(self, x, *args):
        return self.fct(x, *args) + x


def Upsample(dim, dim_out=None, kernel_size=5, padding=2):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="linear"),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding=padding),
    )


def Downsample(dim, dim_out=None, kernel_size=5, stride=2, padding=1):
    return nn.Conv1d(dim, default(dim_out, dim), kernel_size, stride, padding)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.gamma


class Prenorm(nn.Module):
    def __init__(self, dim, fct):
        super().__init__()
        self.fct = fct
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fct(self.norm(x))


class Block(nn.Module):
    def __init__(self, dim, dim_out, norm="group", groups=8, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim_out, kernel_size, padding=padding)

        if norm == "group":
            self.norm = nn.GroupNorm(groups, dim_out)
        elif norm == "instance":
            self.norm = nn.InstanceNorm1d(dim_out)
        else:
            raise ValueError(f"Normalization {norm} not supported.")

        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.norm(self.conv(x))

        if exists(scale_shift):
            scale, shift = scale_shift
            log.debug(
                f"scale shape: {scale.shape}, shift shape: {shift.shape}, x shape: {x.shape}"
            )
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        time_dim=None,
        groups=8,
        norm="group",
        kernel_size=3,
        padding=1,
    ):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, dim_out * 2))
        self.block1 = Block(dim, dim_out, norm, groups, kernel_size, padding)
        self.block2 = Block(dim_out, dim_out, norm, groups, kernel_size, padding)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        # time_emb = rearrange(self.mlp(time_emb), 'b c -> b c 1')
        log.debug(f"time emb shape: {time_emb.shape}")
        time_emb = rearrange(self.mlp(time_emb), "t c -> 1 c t")
        log.debug(f"time emb shape after reshape: {time_emb.shape}")
        scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)
