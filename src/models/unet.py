"""
Unet denoiser for diffusion
"""

import logging
from functools import partial

import torch
from einops import rearrange
from torch import nn

from .attention import Attention, LinearAttention
from .layers import Downsample, Prenorm, Residual, ResnetBlock, Upsample
from .pos_emb import SinusoidalPosEmb

log = logging.getLogger(__name__)


class Unet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.channels = config.channels

        self.init_conv = nn.Conv1d(config.channels, config.dim, 7, padding=3)
        dims = [config.dim] + [config.dim * i for i in config.dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = config.dim * 4
        sinu_pos_emb = SinusoidalPosEmb(config.dim)
        fourier_dim = config.dim

        block = partial(
            ResnetBlock,
            time_dim=time_dim,
            groups=config.resnet_block_groups,
            norm=config.norm,
            kernel_size=config.kernel,
            padding=config.padding,
        )

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs, self.ups = nn.ModuleList([]), nn.ModuleList([])
        num_resolutions = len(in_out)

        log.debug(f"{num_resolutions=}, {in_out=}")

        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i == (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block(dim_in, dim_in),
                        block(dim_in, dim_in),
                        Residual(Prenorm(dim_in, LinearAttention(dim_in))),
                        Downsample(
                            dim_in,
                            dim_out,
                            kernel_size=config.kernel,
                            stride=config.stride,
                            padding=config.padding,
                        )
                        if not is_last
                        else nn.Conv1d(
                            dim_in, dim_out, config.kernel, padding=config.padding
                        ),
                        Downsample(
                            time_dim,
                            time_dim,
                            kernel_size=config.kernel,
                            stride=config.stride,
                            padding=config.padding,
                        )
                        if not is_last
                        else nn.Conv1d(
                            time_dim, time_dim, config.kernel, padding=config.padding
                        ),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block(mid_dim, mid_dim)
        self.mid_attn = Residual(Prenorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block(mid_dim, mid_dim)

        for i, (dim_out, dim_in) in enumerate(reversed(in_out)):
            is_last = i == (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block(dim_in + dim_out, dim_in),
                        block(dim_in + dim_out, dim_in),
                        Residual(Prenorm(dim_in, LinearAttention(dim_in))),
                        Upsample(
                            dim_in,
                            dim_out,
                            kernel_size=config.kernel,
                            padding=config.padding,
                        )
                        if not is_last
                        else nn.Conv1d(
                            dim_in, dim_out, config.kernel, padding=config.padding
                        ),
                        Upsample(
                            time_dim,
                            time_dim,
                            kernel_size=config.kernel,
                            padding=config.padding,
                        )
                        if not is_last
                        else nn.Conv1d(
                            time_dim, time_dim, config.kernel, padding=config.padding
                        ),
                    ]
                )
            )

        self.final_res_block = block(config.dim * 2, config.dim)
        self.final_conv = nn.Conv1d(config.dim, config.channels, 1)

    def forward(self, x, t):
        x = self.init_conv(x)
        x_ = x.clone()

        t = self.time_mlp(t)

        skip_connects = []

        for block1, block2, *attn, down_x, down_t in self.downs:
            x = block1(x, t)
            skip_connects.append(x)
            x = block2(x, t)
            x = attn[0](x)
            skip_connects.append(x)
            x = down_x(x)
            t = rearrange(t, "t c -> c t")
            t = down_t(t)
            t = rearrange(t, "c t-> t c")

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, *attn, up_x, up_t in self.ups:
            x = torch.cat((x, skip_connects.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, skip_connects.pop()), dim=1)
            x = block2(x, t)
            x = attn[0](x)
            x = up_x(x)
            t = rearrange(t, "t c -> 1 c t")
            t = up_t(t)
            t = rearrange(t, "1 c t-> t c")

        log.debug(f"x has shape {x.shape}, x_ has shape {x_.shape}")
        x = torch.cat((x, x_), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)
