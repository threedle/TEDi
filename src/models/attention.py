"""
Attention layer implementations.
"""

import logging

import torch
from einops import rearrange
from torch import einsum, nn

from .layers import LayerNorm

log = logging.getLogger(__name__)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x):
        *_, f = x.shape
        log.debug(f"dimension of input: {x.shape}")
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) f -> b h c f", h=self.heads),
            self.to_qkv(x).chunk(3, dim=1),
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c f -> b (h c) f")

        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        *_, f = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, "b (h c) f -> b h c f", h=self.heads), qkv)
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h i d -> b (h d) i")

        return self.to_out(out)


# class SpatialAttention(nn.Module):
#     def __init__(self, dim, heads=4, dim_head=32):
#         super().__init__()
#         self.scale = dim_head**-0.5
#         self.heads = heads
#         hidden_dim = dim_head * heads
#         self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
#         self.to_out = nn.Conv1d(hidden_dim, dim, 1)

#     def forward(self, x):
#         _, c, _ = x.shape
#         x = rearrange(x, "b c f -> b f c")
#         qkv = self.to_qkv(x).chunk(3, dim=1)
#         q, k, v = map(lambda t: rearrange(t, "b (h f) c -> b h f c", h=self.heads), qkv)
#         q = q * self.scale

#         sim = einsum("b h d i, b h d j -> b h i j", q, k)
#         sim = sim - sim.amax(dim=-1, keepdim=True).detach()
#         attn = sim.softmax(dim=-1)

#         out = einsum("b h i j, b h d j -> b h i d", attn, v)
#         out = rearrange(out, "b h i d -> b (h d) i", i=c)
#         out = self.to_out(out)
#         out = rearrange(out, "b f c -> b c f")
#         return out
