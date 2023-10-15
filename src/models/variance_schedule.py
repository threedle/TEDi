"""
Variance schedules for diffusion.
"""
import logging
import math

import torch
from torch import Tensor

log = logging.getLogger(__name__)


def linear_beta_schedule(timesteps: int) -> Tensor:
    """
    Linear beta schedule.

    Scale is 1000 / timesteps, and beta starts at scale * 0.0001 and ends at
    scale * 0.02.

    Parameters:
    - timesteps: Number of diffusion timesteps.

    Returns:
    A 1d tensor of linear beta values of shape (timesteps)
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    """
    Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ.

    Parameters:
    - timesteps: Number of diffusion timesteps.
    - s: Small constant to prevent bad division.

    Returns:
    A 1d tensor of cosine beta values of shape (timesteps)
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(
    timesteps: int, start: int = -3, end: int = 3, tau: int = 1, clamp_min: float = 1e-5
) -> Tensor:
    """
    Sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training

    Parameters:
    - timesteps: Number of diffusion timesteps.

    Returns:
    A 1d tensor of sigmoid beta values of shape (timesteps)
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
