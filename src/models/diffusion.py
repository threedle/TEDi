""" 
Gaussian diffusion for motion. Also constains geometric losses (FK & velo).
"""

import logging
from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import Tensor, nn
from tqdm.auto import tqdm
from util import default, extract, torch_forward_kinematics

from .losses import GeometricLoss
from .variance_schedule import (cosine_beta_schedule, linear_beta_schedule,
                                sigmoid_beta_schedule)

log = logging.getLogger(__name__)

# For original CMU
ROTS_IDX = slice(93, -4)
CONTACT_JOINTS_IDX = [4, 5, 9, 10]


def fk_loss_fn(
    model_out: Tensor,
    t: Tensor,
    offsets: Tensor,
    hierarchy: list[int],
    target: Tensor,
    loss_weight: Tensor,
) -> Tuple[Tensor, Tensor]:
    l, b, c = model_out.shape

    fk_pos = torch_forward_kinematics(
        rearrange(
            model_out[..., ROTS_IDX], "l b (j k) -> b l j k", k=6
        ),  # reshaped 6D rotations
        offsets=offsets,  # offsets of skeleton
        root_pos=torch.zeros(b, l, 3, device=model_out.device),
        hierarchy=hierarchy,
        world=True,
    )
    target_fk_pos = torch_forward_kinematics(
        rearrange(
            target[..., ROTS_IDX], "l b (j k) -> b l j k", k=6
        ),  # reshaped 6D rotations
        offsets=offsets,  # offsets of skeleton
        root_pos=torch.zeros(b, l, 3, device=model_out.device),
        hierarchy=hierarchy,
        world=True,
    )
    log.debug(f"{fk_pos.shape=}, {target_fk_pos.shape=}")
    fk_pos_loss = F.mse_loss(
        rearrange(fk_pos, "b l j k -> l b (j k)")[1:, ...],
        rearrange(target_fk_pos, "b l j k -> l b (j k)")[1:, ...],
        reduction="none",
    )
    fk_pos_loss = reduce(fk_pos_loss, "l ... -> l (...)", "mean")
    fk_pos_loss = fk_pos_loss * extract(loss_weight, t[1:], fk_pos_loss.shape)
    return fk_pos_loss, fk_pos


def velo_loss_fn(
    model_out: Tensor,
    t: Tensor,
    offsets: Tensor,
    hierarchy: list[int],
    loss_weight: Tensor,
    mean: Tensor,
    std: Tensor,
    detach: bool,
) -> Tensor:

    contact = model_out[..., -4:]
    contact = rearrange(contact, "l b k -> l k b")
    root_pos = rearrange(model_out[..., :3], "l b k -> b l k")
    root_pos = root_pos * std[None, None, :3] + mean[None, None, :3]
    root_pos_y = root_pos[..., 2:3]
    root_pos = root_pos.cumsum(dim=1)
    root_pos = torch.cat((root_pos[..., 0:1], root_pos_y, root_pos[..., 1:2]), dim=2)
    rots = model_out[..., ROTS_IDX]
    rots = rearrange(rots, "l b (j k) -> b l j k", k=6)
    fk_pos = torch_forward_kinematics(rots, offsets, root_pos, hierarchy, world=True)

    fc_loc = torch.tensor(CONTACT_JOINTS_IDX, device=model_out.device)
    model_fc = rearrange(torch.index_select(fk_pos, 2, fc_loc), "b l j k -> l b (j k)")

    model_fc_label = model_out[..., -4:]
    if detach:
        model_fc_label = model_fc_label.detach()

    model_fc_label = torch.sigmoid((model_fc_label - 0.5) * 12)

    # Velocity loss on foot joints
    velo = (model_fc[1:, ...] - model_fc[:-1, ...]).chunk(4, 2)
    velo = [
        torch.mul(v, model_fc_label[:-1, :, i].unsqueeze(2)) for i, v in enumerate(velo)
    ]

    velo = torch.cat(velo, dim=2)

    velo_loss = F.mse_loss(
        velo, torch.zeros_like(velo, device=velo.device), reduction="none"
    )
    velo_loss = reduce(velo_loss, "l ... -> l (...)", "mean")
    velo_loss = velo_loss * extract(loss_weight, t[1:], velo_loss.shape)
    return velo_loss


class Diffusion(nn.Module):
    def __init__(self, denoiser, config):
        super().__init__()
        self.channels = denoiser.channels
        self.T = config.T
        self.denoiser = denoiser
        self.objective = config.objective
        self.loss_type = config.loss_type
        self.t_variation = config.t_variation
        self.pos_loss = config.pos_loss
        self.velo_loss = config.velo_loss
        self.fk_loss_lambda = config.fk_loss_lambda
        self.detach_fc = config.detach_fc

        if config.beta_schedule == "cosine":
            betas = cosine_beta_schedule(config.T)
        elif config.beta_schedule == "linear":
            betas = linear_beta_schedule(config.T)
        elif config.beta_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(config.T)
        else:
            raise ValueError(f"{config.beta_schedule} is not supported.")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # calculate p2 reweighting
        register_buffer(
            "p2_loss_weight",
            (config.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -config.p2_loss_weight_gamma,
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        log.debug(f"{extract(self.posterior_mean_coef1, t, x_t.shape).size()=}")
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        log.debug(f"{x.shape=}, {t.shape=}")
        _x = rearrange(x, "l b c -> b c l")
        model_output = self.denoiser(_x, t)
        model_output = rearrange(model_output, "b c l -> l b c")

        if self.objective == "pred_noise":
            x_start = self.predict_start_from_noise(x, t=t, noise=model_output)
        elif self.objective == "pred_x0":
            x_start = model_output
        else:
            raise ValueError(f"unknown objective {self.objective}")

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x)
        # no noise when t == 0, but should be disabled with parallel generation
        # if t[0] == 0:
        #     nonzero_mask = 0
        # else:
        #     nonzero_mask = 1
        # nonzero_mask = 1.
        nonzero_mask = (1 - (t == 0).float()).reshape(
            self.T, *((1,) * (len(x.shape) - 1))
        )
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.betas.device
        imgs = []
        img = torch.randn(shape, device=device)
        imgs.append(img)
        for i in tqdm(
            reversed(range(0, self.T)), desc="sampling loop time step", total=self.T
        ):
            img = self.p_sample(
                img, torch.full((self.T,), i, device=device, dtype=torch.long)
            )
        return img

    @torch.no_grad()
    def sample(self, batch_size=1):
        motion_length = self.T
        channels = self.channels
        return self.p_sample_loop((motion_length, batch_size, channels))

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == "l1":
            return F.l1_loss
        elif self.loss_type == "l2":
            return F.mse_loss
        else:
            raise ValueError(f"invalid loss type {self.loss_type}")

    def p_losses(self, x_start, t, noise=None, *args, **kwargs):
        # Generate the noisy samples, rearranging x_start so the noisy
        # samples are generated across the right axis
        x_start = rearrange(x_start, "b c l -> l b c")
        noise = default(noise, lambda: torch.randn_like(x_start))
        log.debug(f"{x_start.shape=}, {t.shape=}, {noise.shape=}.")
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Rearrange x and pass to the network
        model_out = self.denoiser(rearrange(x, "l b c -> b c l"), t)
        # Rearrange model_out to pose order
        model_out = rearrange(model_out, "b c l -> l b c")

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        else:
            raise ValueError(f"unknown objective {self.objective}")

        # Compute loss
        loss = self.loss_fn(model_out, target, reduction="none")
        loss = reduce(loss, "l ... -> l (...)", "mean")
        loss *= extract(self.p2_loss_weight, t, loss.shape)

        geometric_loss: GeometricLoss = kwargs["geometric_loss"]

        if self.pos_loss:
            if self.velo_loss:
                return (
                    (1 - self.fk_loss_lambda) * loss.mean()
                    + (self.fk_loss_lambda / 2) * geometric_loss.fk_loss(model_out, target, t, self.p2_loss_weight).mean()
                    + (self.fk_loss_lambda / 2) * geometric_loss.ik_loss(model_out, t, self.p2_loss_weight).mean()
                )
            else:
                return (1 - self.fk_loss_lambda) * loss.mean() + (
                    self.fk_loss_lambda
                ) * geometric_loss.fk_loss(model_out, target, t, self.p2_loss_weight).mean()
        else:
            return loss.mean()

    def forward(self, motion: torch.Tensor, *args, **kwargs):
        b, c, l, device, = (
            *motion.shape,
            motion.device,
        )
        log.debug(f"({b}, {c}, {l}) is the input shape, with device {device}")

        # For curriculum
        if kwargs["curriculum"]:
            # Same t's
            t = torch.randint(0, self.T, (1,), device=device).long().repeat(self.T)
        elif torch.rand(1) >= self.t_variation:
            # Instead of sampling t as a random vector, we assign t = (0,1,...,T)
            t = torch.tensor(list(range(0, self.T)), device=device).long()
        else:
            # Randomized t's
            t = torch.randint(0, self.T, (self.T,), device=device).long()
        log.debug(f"{t=}")
        return self.p_losses(motion, t, *args, **kwargs)
