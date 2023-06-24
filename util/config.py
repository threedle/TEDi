import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from dataclasses_json import dataclass_json

log = logging.getLogger(__name__)


@dataclass_json
@dataclass
class PathConfig:
    data_path: Union[str, Path]
    checkpoints_folder: Union[str, Path]

    def __post_init__(self):
        self.data_path = Path(self.data_path) if type(self.data_path) == str else self.data_path
        self.checkpoints_folder = Path(self.checkpoints_folder) if type(self.checkpoints_folder) == str else self.checkpoints_folder
            


@dataclass_json
@dataclass
class UnetConfig:
    dim: int
    dim_mults: tuple[int]
    channels: int
    kernel: int
    stride: int
    padding: int
    norm: str
    resnet_block_groups: int
    spatial_attn: bool


@dataclass_json
@dataclass
class DiffusionConfig:
    T: int
    loss_type: str
    objective: str
    beta_schedule: str
    p2_loss_weight_gamma: float
    p2_loss_weight_k: float
    t_variation: float  # chance of training with variable t's
    fk_loss_lambda: float = 0.1
    detach_fc: bool = True

    def __post_init__(self):
        if self.beta_schedule not in ["cosine", "linear", "sigmoid"]:
            raise ValueError(f"Variance schedule {self.beta_schedule} not supported")


@dataclass_json
@dataclass
class TrainerConfig:
    exp_name: str
    ema_decay: float
    batch_size: int
    lr: float
    num_steps: int
    gradient_accumulate_every: int
    amp: bool
    ema_update_every: int
    adam_betas: tuple[float]
    fp16: bool
    split_batches: bool
    save_every: int
    pos_loss: bool
    velo_loss: bool


@dataclass_json
@dataclass
class GlobalConfig:
    unet_config: UnetConfig
    trainer_config: TrainerConfig
    diffusion_config: DiffusionConfig
    path_config: PathConfig
