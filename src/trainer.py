""" 
Trainer class for progressive diffusion
"""

import logging
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from ema_pytorch import EMA
from src.datasets import Dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from util import PathConfig, TrainerConfig, cycle, exists, parse_bvh_file

log = logging.getLogger(__name__)


class Trainer:
    """Trainer class, handles training loop, saving, loading"""

    def __init__(
        self, diffusion, config: TrainerConfig, path_config: PathConfig
    ) -> None:
        super().__init__()
        self.config = config

        self.exp_name = "_".join((path_config.data_path.stem, config.exp_name))
        log.debug(f"{self.exp_name=}")

        # Accelerator specific settings
        self.accelerator = Accelerator(
            split_batches=config.split_batches,
            mixed_precision="fp16" if config.fp16 else "no",
        )
        self.accelerator.native_amp = config.amp

        # Training specific settings
        self.model = diffusion
        self.save_every = config.save_every
        self.gradient_accumulate_every = config.gradient_accumulate_every
        self.train_num_steps = config.num_steps

        # Dataset/DataLoader settings
        self.ds = Dataset(path_config.data_path)
        dl = DataLoader(
            self.ds,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=4,
        )
        dl = self.accelerator.prepare(dl)  # Prepare DataLoader for accelerator
        self.dl = cycle(dl)  # Cycle the DataLoader

        # Optimizer settings
        self.optimizer = Adam(
            diffusion.parameters(), lr=config.lr, betas=config.adam_betas
        )
        if self.accelerator.is_main_process:
            self.ema = EMA(
                diffusion, beta=config.ema_decay, update_every=config.ema_update_every
            )
            self.checkpoints_folder = (
                Path(__file__).parent.parent
                / path_config.checkpoints_folder
                / f"{self.exp_name}"
            )
            self.checkpoints_folder.mkdir(exist_ok=True)

        # step counter
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )

        # Skeleton hierarchy and offsets for FK losses
        joints, fps, n = parse_bvh_file(
            "../train_data/01_01.bvh"
        )
        self.offsets = torch.tensor(
            [joint.offsets for joint in joints if not joint.end_joint]
        )
        self.offsets = self.offsets.unsqueeze(0)
        self.hierarchy = [0] * len([joint for joint in joints if not joint.end_joint])
        for joint in joints:
            if joint.end_joint:
                continue
            if joint.parent is None:
                self.hierarchy[joint.idx] = -1
            else:
                self.hierarchy[joint.idx] = joint.parent.idx

        # Mean and std for FK
        if "500" in self.exp_name:
            self.mean = torch.from_numpy(
                np.load(
                    "../train_data/cmu_full_500_mean.npy"
                )
            ).float()
            self.std = torch.from_numpy(
                np.load(
                    "../train_data/cmu_full_500_mean.npy"
                )
            ).float()
        else:
            raise ValueError(f"Exp name not good {self.exp_name}")

        self.pos_loss = config.pos_loss
        self.velo_loss = config.velo_loss

        # Make runs folder for tensorboard
        self.runs_dir = Path(
            f"../runs/{path_config.checkpoints_folder.stem}/{self.exp_name}"
        )
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def save(self, milestone):
        if not self.accelerator.is_main_process:
            return

        optimizer = self.accelerator.unwrap_model(self.optimizer)
        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": optimizer.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.accelerator.scaler.state_dict()
            if exists(self.accelerator.scaler)
            else None,
        }
        # remove existing checkpoints
        checkpoints = self.checkpoints_folder.glob("**/*.pt")
        for checkpoint in checkpoints:
            if not int(checkpoint.stem.split("-")[1]) % 50 == 0:
                checkpoint.unlink(missing_ok=False)
        torch.save(data, self.checkpoints_folder / f"model-{milestone}.pt")

    def load(self, data):
        model = self.accelerator.unwrap_model(self.model)
        opt = self.accelerator.unwrap_model(self.optimizer)
        model.load_state_dict(data["model"])
        opt.load_state_dict(data["opt"])
        self.step = data["step"]
        self.optimizer.load_state_dict(data["opt"])
        self.ema.load_state_dict(data["ema"])

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def train(self):
        writer = SummaryWriter(self.runs_dir)

        accelerator = self.accelerator
        device = accelerator.device

        # Stop training if there are nan gradients
        # torch.autograd.set_detect_anomaly(True)
        self.offsets = self.offsets.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).float().to(device)
                    log.debug(data.shape)
                    with self.accelerator.autocast():
                        loss = self.model(
                            data,
                            offsets=self.offsets,
                            hierarchy=self.hierarchy,
                            pos_loss=self.pos_loss,
                            velo_loss=self.velo_loss,
                            step=self.step,
                            mean=self.mean,
                            std=self.std,
                        )
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                writer.add_scalar("Loss", total_loss, self.step)
                pbar.set_description(f"loss: {total_loss:.4f}")

                accelerator.wait_for_everyone()

                self.optimizer.step()
                self.optimizer.zero_grad()

                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_every
                        self.save(milestone)

                self.step += 1
                pbar.update(1)
        writer.close()
        accelerator.print("Training complete!")
