""" 
Trainer class for TEDi diffusion.
"""

import logging
from pathlib import Path

import torch
from accelerate import Accelerator
from ema_pytorch import EMA
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from util import cycle, exists
from .datasets import Dataset
from .losses import GeometricLoss

log = logging.getLogger(__name__)


class Trainer:
    """Trainer class, handles training loop, saving, loading"""

    def __init__(self, diffusion, config, device) -> None:
        super().__init__()
        self.config = config
        # Convert to Path objects for convenience
        config.data_path = Path(config.data_path)
        config.checkpoints_folder = Path(config.checkpoints_folder)
        log.debug(f"{config.data_path=}")
        log.debug(f"{config.checkpoints_folder=}")

        self.exp_name = "_".join((config.data_path.stem, config.exp_name))
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
        self.curriculum_step = config.curriculum_step

        # Dataset, DataLoader settings
        self.ds = Dataset(config.data_path)
        dl = DataLoader(
            self.ds,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=4,  # TODO: Make option
        )
        dl = self.accelerator.prepare(dl)  # Prepare DataLoader for accelerator
        self.dl = cycle(dl)  # Cycle the DataLoader

        # Optimizer settings
        self.optimizer = Adam(
            diffusion.parameters(), lr=config.lr, betas=config.adam_betas
        )
        # If curriculum with LR schedule
        if self.curriculum_step != -1 and config.lr_schedule:
            from torch.optim.lr_scheduler import StepLR

            self.scheduler = StepLR(self.optimizer, step_size=100_000, gamma=0.1)
            self.schedular = self.accelerator.prepare(self.scheduler)
        else:
            self.scheduler = None
        if self.accelerator.is_main_process:
            self.ema = EMA(
                diffusion, beta=config.ema_decay, update_every=config.ema_update_every
            )

        # Individual experiments have own checkpoints folder
        # inside the general checkpoints folder
        self.checkpoints_folder = config.checkpoints_folder / f"{self.exp_name}"
        log.debug(f"{self.checkpoints_folder=}")
        self.checkpoints_folder.mkdir(exist_ok=True)

        # Make runs folder for tensorboard
        self.runs_dir = Path(f"../runs/{Path(*self.checkpoints_folder.parts[-2:])}")
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        # Step counter
        self.step = 0

        # Geometric losses
        self.geometric_loss = GeometricLoss(config.data_path, device)

        # Prepare model, dataloader, optimizer, scheduler with accelerator
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )

    def save(self, milestone):
        if not self.accelerator.is_main_process:
            return

        optimizer = self.accelerator.unwrap_model(self.optimizer)
        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
            if exists(self.scheduler)
            else None,
            "ema": self.ema.state_dict(),
            "scaler": self.accelerator.scaler.state_dict()
            if exists(self.accelerator.scaler)
            else None,
        }
        # ! remove existing checkpoints due to storage issue
        checkpoints = self.checkpoints_folder.glob("*.pt")
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
        if exists(self.scheduler):
            try:
                self.scheduler.load_state_dict(data["scheduler"])
            except KeyError:
                pass
            if self.scheduler._step_count < self.step + 1:
                for _ in range(self.step - 1):
                    self.scheduler.step()
            log.info(
                f"Training at step {self.step}, scheduler at step {self.scheduler._step_count}."
            )
        self.step = data["step"]
        self.optimizer.load_state_dict(data["opt"])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def train(self):
        writer = SummaryWriter(self.runs_dir)

        accelerator = self.accelerator
        device = accelerator.device

        # Stop training if there are nan gradients
        # torch.autograd.set_detect_anomaly(True)

        data = next(self.dl).float()
        log.info(f"Data tensor has shape {data.shape}")

        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not accelerator.is_main_process,
        ) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).float()
                    with self.accelerator.autocast():
                        loss = self.model(
                            data,
                            geometric_loss=self.geometric_loss,
                            curriculum=self.curriculum_step >= self.step,
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
                        # self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_every
                        self.save(milestone)

                self.step += 1
                if self.scheduler is not None:
                    self.scheduler.step()
                pbar.update(1)
        writer.close()
        accelerator.print("TRAINING COMPLETE!")
