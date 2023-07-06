"""
Trains a progressive diffusion model.

Usage:
    python train_progressive_diffusion [-v] -c <CHECKPT_FOLDER> -p <DATA_PATH> [--<HYPERPARAM_1> <value> ...]

"""

import logging
from pathlib import Path

import torch
from models.diffusion import Diffusion
from models.trainer import Trainer
from models.unet import Unet
import argparse
from util import (
    TrainOptions,
    get_latest_checkpoint,
    load_config,
    parse_configs,
    save_config,
    set_device,
    setup_logging,
)

log = logging.getLogger(__name__)


def initialize_trainer(
    configs: dict[str, argparse.Namespace], device: torch.device
) -> Trainer:
    log.info("Initializing unet...")
    unet = Unet(configs["unet_config"])
    log.info("Initializing diffusion...")
    diffusion = Diffusion(unet, configs["diffusion_config"])
    log.info("Initializing trainer...")
    trainer = Trainer(diffusion, configs["trainer_config"], device)

    return trainer


def main():
    configs = parse_configs(TrainOptions)

    train_config = configs["train_config"]
    setup_logging(train_config.verbose)

    device = set_device()

    if train_config.continue_train:
        # Get checkpoints folder
        exp_folder = Path(train_config.continue_train)
        if exp_folder.exists():
            # If the experiment exists, load the config and continue training
            checkpoint = torch.load(get_latest_checkpoint(exp_folder))
            configs = load_config(exp_folder)
            trainer = initialize_trainer(configs, device)
            trainer.load(checkpoint)
            trainer.train()
        else:
            raise FileNotFoundError(f"Experiment folder '{exp_folder}' does not exist.")
    else:
        log.debug(configs)
        trainer = initialize_trainer(configs, device)
        save_config(configs, trainer.checkpoints_folder / "config.yaml")
        trainer.train()


if __name__ == "__main__":
    main()
