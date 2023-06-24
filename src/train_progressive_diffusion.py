""" 
Trains a progressive diffusion model. Give command line options to 
override hyperparams. Can massive train for parameter search

Usage:
    python train_progressive_diffuion [-v] -c CHECKPT_FOLDER [HYPERPARAMS] -p DATA_PATH

Description:
    -c -- folder path to save checkpoints
    -p -- path to .npy data
    -v -- verbosity
    -s -- for torchinfo summary only (to check correctness of dimensions)
    --massive -- Run massive hyperparameter search training, with predefined
                set of possible hyperparameters
    ! For all hyperparams check options.py
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import torch
from src import Diffusion, Trainer, Unet
from torchinfo import summary
from util import (DiffusionConfig, GlobalConfig, PathConfig, TrainerConfig,
                  TrainOptions, UnetConfig, get_latest_checkpoint,
                  load_config_from_json, save_config, setup_logging)

log = logging.getLogger(__name__)


def initialize_model(
    unet_config: UnetConfig,
    trainer_config: TrainerConfig,
    diffusion_config: DiffusionConfig,
    path_config: PathConfig,
    device: torch.device,
):
    """
    Initializes diffusion, unet, and trainer modules
    """

    log.info("Initializing unet...")
    unet = Unet(unet_config).to(device)

    log.info("Initializing diffusion...")
    diffusion = Diffusion(unet, diffusion_config).to(device)

    log.info("Initializing trainer...")
    trainer = Trainer(diffusion, trainer_config, path_config)

    return diffusion, unet, trainer


def get_massive_train_params():
    """Generates possible hyperparams for parameter search"""
    params_override = {}

    for key, values in params_override.items():
        for value in values:
            exp_name = key.split("-")[-1] + str(value).replace(" ", "")
            if (key, value) == ("--kernel", 5):
                value = f"{value} --padding 2"
            elif (key, value) == ("--kernel", 7):
                value = f"{value} --padding 3"
            elif (key, value) == ("--kernel", 11):
                value = f"{value} --padding 5"
            yield f"{key} {value} --exp_name {exp_name}"


def parse_configs() -> Tuple[
    argparse.Namespace, UnetConfig, TrainerConfig, DiffusionConfig, PathConfig
]:
    """
    Overrides default training params if given as command line args
    """
    options = TrainOptions()
    args = options.parser.parse_args()

    arg_groups = {}
    for group in options.parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)

    train_config = arg_groups["train_config"]
    unet_config = UnetConfig.from_dict(vars(arg_groups["unet_config"]))  # type: ignore
    trainer_config = TrainerConfig.from_dict(vars(arg_groups["trainer_config"]))  # type: ignore
    diffusion_config = DiffusionConfig.from_dict(vars(arg_groups["diffusion_config"]))  # type: ignore
    path_config = PathConfig.from_dict(vars(arg_groups["path_config"]))  # type: ignore
    return train_config, unet_config, trainer_config, diffusion_config, path_config


def queue_commands(cmds: list[str]) -> None:
    """Queue list of commands for cluster"""
    procs = []
    for cmd in cmds:
        procs.append(subprocess.Popen(cmd.split(" ")))
    _ = [p.wait() for p in procs]


def main():

    (
        train_config,
        unet_config,
        trainer_config,
        diffusion_config,
        path_config,
    ) = parse_configs()

    # Set global logging level
    setup_logging(train_config.verbose)

    # Get device type
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

    if train_config.summary:
        _, denoiser, _ = initialize_model(
            unet_config, trainer_config, diffusion_config, path_config, device
        )
        summary(
            denoiser, input_size=[(32, 190, 500), (500,)]
        )  #! This summary is hardcoded, change the dimension when necessary
    elif train_config.massive:
        if train_config.continue_train:
            massive_folder = Path(train_config.continue_train)
            assert (
                massive_folder.exists()
            ), f"Massive training {massive_folder} doesn't exist."
            exp_folders = [d for d in massive_folder.iterdir() if d.is_dir()]
            cmds = []
            for exp_folder in exp_folders:
                cmd = f"python train_progressive_diffusion.py --continue_train {exp_folder}"
                cmds.append(cmd)
            # Run all commands
            queue_commands(cmds)
        else:
            cmds = []
            log.info("Queuing commands...")
            for params_override in get_massive_train_params():
                cmd = (
                    f"python train_progressive_diffusion.py {params_override} "
                    f"-p {path_config.data_path} "
                    f"-c {path_config.checkpoints_folder}"
                )
                cmds.append(cmd)
                # Run all commands
            queue_commands(cmds)
    elif train_config.continue_train:
        # Get checkpoints folder
        exp_folder = Path(train_config.continue_train)
        if exp_folder.exists():
            # If the experiment exists, load the config and continue training
            checkpoint = torch.load(get_latest_checkpoint(exp_folder))
            global_config = load_config_from_json(next(exp_folder.glob("**/*.json")))
            log.debug(f"{global_config.path_config=}")
            *_, trainer = initialize_model(*vars(global_config).values(), device)
            trainer.load(checkpoint)
            trainer.train()
        else:
            log.error(f"{exp_folder} doesn't exist, skipping...")

    else:
        global_config = GlobalConfig(
            unet_config, trainer_config, diffusion_config, path_config
        )
        *_, trainer = initialize_model(*vars(global_config).values(), device)
        path_config.data_path = str(path_config.data_path)
        path_config.checkpoints_folder = str(path_config.checkpoints_folder)
        save_config(global_config, trainer.checkpoints_folder, trainer.exp_name)
        trainer.train()


if __name__ == "__main__":
    main()
