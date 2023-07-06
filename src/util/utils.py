import argparse
import logging
import subprocess
from inspect import isfunction
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

log = logging.getLogger(__name__)


def seed_everything(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_configs(OptionsCls: type) -> Dict[str, argparse.Namespace]:
    """
    Parse args into groups from given option class.

    Parameters:
    - OptionsCls: class with argparse parser object and added arguments.
    """
    options = OptionsCls()
    args = options.parser.parse_args()

    arg_groups = {}
    for group in options.parser._action_groups:
        arg_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**arg_dict)

    return arg_groups


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def get_latest_checkpoint(exp_folder: Path) -> Path:
    """Find the latest checkpoint in an exp folder"""
    checkpoints = list(exp_folder.glob("*.pt"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda f: f.stat().st_mtime)
    else:
        raise FileNotFoundError(f"There are no checkpionts in '{exp_folder}'")
    return latest_checkpoint


def setup_logging(v: int = 0, log_file_path: Optional[str] = None) -> None:
    """Set log level"""
    level = [logging.INFO, logging.DEBUG][min(1, v)]
    logger = logging.getLogger()
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # If also logging to file.
    if log_file_path:
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def save_config(configs, path) -> None:
    """Saves configs as YAML"""
    if path.exists():
        log.warning(f"Config file '{path}' already exists. Cannot overwrite.")
        return
    log.info(f"Saving configs to '{path}'.")
    with open(path, "w") as f:
        yaml.dump(configs, f)


def load_config(path: Path) -> object:
    """Load yaml hyperparam config file.

    Args:
        path : Path to dir containing config file.

    Raises:
        FileNotFoundError : If no yaml config file in dir.

    Returns:
        Dictionary of argument groups.
    """
    config_path = list(path.glob("*.yaml"))
    if config_path:
        with open(config_path[0], "r") as f:
            arg_groups = yaml.unsafe_load(f)
        return arg_groups
    else:
        raise FileNotFoundError(f"No configs found in '{path}'")


def exists(x) -> bool:
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    """Cycle dataloader.

    Args:
        dl : Torch DataLoader

    Yields:
        Batch of data
    """
    while True:
        for data in dl:
            yield data


def extract(a, t, x_shape):
    b, *_ = t.shape
    # Gets the values of vector a at index set t
    out = a.gather(-1, t)
    # Returns reshaped vector with batch size on first axis
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
