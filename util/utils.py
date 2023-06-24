import logging
import sys
from inspect import isfunction
from pathlib import Path

from util.config import GlobalConfig

log = logging.getLogger(__name__)


def get_latest_checkpoint(exp_folder: Path) -> Path:
    """Find the latest checkpoint in an exp folder"""
    checkpoints = list(exp_folder.glob("**/*.pt"))
    assert len(checkpoints) > 0, f"There are no checkpoints in the directory {exp_folder}"
    latest_checkpoint = max(checkpoints, key=lambda a: int(a.stem.split("-")[1]))
    return latest_checkpoint


def setup_logging(level: int):
    """ Set log level """
    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logging.basicConfig(stream=sys.stdout, level=levels[min(level, 2)])


def save_config(config: GlobalConfig, checkpoints_folder: Path, exp_name: str) -> None:
    """Saves config class as JSON under the specified folder"""
    config_path = checkpoints_folder / f"{exp_name}_config.json"
    config_path.touch(exist_ok=True)
    json = config.to_json(indent=4)
    log.info(json)
    with open(config_path, "w") as f:
        f.write(json)


def load_config_from_json(json_path: Path) -> GlobalConfig:
    """Loads a GlobalConfig object from JSON file"""
    f = open(json_path, "r")
    lines = f.read()
    config = GlobalConfig.from_json(lines)
    f.close()
    return config


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def extract(a, t, x_shape):
    b, *_ = t.shape
    # Gets the values of vector a at index set t
    out = a.gather(-1, t)
    # Returns reshaped vector with batch size on first axis
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
