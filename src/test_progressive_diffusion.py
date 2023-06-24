""" 
This program loads a trained model and tests it.

Usage:
[--massive] [massive training folder]: pass in the massive training folder and it
will generate a few pieces of motion per experiment.
[--exp_names] [LIST of experiment folders]: pass in specific experiment folder for
generation.
"""

import logging
import traceback
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from einops import rearrange
from src import Diffusion, Trainer, Unet
from src.train_progressive_diffusion import initialize_model
from torch import Tensor
from tqdm.auto import tqdm
from util import (GlobalConfig, TestOptions, get_latest_checkpoint,
                  load_config_from_json, setup_logging)
from util.animation import test_to_bvh

log = logging.getLogger(__name__)

#! Must be CUDA
device = torch.device("cuda")

MODES = ("unconditional")


def get_file_num(save_dir: Path, name: str) -> int:
    i = 0
    while (save_dir / f"{name}{i}.pt").exists():
        i += 1
    return i


def get_random_primer_idx(dataset_len: int, n_samples: int):
    idx = np.random.randint(0, dataset_len, size=n_samples)
    log.info(idx)
    return idx


def load_checkpoint(exp_dir, milestone) -> Tuple[GlobalConfig, Diffusion, Trainer]:
    """Loads desired checkpoint for an experiment directory"""
    if milestone == 0:
        checkpoint = torch.load(get_latest_checkpoint(exp_dir), map_location=device)
        global_config = load_config_from_json(next(exp_dir.glob("**/*.json")))
        diffusion, denoiser, trainer = initialize_model(
            global_config.unet_config,
            global_config.trainer_config,
            global_config.diffusion_config,
            global_config.path_config,
            device,
        )
        trainer.load(checkpoint)
        log.info(f"Checkpoint loaded, step {checkpoint['step']}.")

    else:
        checkpoint = torch.load(exp_dir / f"model-{milestone}.pt", map_location=device)
        global_config = load_config_from_json(next(exp_dir.glob("**/*.json")))
        diffusion, denoiser, trainer = initialize_model(
            global_config.unet_config,
            global_config.trainer_config,
            global_config.diffusion_config,
            global_config.path_config,
            device,
        )
        trainer.load(checkpoint)
        log.info(f"Checkpoint loaded, step {checkpoint['step']}.")
    return global_config, diffusion, trainer


def main():
    # Get args
    opt = TestOptions()
    test_config = opt.parser.parse_args()

    # Check if args are correct
    assert test_config.mode in MODES, f"{test_config.mode} not supported."

    # Set log level
    setup_logging(test_config.verbose)

    if test_config.massive:
        # Get experiments in the massive folder
        massive_dir = Path(test_config.massive)
        assert massive_dir.exists(), f"Given massive directory doesn't exist."
        exp_dirs = [d for d in massive_dir.iterdir() if d.is_dir()]
        test_experiments(
            exp_dirs,
            test_config.mode,
            test_config.n_samples,
            test_config.sample_len,
            test_config.milestones,
            massive_dir=massive_dir,
        )


def test_experiments(
    exp_dirs, mode, n_samples, sample_len, milestones: list[int], massive_dir=None
):
    """Test experiments with given set of parameters"""
    dataset = None
    primers = None
    t = None  # time vector
    output = {}
    log.info(f"Testing mode: {mode}")
    for i, exp_dir in enumerate(exp_dirs):
        # Try catch block in case some model fails (this is primarily a slurm issue)
        try:
            log.info(f"Testing {exp_dir.stem}.")
            global_config, diffusion, trainer = load_checkpoint(
                exp_dir, milestones[i] if milestones != 0 else 0
            )

            if primers is None:
                T = global_config.diffusion_config.T
                t = torch.tensor(list(range(0, T)), device=device).long()
                dataset = np.load(global_config.path_config.data_path, mmap_mode="r")
                primers = dataset[get_random_primer_idx(len(dataset), n_samples)]
                primers = [torch.from_numpy(p) for p in primers]
                primers = [
                    rearrange(p, "k frames -> frames 1 k").to(device) for p in primers
                ]
                primers = torch.cat(primers, dim=1)
                primers = diffusion.q_sample(
                    primers[:T, ...], t, torch.randn_like(primers[:T, ...])
                ).float()
            if mode == "unconditional":
                generated_motions = progressive_generation(
                    diffusion, primers, t, sample_len
                )
                for i in range(n_samples):
                    output[f"motion{i+1}"] = generated_motions[..., i : i + 1, :]
            else:
                raise ValueError(f"{mode} is not supported")
            # Save the results
            save_dir = (
                Path("../out")
                / (massive_dir.stem if massive_dir is not None else "")
                / trainer.exp_name
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            file_num = get_file_num(save_dir, mode)
            print(len(output.keys()))
            saved_path = save_dir / f"{mode}{file_num}.pt"
            torch.save(output, saved_path)

            # Convert to bvhs
            ms = saved_path.__repr__().split("/")
            bvh_save_path = (
                Path("../out/bvhs")
                / "/".join(ms[ms.index("out") + 1 : -1])
                / ms[-1].split(".")[0]
            )
            bvh_save_path.mkdir(parents=True, exist_ok=True)
            gen = torch.load(saved_path, map_location="cpu")
            for key, value in gen.items():
                print(key, value.shape)
                if (bvh_save_path / f"{key}.bvh").exists():
                    raise KeyError("BVH files already exist")
                else:
                    test_to_bvh(value, bvh_save_path, f"{key}.bvh", dataset=500 if "500" in str(bvh_save_path) else 128)
        except Exception as e:
            traceback.print_exc()
            log.error(e)
            continue


def progressive_generation(
    diffusion: Diffusion, condition: torch.Tensor, t: torch.Tensor, length: int
) -> Tensor:
    """
    Generates animation progressively with "parallel" diffusion

    * diffusion: diffusion model
    * condition: input tensor, should contain clean frame and expectation of future frames
    * t: time step conditioning
    * length: iterations to run
    """

    # NOTE: Assuming condition only has one clean frame at the start
    result = []
    diffusion.eval()
    diffusion.denoiser.eval()
    with torch.no_grad():
        for i in tqdm(range(0, length), desc="generation loop time step", total=length):
            model_out = diffusion.p_sample(condition, t)
            result.append(model_out[:1, ...])
            # Concat Gaussian noise to last step in conditioning
            condition = torch.concat(
                (
                    model_out[1:, ...],
                    torch.randn_like(model_out[:1, ...], device=device),
                ),
                dim=0,
            )
    result = torch.concat(result, dim=0)

    #! For fuzzy visualization
    # result = torch.cat((result, condition), dim=0)

    return result


if __name__ == "__main__":
    main()
