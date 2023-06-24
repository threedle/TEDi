""" 
Generate BVH files from model outputs
"""
import argparse
import traceback
from pathlib import Path

import torch
from util.animation import test_to_bvh

parser = argparse.ArgumentParser()
parser.add_argument("--motion_path", "-p", type=str, required=True)
args = parser.parse_args()
path = Path(args.motion_path)
assert path.exists()
motions = path.glob("**/*.pt")
for m in motions:
    ms = m.__repr__().split("/")
    print(ms[-1].split(".")[0])
    save_path = (
        Path("../out/bvhs")
        / "/".join(ms[ms.index("out") + 1 : -1])
        / ms[-1].split(".")[0]
    )
    save_path.mkdir(parents=True, exist_ok=True)
    gen = torch.load(m, map_location="cpu")
    for key, value in gen.items():
        try:
            print(key, value.shape)
            if (save_path / f"{key}.bvh").exists():
                print("skip.")
                continue
            else:
                test_to_bvh(value, save_path, f"{key}.bvh")
        except Exception as e:
            traceback.print_exc()
            break
