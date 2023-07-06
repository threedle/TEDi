import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from util import extract, parse_bvh_file, torch_forward_kinematics, load_config

log = logging.getLogger(__name__)


class GeometricLoss:
    def __init__(self, path: Path, device, detach_labels=False) -> None:
        standard_bvh = list(path.parent.glob("*.bvh"))
        mean_file = list(path.parent.glob("*mean.npy"))
        std_file = list(path.parent.glob("*std.npy"))
        if not standard_bvh:
            raise FileNotFoundError(f"There are no standard bvh file in '{path}'.")
        if not mean_file:
            raise FileNotFoundError(f"There are no saved mean in '{path}'.")
        if not std_file:
            raise FileNotFoundError(f"There are no saved std in '{path}'.")

        # TODO: absorb the following into data_config
        joints, fps, n = parse_bvh_file(standard_bvh[0])
        self.offsets = torch.tensor(
            [joint.offsets for joint in joints if not joint.end_joint]
        )
        self.offsets = self.offsets.unsqueeze(0).to(device)
        self.hierarchy = [0] * len([joint for joint in joints if not joint.end_joint])
        for joint in joints:
            if joint.end_joint:
                continue
            if joint.parent is None:
                self.hierarchy[joint.idx] = -1
            else:
                self.hierarchy[joint.idx] = joint.parent.idx

        log.debug(f"Skeleton hierarchy is {self.hierarchy}")

        self.mean = torch.from_numpy(np.load(mean_file[0])).float().to(device)
        self.std = torch.from_numpy(np.load(std_file[0])).float().to(device)

        # Load data config
        data_config = load_config(path.parent)
        self.rots_idx = data_config["rots_idx"]
        self.contact_joints_idx = data_config["contact_joints_idx"]
        log.debug(f"Rotation data indices are {self.rots_idx}")
        log.debug(f"Contact label joints indices are {self.contact_joints_idx}")

        self.detach_labels = detach_labels
        log.debug(f"Detaching labels is {self.detach_labels}")

    def fk_loss(self, model_out, target, t, loss_weight):
        l, b, *_ = model_out.shape
        fk_pos = torch_forward_kinematics(
            rearrange(
                model_out[..., self.rots_idx], "l b (j k) -> b l j k", k=6
            ),  # reshaped 6D rotations
            offsets=self.offsets,  # offsets of skeleton
            root_pos=torch.zeros(b, l, 3, device=model_out.device),
            hierarchy=self.hierarchy,
            world=True,
        )
        target_fk_pos = torch_forward_kinematics(
            rearrange(
                target[..., self.rots_idx], "l b (j k) -> b l j k", k=6
            ),  # reshaped 6D rotations
            offsets=self.offsets,  # offsets of skeleton
            root_pos=torch.zeros(b, l, 3, device=model_out.device),
            hierarchy=self.hierarchy,
            world=True,
        )
        log.debug(f"{fk_pos.shape=}, {target_fk_pos.shape=}")
        fk_pos_loss = F.mse_loss(
            rearrange(fk_pos, "b l j k -> l b (j k)"),
            rearrange(target_fk_pos, "b l j k -> l b (j k)"),
            reduction="none",
        )
        fk_pos_loss = reduce(fk_pos_loss, "l ... -> l (...)", "mean")
        fk_pos_loss = fk_pos_loss * extract(loss_weight, t, fk_pos_loss.shape)
        return fk_pos_loss

    def ik_loss(self, model_out, t, loss_weight):
        contact = model_out[..., -4:]
        contact = rearrange(contact, "l b k -> l k b")
        root_pos = rearrange(model_out[..., :3], "l b k -> b l k")
        root_pos = root_pos * self.std[None, None, :3] + self.mean[None, None, :3]
        root_pos_y = root_pos[..., 2:3]
        root_pos = root_pos.cumsum(dim=1)
        root_pos = torch.cat(
            (root_pos[..., 0:1], root_pos_y, root_pos[..., 1:2]), dim=2
        )
        rots = model_out[..., self.rots_idx]
        rots = rearrange(rots, "l b (j k) -> b l j k", k=6)
        fk_pos = torch_forward_kinematics(
            rots, self.offsets, root_pos, self.hierarchy, world=True
        )

        fc_loc = torch.tensor(self.contact_joints_idx, device=model_out.device)
        model_fc = rearrange(
            torch.index_select(fk_pos, 2, fc_loc), "b l j k -> l b (j k)"
        )

        model_fc_label = model_out[..., -4:]
        if self.detach_labels:
            model_fc_label = model_fc_label.detach()

        model_fc_label = torch.sigmoid((model_fc_label - 0.5) * 12)

        # Velocity loss on foot joints
        velo = (model_fc[1:, ...] - model_fc[:-1, ...]).chunk(4, 2)
        velo = [
            torch.mul(v, model_fc_label[:-1, :, i].unsqueeze(2))
            for i, v in enumerate(velo)
        ]

        velo = torch.cat(velo, dim=2)

        velo_loss = F.mse_loss(
            velo, torch.zeros_like(velo, device=velo.device), reduction="none"
        )
        velo_loss = reduce(velo_loss, "l ... -> l (...)", "mean")
        velo_loss = velo_loss * extract(loss_weight, t[:-1], velo_loss.shape)
        return velo_loss
