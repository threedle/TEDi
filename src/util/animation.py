"""
Deals with all animation related stuff
"""
import re
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from matplotlib.animation import FuncAnimation
from torch import Tensor
from transforms3d.affines import compose
from transforms3d.euler import mat2euler
from transforms3d.taitbryan import euler2mat


class Joint:
    def __init__(self, name: str, parent, idx, end_joint=False) -> None:
        # a joint object has name, parents, num_params (3 or 6 or 0 for end joints)
        # 2d array of size frames x num_params, 3vector offsets, index in joint list and euler order
        self.name = name
        self.parent = parent
        self.num_params = 0
        self.params = []
        self.offsets = []
        self.end_joint = end_joint
        self.idx = idx
        self.rot_ord = ""

    # given frame number get rotation matrix
    def get_rotation_matrix(self, f: int, rotation_matrices: list[np.ndarray]):
        local = self.get_local_rotation_matrix(f)
        if self.parent is not None:
            if rotation_matrices[self.parent.idx] is None:
                rotation_matrices[self.idx] = (
                    self.parent.get_rotation_matrix(f, rotation_matrices) @ local
                )
            else:
                rotation_matrices[self.idx] = rotation_matrices[self.parent.idx] @ local
        else:
            rotation_matrices[self.idx] = local
        return rotation_matrices[self.idx].round(3)

    # this is the total transformation
    def get_local_rotation_matrix(self, f: int, no_translation=False):
        d = {
            "x": compose(
                [0, 0, 0],
                euler2mat(0, 0, np.radians(self.params[f][self.rot_ord.find("x") - 3])),
                [1, 1, 1],
            ),
            "y": compose(
                [0, 0, 0],
                euler2mat(0, np.radians(self.params[f][self.rot_ord.find("y") - 3]), 0),
                [1, 1, 1],
            ),
            "z": compose(
                [0, 0, 0],
                euler2mat(np.radians(self.params[f][self.rot_ord.find("z") - 3]), 0, 0),
                [1, 1, 1],
            ),
        }
        rotation_matrix = d[self.rot_ord[0]] @ d[self.rot_ord[1]] @ d[self.rot_ord[2]]
        return (
            rotation_matrix
            if no_translation
            else self.get_local_translation_matrix(f) @ rotation_matrix
        )

    # translation matrix
    def get_local_translation_matrix(self, f: int):
        if self.parent is None:
            offsets = np.array(self.params[f][:3]) + self.offsets
            return compose(offsets, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [1, 1, 1])
        return compose(self.offsets, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [1, 1, 1])


def plot(
    xyz,
    name,
    joints,
    fps,
    n,
    contact=[],
    save=False,
    show=True,
    save_folder: Path = None,
):
    # TODO: change this
    LEFT_FOOT_IDX = 38
    RIGHT_FOOT_IDX = 42
    # LEFT_FOOT_IDX = 5
    # RIGHT_FOOT_IDX = 10
    
    assert (
        len(xyz.shape) == 3 and xyz.shape[2] == 3
    ), f"input must have shape (frames, joints, 3), given {xyz.shape}"

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    title = ax.set_title("")
    inf = np.min(xyz)
    sup = np.max(xyz)
    ax.set_xlim(inf, sup)
    ax.set_ylim(inf, sup)
    ax.set_zlim(inf, sup)

    lines = [
        ax.plot([], [], [], "r.-", lw=1)[0]
        for joint in joints
        if not joint.end_joint and joint.name != joints[0].name
    ]
    contact_joints = [
        ax.plot([], [], [], "b.", lw=1)[0],
        ax.plot([], [], [], "b.", lw=1)[0],
    ]

    lines.extend(contact_joints)

    def func(i, lines, title):
        title.set_text(f"ANIMATION \n{name} \nframe={i}/{n}")
        x, y, z = map(lambda a: a.flatten(), np.split(xyz[i], 3, axis=1))
        for joint in joints:
            if joint.end_joint or joint.name == joints[0].name:
                continue
            lines[joint.idx - 1].set_xdata(
                np.array([x[joint.idx], x[joint.parent.idx]])
            )
            lines[joint.idx - 1].set_ydata(
                np.array([z[joint.idx], z[joint.parent.idx]])
            )
            lines[joint.idx - 1].set_3d_properties(
                np.array([y[joint.idx], y[joint.parent.idx]])
            )

        if len(contact) > 0:
            if contact[i, 1, 0] > 0.8:
                lines[-2].set_xdata(np.array([x[LEFT_FOOT_IDX], x[LEFT_FOOT_IDX]]))
                lines[-2].set_ydata(np.array([z[LEFT_FOOT_IDX], z[LEFT_FOOT_IDX]]))
                lines[-2].set_3d_properties(np.array([y[LEFT_FOOT_IDX], y[LEFT_FOOT_IDX]]))
            else:
                lines[-2].set_xdata([])
                lines[-2].set_ydata([])
                lines[-2].set_3d_properties([])
            if contact[i, 3, 0] > 0.8:
                lines[-1].set_xdata(np.array([x[RIGHT_FOOT_IDX], x[RIGHT_FOOT_IDX]]))
                lines[-1].set_ydata(np.array([z[RIGHT_FOOT_IDX], z[RIGHT_FOOT_IDX]]))
                lines[-1].set_3d_properties(np.array([y[RIGHT_FOOT_IDX], y[RIGHT_FOOT_IDX]]))
            else:
                lines[-1].set_xdata([])
                lines[-1].set_ydata([])
                lines[-1].set_3d_properties([])
        return title, lines

    anim = FuncAnimation(
        fig, func, n, fargs=(lines, title), interval=min(41.67, 1000 / fps), blit=False
    )
    if save:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1200)
        # anim.save(f'./gifs/{name}.gif', writer='imagemagick', fps=30)
        save_folder = (
            save_folder / name if save_folder is not None else f"../out/gifs/{name}"
        )
        anim.save(f"{save_folder}.mp4", writer=writer)
    if show:
        plt.show()

    anim._stop()


def parse_bvh_file(path: str) -> tuple[list[Joint], float, int]:
    # define Scanner object
    scanner = re.Scanner(
        [
            (
                r"([a-zA-Z]+[0-9]*_*[a-zA-Z]*[0-9]*\:?)+",
                lambda scanner, token: ("str", token),
            ),
            (r"{", lambda scanner, token: ("open_brace", token)),
            (r"}", lambda scanner, token: ("close_brace", token)),
            (
                r"-*[0-9]*(\.*[0-9]*e*-*[0-9]*)?[0-9]",
                lambda scanner, token: ("num", token),
            ),
            (r"\s+", None),
        ]
    )

    # tokenizer
    def tokenizer(scanner: re.Scanner, string):
        tokens, _ = scanner.scan(string)
        for token in tokens:
            yield token

    joints = []
    curr_joint = None
    parent_joint = None
    file = open(path)
    lines = file.read()
    file.close()
    lines = lines.split("MOTION")
    tokens = tokenizer(scanner, lines[0])
    idx = 0
    # parse hierarchy
    for identifier, match in tokens:
        if identifier == "str":
            if match == "ROOT" or match == "JOINT":
                curr_joint = Joint(next(tokens)[1], parent_joint, idx)
                idx += 1
                joints.append(curr_joint)
            elif match == "End":
                curr_joint = Joint(parent_joint.name + "_end", parent_joint, -1, True)
                joints.append(curr_joint)
            elif match == "OFFSET":
                for _ in range(0, 3):
                    curr_joint.offsets.append(float(next(tokens)[1]))
            elif match == "CHANNELS":
                curr_joint.num_params = n = int(next(tokens)[1])
                for _ in range(0, n - 3):
                    next(tokens)
                for i in range(0, 3):
                    s = next(tokens)[1]
                    if s == "Xrotation":
                        curr_joint.rot_ord += "x"
                    elif s == "Yrotation":
                        curr_joint.rot_ord += "y"
                    elif s == "Zrotation":
                        curr_joint.rot_ord += "z"
                    else:
                        raise RuntimeError(f"unconventional rotation: {s}")
            else:
                continue
        if identifier == "open_brace":
            parent_joint = curr_joint
        if identifier == "close_brace":
            parent_joint = parent_joint.parent
    # parse motion
    tokens = tokenizer(scanner, lines[1])
    num_frames = None
    fps = None
    motion_frames = []
    cnt = 0
    for identifier, match in tokens:
        if identifier == "str":
            if match == "Frames:":
                num_frames = int(next(tokens)[1])
            elif match == "Frame":
                next(tokens)
                fps = 1 / float(next(tokens)[1])
        else:
            motion_frames.append(float(match))
    motion_frames = np.array(motion_frames).reshape(num_frames, -1)
    for frame in range(0, num_frames):
        cnt = 0
        for joint in joints:
            if "end" in joint.name:
                continue
            params = []
            for _ in range(0, joint.num_params):
                params.append(motion_frames[frame][cnt])
                cnt += 1
            joint.params.append(params)
    return joints, round(fps, 2), num_frames


def get_raw_motion_data(joints):
    """Returns raw motion data matrix in euler"""
    motion = []
    frames = len(joints[0].params)
    for f in range(frames):
        motion_one_frame = []
        for joint in joints:
            if joint.end_joint:
                continue
            motion_one_frame.extend(joint.params[f])
        motion.append(motion_one_frame)
    return np.array(motion)


def get_xyz_motion(joints: list[Joint]) -> np.ndarray:
    motion = []
    frames = len(joints[0].params)
    non_end_joints = len([None for joint in joints if not joint.end_joint])
    for f in range(0, frames):
        motion_one_frame = []
        rotation_matrices = [None] * non_end_joints
        for joint in joints:
            if joint.end_joint:
                continue
            motion_one_frame.append(
                (joint.get_rotation_matrix(f, rotation_matrices) @ [0, 0, 0, 1])[:-1]
            )
        motion.append(motion_one_frame)
    return np.array(motion)


def get_6d_motion(joints: list[Joint]) -> np.ndarray:
    """
    Generate 6D motion representation given list of Joint objects.
    """
    motion = []
    frames = len(joints[0].params)
    non_end_joints = [joint for joint in joints if not joint.end_joint]
    for joint in non_end_joints:
        joint_6d = [
            joint.get_local_rotation_matrix(f, True)[:3, :2].flatten("F")
            for f in range(frames)
        ]
        motion.append(joint_6d)
    motion = rearrange(np.array(motion), "j f k -> f j k")
    print(f"{motion.shape=}")
    return motion

def get_foot_contact(
    xyz: np.ndarray, joints_idx: list[int] = None, eps=0.018
) -> np.ndarray:
    assert (
        len(xyz.shape) == 3 and xyz.shape[2] == 3
    ), f"input must have shape (frames, joints, 3), given {xyz.shape}"
    if joints_idx is None:
        joints_idx = list(range(0, xyz.shape[1]))
    velocities = np.linalg.norm(
        xyz[1:, joints_idx, :] - xyz[:-1, joints_idx, :], axis=2, keepdims=True
    )
    velocities = velocities < eps
    velocities = np.append(velocities, [velocities[-1, ...]], axis=0)
    return velocities.astype("float")


def sixd2mat(vectors):
    a_1, a_2 = vectors[0], vectors[1]
    b_1 = a_1 / np.linalg.norm(a_1)
    b_2 = a_2 - np.dot(b_1, a_2) * b_1
    b_2 = b_2 / np.linalg.norm(b_2)
    b_3 = np.cross(b_1, b_2)
    return np.transpose(np.vstack((b_1, b_2, b_3)))


def test_to_bvh(motion: Tensor, path: Path, name: str, dataset: int = 500) -> None:
    """
    Convert test data to bvh
    """
    s1 = slice(0, 3)
    s2 = slice(93, -4)

    # Check path validity
    save_path = path / name
    assert not save_path.exists(), f"{save_path} already exists."
    # Parse a stereotype bvh file
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed" / "high"
    joints, *_ = parse_bvh_file(
        data_dir / "01_01.bvh"
    )
    non_end_joints = [joint for joint in joints if not joint.end_joint]
    # Take the firt half of a stereotype bvh file
    source_bvh = open(
          data_dir / "01_01.bvh",
        "rt",
    )
    source_data = source_bvh.read()
    source_bvh.close()
    source_data = source_data.split("MOTION")[0]
    # Create new bvh
    bvh = open(save_path, "w")
    # Print first half to the new bvh
    print(source_data, file=bvh)
    # Save frame number and fps = 30
    print(f"MOTION\nFrames: {len(motion)}\nFrame Time: 0.03333", file=bvh)
    # Load mean and std for motion

    mean = np.load(
        list(data_dir.glob("*mean.npy"))[0]
    )
    std = np.load(
        list(data_dir.glob("*std.npy"))[0]
    )

    # Get root motion
    root_pos = rearrange(motion[..., s1], "l 1 k -> l k")
    root_pos_y = root_pos[..., 2:3].clone()
    root_pos = root_pos * std[None, s1] + mean[None, s1]
    root_pos = root_pos.cumsum(axis=0).numpy()
    root_pos = np.concatenate((root_pos[..., 0:1], root_pos_y, root_pos[..., 1:2]), axis=1)
    motion = rearrange(motion[..., s2], "l 1 (j k) -> 1 l j k", k=6)
    
    # Convert 6D to rotation matrices
    motion = torch_sixd2mat(motion)[0]
    motion = rearrange(motion, "l j k m -> j l k m", k=3, m=3)
    for joint in non_end_joints:
        joint_motion = motion[joint.idx]
        params = (
            np.array(
                [
                    mat2euler(mat.numpy(), axes="r" + joint.rot_ord)   
                    for mat in joint_motion
                ]
            )
            / np.pi
            * 180
        )
        if joint.num_params == 3:
            joint.params = params
        elif joint.num_params == 6:
            joint.params = [[0] * 6 for _ in range(len(joint_motion))]
            for f in range(len(joint.params)):
                joint.params[f][:3] = root_pos[f, :]
                joint.params[f][3:] = params[f]
        else:
            raise ValueError(f"Unconventional joint has {joint.num_params} parameters")
    motion = get_raw_motion_data(non_end_joints)
    np.set_printoptions(suppress=True)
    for one_frame in motion:
        print(
            " ".join(
                np.array_str(np.nan_to_num(one_frame), max_line_width=np.inf).strip("[]").split("  ")
            ),
            file=bvh,
        )
    bvh.close()


def torch_forward_kinematics(
    rotations: Tensor,
    offsets: Tensor,
    root_pos: Tensor,
    hierarchy: list[int],
    world: bool = True,
) -> Tensor:
    """Return bone transformations given rotations, assume a valid hierarchy,
    implemented in torch for use in training loss.
    For 6D:
        rotations: (batch_size, frames, joint, 6)
        offsets: (batch_size, joints, 3)
        root_pos: (batch_size, frames, 3)
    Returns:
        result: (batch_size, frames, joints, 3)
    """

    # Switch to matrices first
    rotations = torch_sixd2mat(rotations)
    offsets = offsets.reshape((-1, 1, offsets.shape[-2], offsets.shape[-1], 1))
    result = torch.empty(rotations.shape[:-2] + (3,), device=rotations.device)

    result[..., 0, :] = root_pos
    # Calculate positions
    for i, j in enumerate(hierarchy):
        if j == -1:
            assert i == 0
            continue
        result[..., i, :] = torch.matmul(
            rotations[..., j, :, :], offsets[..., i, :, :]
        ).squeeze()
        rotations[..., i, :, :] = torch.matmul(
            rotations[..., j, :, :].clone(), rotations[..., i, :, :].clone()
        )
        if world:
            result[..., i, :] += result[..., j, :]

    return result


def torch_sixd2mat(vec: Tensor) -> Tensor:
    """Return rotation matrices given 6D representations,
    implemented in torch for use in training loss

    Input:
        vec: Rotation, should have dimension (batch_size, frames, joints, 6)
    Returns:
        Ouput has dimension (batch, frames, joints, 3, 3)
    """
    a_1, a_2 = vec[..., :3], vec[..., 3:]
    a_1 = a_1 / a_1.norm(dim=-1, keepdim=True)
    a_3 = torch.cross(a_1, a_2)
    a_3 = a_3 / a_3.norm(dim=-1, keepdim=True)
    a_2 = torch.cross(a_3, a_1)
    mat = torch.transpose(
        torch.cat([a.unsqueeze(-2) for a in [a_1, a_2, a_3]], dim=-2), -1, -2
    )
    return mat
