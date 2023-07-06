import logging

import numpy as np
import torch

log = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    """
    Generic dataset instance
    """

    def __init__(self, path):
        super().__init__()
        log.info(f"Loading data {path=}")
        self.data = np.load(path, mmap_mode="r")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx].copy())
