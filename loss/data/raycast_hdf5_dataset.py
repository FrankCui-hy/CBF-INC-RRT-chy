from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class RaycastHDF5Dataset(Dataset):
    """Whole-state HDF5 dataset for all-rays-at-once raycast supervision."""

    def __init__(self, hdf5_path: Union[str, Path], metadata_path: Union[str, Path]) -> None:
        self.hdf5_path = Path(hdf5_path)
        self.metadata_path = Path(metadata_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata: Dict[str, Any] = json.load(f)
        with h5py.File(self.hdf5_path, "r") as f:
            self.length = int(f["q_ego"].shape[0])
            self.num_rays = int(f["hit_mask"].shape[1])
        expected_rays = self.metadata.get("num_rays_total")
        if expected_rays is not None and int(expected_rays) != self.num_rays:
            raise ValueError(
                f"HDF5 ray count {self.num_rays} does not match metadata num_rays_total {int(expected_rays)}"
            )
        self._h5: Optional[h5py.File] = None

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f = self._file()
        return {
            "q_ego": torch.from_numpy(f["q_ego"][idx].astype(np.float32)),
            "q_obs": torch.from_numpy(f["q_obs"][idx].astype(np.float32)),
            "hit_mask": torch.from_numpy(f["hit_mask"][idx].astype(np.float32)),
            "depth_norm": torch.from_numpy(f["depth_norm"][idx].astype(np.float32)),
            "sample_type": torch.tensor(int(f["sample_type"][idx]), dtype=torch.long),
        }

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self) -> None:
        self.close()


def load_raycast_metadata(dataset_dir: Union[str, Path]) -> Dict[str, Any]:
    metadata_path = Path(dataset_dir) / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_path(dataset_dir: Union[str, Path], split: str) -> Path:
    return Path(dataset_dir) / f"{split}.hdf5"


def compute_hit_pos_weight(hdf5_path: Union[str, Path], chunk_size: int = 4096) -> float:
    """Compute BCE positive-class weight from a split without materializing all rays."""
    pos = 0
    total = 0
    with h5py.File(hdf5_path, "r") as f:
        masks = f["hit_mask"]
        n = int(masks.shape[0])
        for start in range(0, n, int(chunk_size)):
            batch = masks[start : start + int(chunk_size)]
            pos += int(np.asarray(batch).sum())
            total += int(np.asarray(batch).size)
    neg = total - pos
    return float(neg / max(pos, 1))
