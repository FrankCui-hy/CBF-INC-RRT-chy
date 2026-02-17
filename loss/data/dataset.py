from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    full_dataset: "RaycastEpisodeDataset"


class RaycastEpisodeDataset(Dataset):
    """Per-step dataset with shape-safe tensor fields.

    Required keys:
        q_ego: (N, n)
        qdot_ego: (N, n)
        q_obs: (N, m)
        qdot_obs: (N, m)
        p_gt: (N, R, 3)
        n_gt: (N, R, 3)
        m: (N, R)
        ray_origin: (N, R, 3)
        ray_dir: (N, R, 3)
        y: (N,) in {-1, +1}
    """

    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload
        self.length = int(payload["q_ego"].shape[0])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "q_ego": self.payload["q_ego"][idx].float(),
            "qdot_ego": self.payload["qdot_ego"][idx].float(),
            "q_obs": self.payload["q_obs"][idx].float(),
            "qdot_obs": self.payload["qdot_obs"][idx].float(),
            "p_gt": self.payload["p_gt"][idx].float(),
            "n_gt": self.payload["n_gt"][idx].float(),
            "m": self.payload["m"][idx].float(),
            "ray_origin": self.payload["ray_origin"][idx].float(),
            "ray_dir": self.payload["ray_dir"][idx].float(),
            "y": self.payload["y"][idx].float(),
        }
        if "episode_type" in self.payload:
            item["episode_type"] = self.payload["episode_type"][idx].float()
        return item



def load_dataset(path: str) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    required = ["q_ego", "qdot_ego", "q_obs", "qdot_obs", "p_gt", "n_gt", "m", "ray_origin", "ray_dir", "y"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError(f"Missing keys in dataset: {missing}")
    return payload


def build_dataloaders(cfg: Dict[str, Any], batch_size: int | None = None) -> DatasetBundle:
    payload = load_dataset(cfg["paths"]["dataset_path"])
    ds = RaycastEpisodeDataset(payload)

    split = float(cfg["data"]["train_split"])
    n_train = int(len(ds) * split)
    n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(int(cfg["seed"])))

    if batch_size is None:
        batch_size = int(cfg["train"]["g_phi"]["batch_size"])
    workers = int(cfg["data"].get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    return DatasetBundle(train_loader=train_loader, val_loader=val_loader, full_dataset=ds)
