from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator

import torch

from loss.models.g_phi import SurrogateObservationNet
from loss.utils.config import load_config


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    p = ensure_parent(path)
    torch.save(payload, p)


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def load_obs_dataset(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    required = ["q_ego", "q_obs", "p_gt", "n_gt", "m"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError(f"Dataset missing keys: {missing}")
    return payload


def _strip_prefix_state_dict(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def load_g_phi_checkpoint(
    ckpt_path: str | Path,
    device: torch.device,
    config_path: str | Path | None = None,
) -> SurrogateObservationNet:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        raise ValueError("Unsupported checkpoint format. Expect dict with 'model' or 'state_dict'.")

    state_dict = _strip_prefix_state_dict(state_dict, "module.")

    model_cfg = ckpt.get("model_cfg", None) if isinstance(ckpt, dict) else None
    system_cfg = ckpt.get("system_cfg", None) if isinstance(ckpt, dict) else None

    if model_cfg is None or system_cfg is None:
        if config_path is None:
            raise ValueError("Checkpoint lacks model_cfg/system_cfg, please provide --config.")
        cfg = load_config(config_path)
        model_cfg = cfg["model"]["g_phi"]
        system_cfg = cfg["system"]

    model = SurrogateObservationNet(
        n_ego=int(system_cfg["n_ego"]),
        n_obs=int(system_cfg["n_obs"]),
        rays=int(system_cfg["rays"]),
        hidden_dims=list(model_cfg["hidden_dims"]),
        activation=str(model_cfg.get("activation", "silu")),
        predict_hit_prob=bool(model_cfg.get("predict_hit_prob", True)),
        norm_eps=float(model_cfg.get("norm_eps", 1e-6)),
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def iter_batches(
    payload: dict[str, Any],
    batch_size: int,
    device: torch.device,
    max_batches: int | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    N = int(payload["q_ego"].shape[0])
    keys = ["q_ego", "q_obs", "p_gt", "n_gt", "m"]
    if "qdot_ego" in payload:
        keys.append("qdot_ego")
    if "qdot_obs" in payload:
        keys.append("qdot_obs")

    n_batch = (N + batch_size - 1) // batch_size
    if max_batches is not None and max_batches > 0:
        n_batch = min(n_batch, max_batches)

    for b in range(n_batch):
        s = b * batch_size
        e = min(N, (b + 1) * batch_size)
        out = {}
        for k in keys:
            out[k] = payload[k][s:e].to(device).float()
        yield out


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    p = ensure_parent(path)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = ensure_parent(path)
    if len(rows) == 0:
        with open(p, "w", encoding="utf-8") as f:
            f.write("")
        return
    fields = list(rows[0].keys())
    with open(p, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
