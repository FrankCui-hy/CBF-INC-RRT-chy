from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loss.data.raycast_hdf5_dataset import RaycastHDF5Dataset, split_path
from loss.metrics.raylink_metrics import select_safety_threshold, summarize_predictions
from loss.models.raylink_g_phi import RayLinkMLPGPhi
from loss.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate FK-aware ray-link MLP g_phi.")
    p.add_argument("--config", default="loss/configs/config_raylink_g_phi.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset_dir", default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out_dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None)
    return p.parse_args()


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def build_model(metadata: Dict, cfg: Dict, device: torch.device) -> RayLinkMLPGPhi:
    model_cfg = cfg.get("model", {})
    model = RayLinkMLPGPhi(
        metadata,
        pair_hidden_dim=int(model_cfg.get("pair_hidden_dim", 128)),
        head_hidden_dims=list(model_cfg.get("head_hidden_dims", [128, 64])),
        link_embed_dim=int(model_cfg.get("link_embed_dim", 8)),
        anchor_embed_dim=int(model_cfg.get("anchor_embed_dim", 8)),
        activation=str(model_cfg.get("activation", "silu")),
        finger_open=float(model_cfg.get("finger_open", 0.04)),
    )
    return model.to(device)


@torch.no_grad()
def collect_predictions(model, loader, device, amp_enabled: bool) -> Dict[str, torch.Tensor]:
    model.eval()
    hit_logits: List[torch.Tensor] = []
    depth_pred: List[torch.Tensor] = []
    hit_mask: List[torch.Tensor] = []
    depth_true: List[torch.Tensor] = []
    sample_type: List[torch.Tensor] = []
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(enabled=amp_enabled):
            pred = model(batch["q_ego"], batch["q_obs"])
        hit_logits.append(pred["hit_logits"].detach().float().cpu())
        depth_pred.append(pred["depth_norm"].detach().float().cpu())
        hit_mask.append(batch["hit_mask"].detach().float().cpu())
        depth_true.append(batch["depth_norm"].detach().float().cpu())
        sample_type.append(batch["sample_type"].detach().long().cpu())
    return {
        "hit_logits": torch.cat(hit_logits, dim=0),
        "depth_pred": torch.cat(depth_pred, dim=0),
        "hit_mask": torch.cat(hit_mask, dim=0),
        "depth_true": torch.cat(depth_true, dim=0),
        "sample_type": torch.cat(sample_type, dim=0),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "config" in ckpt:
        merged_cfg = ckpt["config"]
        merged_cfg.setdefault("eval", cfg.get("eval", {}))
        cfg = merged_cfg
    if args.dataset_dir is not None:
        cfg["data"]["dataset_dir"] = args.dataset_dir
    if args.device is not None:
        cfg["device"] = args.device
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = int(args.batch_size)

    use_cuda = str(cfg.get("device", "cuda")).startswith("cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dataset_dir = Path(cfg["data"]["dataset_dir"])
    metadata = ckpt.get("metadata")
    metadata_path = dataset_dir / "metadata.json"
    ds = RaycastHDF5Dataset(split_path(dataset_dir, args.split), metadata_path)
    metadata = metadata or ds.metadata

    model = build_model(metadata, cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])

    loader = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("data", {}).get("num_workers", 0)),
        pin_memory=True,
    )
    pred = collect_predictions(model, loader, device, bool(cfg["train"].get("amp", True)) and device.type == "cuda")

    if args.threshold is not None:
        threshold = float(args.threshold)
        threshold_table = {}
    elif "selected_hit_threshold" in ckpt:
        threshold = float(ckpt["selected_hit_threshold"])
        threshold_table = {}
    else:
        thresholds = [float(x) for x in cfg["eval"].get("thresholds", [i / 100.0 for i in range(5, 100, 5)])]
        threshold, threshold_table = select_safety_threshold(
            pred["hit_logits"],
            pred["hit_mask"],
            pred["sample_type"],
            thresholds,
            near_recall_target=float(cfg["eval"].get("near_recall_target", 0.95)),
            collision_recall_target=float(cfg["eval"].get("collision_recall_target", 0.98)),
        )

    metrics = summarize_predictions(
        pred["hit_logits"],
        pred["depth_pred"],
        pred["hit_mask"],
        pred["depth_true"],
        pred["sample_type"],
        float(metadata["r_max"]),
        float(threshold),
    )
    metrics["selected_hit_threshold"] = float(threshold)
    metrics["split"] = args.split
    metrics["num_samples"] = float(len(ds))

    out_dir = Path(args.out_dir or Path(cfg["paths"]["out_dir"]) / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.split}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if threshold_table:
        with open(out_dir / f"{args.split}_threshold_metrics.json", "w", encoding="utf-8") as f:
            json.dump(threshold_table, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
