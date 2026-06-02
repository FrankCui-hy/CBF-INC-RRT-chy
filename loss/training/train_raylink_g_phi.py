from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loss.data.raycast_hdf5_dataset import RaycastHDF5Dataset, compute_hit_pos_weight, split_path
from loss.metrics.raylink_metrics import raylink_loss, select_safety_threshold, summarize_predictions
from loss.models.raylink_g_phi import RayLinkMLPGPhi
from loss.utils.config import load_config
from loss.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FK-aware ray-link MLP g_phi.")
    p.add_argument("--config", default="loss/configs/config_raylink_g_phi.yaml")
    p.add_argument("--dataset_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


def make_loaders(cfg: Dict, dataset_dir: Path) -> tuple[DataLoader, DataLoader, Dict]:
    metadata_path = dataset_dir / "metadata.json"
    train_ds = RaycastHDF5Dataset(split_path(dataset_dir, "train"), metadata_path)
    val_ds = RaycastHDF5Dataset(split_path(dataset_dir, "val"), metadata_path)
    batch_size = int(cfg["train"]["batch_size"])
    workers = int(cfg["data"].get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, train_ds.metadata


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def run_train_epoch(model, loader, optimizer, scaler, device, cfg, pos_weight):
    model.train()
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    totals = {"total": 0.0, "hit": 0.0, "depth": 0.0}
    count = 0
    for batch in loader:
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            pred = model(batch["q_ego"], batch["q_obs"])
            losses = raylink_loss(
                pred["hit_logits"],
                pred["depth_norm"],
                batch["hit_mask"],
                batch["depth_norm"],
                pos_weight,
                float(cfg["loss"]["lambda_hit"]),
                float(cfg["loss"]["lambda_depth"]),
            )
        scaler.scale(losses["total"]).backward()
        grad_clip = float(cfg["train"].get("grad_clip_norm", 0.0))
        if grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = int(batch["q_ego"].shape[0])
        count += bs
        for key in totals:
            totals[key] += float(losses[key].detach().item()) * bs
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def collect_predictions(model, loader, device, cfg, pos_weight):
    model.eval()
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    totals = {"total": 0.0, "hit": 0.0, "depth": 0.0}
    count = 0
    all_hit_logits: List[torch.Tensor] = []
    all_depth_pred: List[torch.Tensor] = []
    all_hit_mask: List[torch.Tensor] = []
    all_depth_true: List[torch.Tensor] = []
    all_sample_type: List[torch.Tensor] = []
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(enabled=amp_enabled):
            pred = model(batch["q_ego"], batch["q_obs"])
            losses = raylink_loss(
                pred["hit_logits"],
                pred["depth_norm"],
                batch["hit_mask"],
                batch["depth_norm"],
                pos_weight,
                float(cfg["loss"]["lambda_hit"]),
                float(cfg["loss"]["lambda_depth"]),
            )
        bs = int(batch["q_ego"].shape[0])
        count += bs
        for key in totals:
            totals[key] += float(losses[key].detach().item()) * bs
        all_hit_logits.append(pred["hit_logits"].detach().float().cpu())
        all_depth_pred.append(pred["depth_norm"].detach().float().cpu())
        all_hit_mask.append(batch["hit_mask"].detach().float().cpu())
        all_depth_true.append(batch["depth_norm"].detach().float().cpu())
        all_sample_type.append(batch["sample_type"].detach().long().cpu())
    losses_avg = {key: value / max(count, 1) for key, value in totals.items()}
    return losses_avg, {
        "hit_logits": torch.cat(all_hit_logits, dim=0),
        "depth_pred": torch.cat(all_depth_pred, dim=0),
        "hit_mask": torch.cat(all_hit_mask, dim=0),
        "depth_true": torch.cat(all_depth_true, dim=0),
        "sample_type": torch.cat(all_sample_type, dim=0),
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, cfg: Dict, metadata: Dict, threshold: float, metrics: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "config": cfg,
            "metadata": metadata,
            "selected_hit_threshold": float(threshold),
            "val_metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.dataset_dir is not None:
        cfg["data"]["dataset_dir"] = args.dataset_dir
    if args.out_dir is not None:
        cfg["paths"]["out_dir"] = args.out_dir
    if args.device is not None:
        cfg["device"] = args.device

    set_seed(int(cfg.get("seed", 0)))
    use_cuda = str(cfg.get("device", "cuda")).startswith("cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dataset_dir = Path(cfg["data"]["dataset_dir"])
    out_dir = Path(cfg["paths"]["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    eval_dir = out_dir / "eval"
    log_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, metadata = make_loaders(cfg, dataset_dir)
    model_cfg = cfg["model"]
    model = RayLinkMLPGPhi(
        metadata,
        pair_hidden_dim=int(model_cfg.get("pair_hidden_dim", 128)),
        head_hidden_dims=list(model_cfg.get("head_hidden_dims", [128, 64])),
        link_embed_dim=int(model_cfg.get("link_embed_dim", 8)),
        anchor_embed_dim=int(model_cfg.get("anchor_embed_dim", 8)),
        activation=str(model_cfg.get("activation", "silu")),
        finger_open=float(model_cfg.get("finger_open", 0.04)),
    ).to(device)

    pos_weight_value = compute_hit_pos_weight(split_path(dataset_dir, "train"), int(cfg["data"].get("pos_weight_chunk_size", 4096)))
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg["train"].get("lr_factor", 0.5)),
        patience=int(cfg["train"].get("lr_patience", 5)),
    )
    scaler = GradScaler(enabled=bool(cfg["train"].get("amp", True)) and device.type == "cuda")

    thresholds = [float(x) for x in cfg["eval"].get("thresholds", [i / 100.0 for i in range(5, 100, 5)])]
    best_total = float("inf")
    best_depth = float("inf")
    best_safety = float("inf")
    patience = int(cfg["train"].get("early_stopping_patience", 15))
    stale = 0
    metrics_path = log_dir / "train_metrics.jsonl"

    for epoch in range(int(cfg["train"]["epochs"])):
        train_losses = run_train_epoch(model, train_loader, optimizer, scaler, device, cfg, pos_weight)
        val_losses, val_pred = collect_predictions(model, val_loader, device, cfg, pos_weight)
        selected_threshold, threshold_table = select_safety_threshold(
            val_pred["hit_logits"],
            val_pred["hit_mask"],
            val_pred["sample_type"],
            thresholds,
            near_recall_target=float(cfg["eval"].get("near_recall_target", 0.95)),
            collision_recall_target=float(cfg["eval"].get("collision_recall_target", 0.98)),
        )
        val_metrics = summarize_predictions(
            val_pred["hit_logits"],
            val_pred["depth_pred"],
            val_pred["hit_mask"],
            val_pred["depth_true"],
            val_pred["sample_type"],
            float(metadata["r_max"]),
            selected_threshold,
        )
        val_metrics.update({f"val_loss/{k}": v for k, v in val_losses.items()})
        val_metrics["selected_hit_threshold"] = float(selected_threshold)
        val_metrics["pos_weight"] = float(pos_weight_value)
        scheduler.step(val_losses["total"])

        near_fn = float(val_metrics.get("near_boundary/hit_false_negative_rate", 1.0))
        collision_fn = float(val_metrics.get("collision_unsafe/hit_false_negative_rate", 1.0))
        near_depth = float(val_metrics.get("near_boundary/depth_mae_hit_meter", 1.0))
        safety_score = collision_fn * 10.0 + near_fn * 5.0 + near_depth

        record = {
            "epoch": int(epoch),
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_metrics": val_metrics,
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        with open(eval_dir / "val_threshold_metrics.json", "w", encoding="utf-8") as f:
            json.dump(threshold_table, f, indent=2)

        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, epoch, cfg, metadata, selected_threshold, val_metrics)
        improved = False
        if val_losses["total"] < best_total:
            best_total = val_losses["total"]
            save_checkpoint(ckpt_dir / "best_total.pt", model, optimizer, epoch, cfg, metadata, selected_threshold, val_metrics)
            improved = True
        depth_metric = float(val_metrics.get("depth_mae_hit_meter", float("inf")))
        if depth_metric < best_depth:
            best_depth = depth_metric
            save_checkpoint(ckpt_dir / "best_depth.pt", model, optimizer, epoch, cfg, metadata, selected_threshold, val_metrics)
            improved = True
        if safety_score < best_safety:
            best_safety = safety_score
            save_checkpoint(ckpt_dir / "best_safety.pt", model, optimizer, epoch, cfg, metadata, selected_threshold, val_metrics)
            improved = True

        print(
            f"[raylink_g_phi][{epoch:03d}] "
            f"train total={train_losses['total']:.4f} hit={train_losses['hit']:.4f} depth={train_losses['depth']:.4f} | "
            f"val total={val_losses['total']:.4f} hit={val_losses['hit']:.4f} depth={val_losses['depth']:.4f} | "
            f"thr={selected_threshold:.2f} near_fn={near_fn:.4f} collision_fn={collision_fn:.4f}"
        )

        stale = 0 if improved else stale + 1
        if stale >= patience:
            print(f"[raylink_g_phi] early stopping at epoch={epoch}")
            break


if __name__ == "__main__":
    main()
