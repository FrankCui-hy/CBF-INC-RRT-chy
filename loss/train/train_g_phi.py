from __future__ import annotations

import argparse
from pathlib import Path

import torch

from loss.data.dataset import build_dataloaders
from loss.losses.obs_losses import compute_observation_losses
from loss.models.g_phi import SurrogateObservationNet
from loss.utils.config import load_config
from loss.utils.io import save_checkpoint
from loss.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train differentiable observation surrogate g_phi.")
    p.add_argument("--config", type=str, default="loss/configs/config.yaml")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def run_epoch(model, loader, optimizer, cfg_obs, device, train=True):
    model.train(train)
    total = {"total": 0.0, "L_p": 0.0, "L_n": 0.0, "L_m": 0.0, "L_ray": 0.0}
    count = 0

    for batch in loader:
        batch = to_device(batch, device)
        pred = model(batch["q_ego"], batch["q_obs"])
        losses = compute_observation_losses(pred, batch, cfg_obs)

        if train:
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()

        bs = batch["q_ego"].shape[0]
        count += bs
        for k in total.keys():
            total[k] += losses[k].item() * bs

    return {k: v / max(count, 1) for k, v in total.items()}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))

    use_cuda = cfg.get("device", "cuda").startswith("cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = int(cfg["train"]["g_phi"]["batch_size"])
    data_bundle = build_dataloaders(cfg, batch_size=batch_size)

    sys_cfg = cfg["system"]
    mcfg = cfg["model"]["g_phi"]
    model = SurrogateObservationNet(
        n_ego=int(sys_cfg["n_ego"]),
        n_obs=int(sys_cfg["n_obs"]),
        rays=int(sys_cfg["rays"]),
        hidden_dims=list(mcfg["hidden_dims"]),
        activation=str(mcfg.get("activation", "silu")),
        predict_hit_prob=bool(mcfg.get("predict_hit_prob", True)),
        norm_eps=float(mcfg.get("norm_eps", 1e-6)),
    ).to(device)

    tcfg = cfg["train"]["g_phi"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 1e-5)),
    )

    ckpt_path = cfg["paths"]["g_phi_ckpt"]
    start_epoch = 0
    best_val = float("inf")

    if args.resume or bool(tcfg.get("resume", False)):
        p = Path(ckpt_path)
        if p.exists():
            ckpt = torch.load(p, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val = float(ckpt.get("best_val", best_val))
            print(f"[train_g_phi] resumed from {p}, start_epoch={start_epoch}")

    epochs = int(tcfg["epochs"])
    cfg_obs = cfg["loss"]["obs"]

    for epoch in range(start_epoch, epochs):
        tr = run_epoch(model, data_bundle.train_loader, optimizer, cfg_obs, device, train=True)
        va = run_epoch(model, data_bundle.val_loader, optimizer, cfg_obs, device, train=False)

        print(
            f"[g_phi][{epoch:03d}] "
            f"train total={tr['total']:.4f} p={tr['L_p']:.4f} n={tr['L_n']:.4f} m={tr['L_m']:.4f} ray={tr['L_ray']:.4f} | "
            f"val total={va['total']:.4f} p={va['L_p']:.4f} n={va['L_n']:.4f} m={va['L_m']:.4f} ray={va['L_ray']:.4f}"
        )

        is_best = va["total"] < best_val
        if is_best:
            best_val = va["total"]

        save_checkpoint(
            ckpt_path,
            {
                "epoch": epoch,
                "best_val": best_val,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_cfg": mcfg,
                "system_cfg": sys_cfg,
            },
        )
        if is_best:
            save_checkpoint(str(Path(ckpt_path).with_name("g_phi_best.pt")), {"model": model.state_dict(), "model_cfg": mcfg, "system_cfg": sys_cfg})


if __name__ == "__main__":
    main()
