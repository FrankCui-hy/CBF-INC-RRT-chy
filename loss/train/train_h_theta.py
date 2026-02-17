from __future__ import annotations

import argparse
from pathlib import Path

import torch

from loss.data.dataset import build_dataloaders
from loss.losses.cbf_losses import compute_cbf_losses
from loss.models.g_phi import SurrogateObservationNet
from loss.models.h_theta import NeuralCBF
from loss.train.derivatives import composite_jvp_h_and_hdot, decomposed_h_and_hdot
from loss.utils.config import load_config
from loss.utils.io import save_checkpoint
from loss.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Neural CBF h_theta with analytic hdot via torch.func JVP/VJP.")
    p.add_argument("--config", type=str, default="loss/configs/config.yaml")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def build_g_phi_from_ckpt(cfg, device: torch.device):
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

    ckpt_path = Path(cfg["paths"]["g_phi_ckpt"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"g_phi checkpoint not found: {ckpt_path}. Run train_g_phi first.")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return model


def run_epoch(h_model, g_model, loader, optimizer, cbf_cfg, derivative_mode, device, train=True):
    h_model.train(train)
    g_model.eval()

    total = {"total": 0.0, "L_cbf": 0.0, "L_cls": 0.0}
    count = 0

    for batch in loader:
        batch = to_device(batch, device)
        qe = batch["q_ego"]
        qo = batch["q_obs"]
        qde = batch["qdot_ego"]
        qdo = batch["qdot_obs"]
        y = batch["y"]

        if derivative_mode == "composite_jvp":
            h, hdot = composite_jvp_h_and_hdot(g_model, h_model, qe, qo, qde, qdo)
        elif derivative_mode == "decomposed":
            h, hdot = decomposed_h_and_hdot(g_model, h_model, qe, qo, qde, qdo)
        else:
            raise ValueError(f"Unknown derivative_mode: {derivative_mode}")

        losses = compute_cbf_losses(h, hdot, y, cbf_cfg)

        if train:
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()

        bs = qe.shape[0]
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

    batch_size = int(cfg["train"]["h_theta"]["batch_size"])
    data_bundle = build_dataloaders(cfg, batch_size=batch_size)

    g_model = build_g_phi_from_ckpt(cfg, device)
    freeze_g = bool(cfg["train"]["h_theta"].get("freeze_g_phi", True))
    for p in g_model.parameters():
        p.requires_grad = not freeze_g

    hcfg = cfg["model"]["h_theta"]
    obs_dim_per_ray = 7 if bool(hcfg.get("include_hit_prob", True)) else 6
    h_model = NeuralCBF(
        n_ego=int(cfg["system"]["n_ego"]),
        rays=int(cfg["system"]["rays"]),
        obs_dim_per_ray=obs_dim_per_ray,
        encoder=str(hcfg["encoder"]),
        point_feat_dim=int(hcfg["point_feat_dim"]),
        global_feat_dim=int(hcfg["global_feat_dim"]),
        hidden_dims=list(hcfg["hidden_dims"]),
        activation=str(hcfg.get("activation", "silu")),
    ).to(device)

    tcfg = cfg["train"]["h_theta"]
    params = list(h_model.parameters()) + ([] if freeze_g else list(g_model.parameters()))
    optimizer = torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=float(tcfg.get("weight_decay", 1e-5)))

    ckpt_path = cfg["paths"]["h_theta_ckpt"]
    start_epoch = 0
    best_val = float("inf")
    if args.resume or bool(tcfg.get("resume", False)):
        p = Path(ckpt_path)
        if p.exists():
            ckpt = torch.load(p, map_location="cpu")
            h_model.load_state_dict(ckpt["h_model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val = float(ckpt.get("best_val", best_val))
            print(f"[train_h_theta] resumed from {p}, start_epoch={start_epoch}")

    epochs = int(tcfg["epochs"])
    derivative_mode = str(tcfg.get("derivative_mode", "composite_jvp"))

    for epoch in range(start_epoch, epochs):
        tr = run_epoch(h_model, g_model, data_bundle.train_loader, optimizer, cfg["loss"]["cbf"], derivative_mode, device, train=True)
        va = run_epoch(h_model, g_model, data_bundle.val_loader, optimizer, cfg["loss"]["cbf"], derivative_mode, device, train=False)

        print(
            f"[h_theta][{epoch:03d}] "
            f"train total={tr['total']:.4f} cbf={tr['L_cbf']:.4f} cls={tr['L_cls']:.4f} | "
            f"val total={va['total']:.4f} cbf={va['L_cbf']:.4f} cls={va['L_cls']:.4f}"
        )

        is_best = va["total"] < best_val
        if is_best:
            best_val = va["total"]

        save_checkpoint(
            ckpt_path,
            {
                "epoch": epoch,
                "best_val": best_val,
                "h_model": h_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "h_cfg": hcfg,
                "system_cfg": cfg["system"],
            },
        )
        if is_best:
            save_checkpoint(str(Path(ckpt_path).with_name("h_theta_best.pt")), {"h_model": h_model.state_dict(), "h_cfg": hcfg, "system_cfg": cfg["system"]})


if __name__ == "__main__":
    main()
