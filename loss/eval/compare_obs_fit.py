from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from loss.metrics.obs_metrics import compute_obs_metrics
from loss.utils.io import (
    iter_batches,
    load_g_phi_checkpoint,
    load_obs_dataset,
    resolve_device,
    save_csv,
    save_json,
)
from loss.viz.obs_viz import (
    plot_hist,
    plot_per_ray,
    plot_scatter_hit_points,
    plot_summary_metrics,
    plot_sweep_curve,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare fitted observation g_phi against true raycast observation.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to g_phi checkpoint")
    p.add_argument("--data", type=str, required=True, help="Path to dataset .pt")
    p.add_argument("--out_dir", type=str, required=True, help="Output folder")
    p.add_argument("--config", type=str, default="loss/configs/config.yaml")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--num_plot_samples", type=int, default=64)
    p.add_argument("--save_pdf", action="store_true")

    p.add_argument("--sweep_qego", action="store_true")
    p.add_argument("--joint_idx", type=int, default=3)
    p.add_argument("--num_points", type=int, default=50)
    p.add_argument("--sweep_ref_index", type=int, default=0)
    p.add_argument("--sweep_pool", type=int, default=512)
    p.add_argument("--sweep_qobs_weight", type=float, default=0.2)
    return p.parse_args()


def _single_sample_errors(p_gt, n_gt, m, p_pred, n_pred) -> tuple[float, float]:
    hit = m > 0.5
    if hit.sum() == 0:
        return float("nan"), float("nan")
    pos = (p_pred - p_gt).norm(dim=-1)
    rmse = torch.sqrt((pos[hit].pow(2)).mean()).item()
    n_gt = torch.nn.functional.normalize(n_gt, dim=-1, eps=1e-6)
    n_pred = torch.nn.functional.normalize(n_pred, dim=-1, eps=1e-6)
    cos = (n_gt * n_pred).sum(dim=-1).clamp(-1.0, 1.0)
    ang = torch.rad2deg(torch.acos(cos))[hit].mean().item()
    return float(rmse), float(ang)


def run_sweep(
    payload: dict[str, Any],
    model,
    device: torch.device,
    joint_idx: int,
    num_points: int,
    ref_index: int,
    sweep_pool: int,
    qobs_weight: float,
) -> list[dict[str, float]]:
    q_ego = payload["q_ego"].float()
    q_obs = payload["q_obs"].float()
    p_gt = payload["p_gt"].float()
    n_gt = payload["n_gt"].float()
    m = payload["m"].float()

    N, n = q_ego.shape
    assert 0 <= joint_idx < n
    ref_index = int(np.clip(ref_index, 0, N - 1))

    q_obs_ref = q_obs[ref_index]
    qobs_dist = (q_obs - q_obs_ref.unsqueeze(0)).norm(dim=-1)
    pool_k = min(int(sweep_pool), N)
    pool_idx = torch.topk(-qobs_dist, k=pool_k).indices

    q_min = q_ego[:, joint_idx].min().item()
    q_max = q_ego[:, joint_idx].max().item()
    targets = torch.linspace(q_min, q_max, steps=num_points)

    rows: list[dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for t in targets:
            cand_q = q_ego[pool_idx, joint_idx]
            cand_obs_d = qobs_dist[pool_idx]
            score = (cand_q - t).abs() + qobs_weight * cand_obs_d
            sel_local = torch.argmin(score)
            idx = int(pool_idx[sel_local].item())

            qe_i = q_ego[idx : idx + 1].to(device)
            qo_i = q_obs[idx : idx + 1].to(device)
            pred = model(qe_i, qo_i)
            p_pred_i = pred["p_hat"][0].cpu()
            n_pred_i = pred["n_hat"][0].cpu()

            rmse_i, ang_i = _single_sample_errors(
                p_gt[idx], n_gt[idx], m[idx].squeeze(-1) if m[idx].ndim == 2 else m[idx], p_pred_i, n_pred_i
            )
            rows.append(
                {
                    "target_q": float(t.item()),
                    "actual_q": float(q_ego[idx, joint_idx].item()),
                    "sample_idx": float(idx),
                    "qobs_l2_to_ref": float(qobs_dist[idx].item()),
                    "rmse_hit": float(rmse_i),
                    "angle_mean_deg_hit": float(ang_i),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    payload = load_obs_dataset(args.data)
    model = load_g_phi_checkpoint(args.ckpt, device=device, config_path=args.config)

    all_p_gt = []
    all_n_gt = []
    all_m = []
    all_p_pred = []
    all_n_pred = []
    all_m_pred = []

    max_batches = args.max_batches if args.max_batches > 0 else None

    model.eval()
    with torch.no_grad():
        for batch in iter_batches(payload, batch_size=args.batch_size, device=device, max_batches=max_batches):
            qe = batch["q_ego"]
            qo = batch["q_obs"]
            pred = model(qe, qo)

            p_pred = pred["p_hat"]
            n_pred = torch.nn.functional.normalize(pred["n_hat"], dim=-1, eps=1e-6)
            m_pred = pred.get("m_hat", None)

            p_gt = batch["p_gt"]
            n_gt = batch["n_gt"]
            m = batch["m"]

            assert p_gt.shape == p_pred.shape
            assert n_gt.shape == n_pred.shape
            assert m.shape[:2] == p_gt.shape[:2]

            all_p_gt.append(p_gt.cpu())
            all_n_gt.append(n_gt.cpu())
            all_m.append(m.cpu())
            all_p_pred.append(p_pred.cpu())
            all_n_pred.append(n_pred.cpu())
            if m_pred is not None:
                all_m_pred.append(m_pred.cpu())

    p_gt = torch.cat(all_p_gt, dim=0)
    n_gt = torch.cat(all_n_gt, dim=0)
    m = torch.cat(all_m, dim=0)
    p_pred = torch.cat(all_p_pred, dim=0)
    n_pred = torch.cat(all_n_pred, dim=0)
    m_pred = torch.cat(all_m_pred, dim=0) if len(all_m_pred) > 0 else None

    metric_out = compute_obs_metrics(
        p_gt=p_gt,
        n_gt=n_gt,
        m=m,
        p_pred=p_pred,
        n_pred=n_pred,
        m_pred=m_pred,
        huber_delta=0.02,
    )

    save_json(out_dir / "summary.json", metric_out.summary)
    save_csv(out_dir / "per_ray.csv", metric_out.per_ray_rows)
    save_csv(out_dir / "per_sample.csv", metric_out.per_sample_rows)

    plot_summary_metrics(metric_out.summary, fig_dir, save_pdf=args.save_pdf)
    ray_rmse = [row["rmse_hit"] for row in metric_out.per_ray_rows]
    ray_ang = [row["angle_mean_deg_hit"] for row in metric_out.per_ray_rows]
    plot_per_ray(ray_rmse, "Per-ray RMSE (hit)", "rmse", fig_dir / "per_ray_rmse.png", save_pdf=args.save_pdf)
    plot_per_ray(ray_ang, "Per-ray normal angle error (hit)", "deg", fig_dir / "per_ray_angle.png", save_pdf=args.save_pdf)
    plot_hist(metric_out.flat_hit_pos_l2, "RMSE distribution (hit)", "L2 error", fig_dir / "hist_rmse.png", save_pdf=args.save_pdf)
    plot_hist(metric_out.flat_hit_angle_deg, "Normal angle distribution (hit)", "deg", fig_dir / "hist_angle.png", save_pdf=args.save_pdf)

    num_plot = min(args.num_plot_samples, p_gt.shape[0])
    idx = np.random.choice(p_gt.shape[0], size=num_plot, replace=False)
    p_gt_np = p_gt[idx].numpy()
    p_pred_np = p_pred[idx].numpy()
    m_np = m[idx].numpy()
    plot_scatter_hit_points(
        p_gt=p_gt_np,
        p_pred=p_pred_np,
        m=m_np,
        out_path=fig_dir / "scatter_hit_points.png",
        max_points=5000,
        save_pdf=args.save_pdf,
    )

    if args.sweep_qego:
        sweep_rows = run_sweep(
            payload=payload,
            model=model,
            device=device,
            joint_idx=args.joint_idx,
            num_points=args.num_points,
            ref_index=args.sweep_ref_index,
            sweep_pool=args.sweep_pool,
            qobs_weight=args.sweep_qobs_weight,
        )
        save_csv(out_dir / f"sweep_joint_{args.joint_idx}.csv", sweep_rows)
        save_json(
            out_dir / f"sweep_joint_{args.joint_idx}.json",
            {
                "joint_idx": args.joint_idx,
                "num_points": args.num_points,
                "num_rows": len(sweep_rows),
            },
        )
        qv = np.array([r["actual_q"] for r in sweep_rows], dtype=np.float32)
        rv = np.array([r["rmse_hit"] for r in sweep_rows], dtype=np.float32)
        av = np.array([r["angle_mean_deg_hit"] for r in sweep_rows], dtype=np.float32)
        plot_sweep_curve(qv, rv, av, args.joint_idx, fig_dir / f"sweep_joint_{args.joint_idx}.png", save_pdf=args.save_pdf)

    print("[compare_obs_fit] Summary:")
    print(
        "  pos_rmse_hit={:.6f}, pos_mae_hit={:.6f}, normal_angle_mean_deg_hit={:.6f}, hit_ratio={:.4f}".format(
            metric_out.summary.get("pos_rmse_hit", float("nan")),
            metric_out.summary.get("pos_mae_hit", float("nan")),
            metric_out.summary.get("normal_angle_mean_deg_hit", float("nan")),
            metric_out.summary.get("hit_ratio", float("nan")),
        )
    )
    if "hit_acc" in metric_out.summary:
        print(
            "  hit_acc={:.4f}, precision={:.4f}, recall={:.4f}".format(
                metric_out.summary.get("hit_acc", float("nan")),
                metric_out.summary.get("hit_precision", float("nan")),
                metric_out.summary.get("hit_recall", float("nan")),
            )
        )
    print(f"[compare_obs_fit] outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
