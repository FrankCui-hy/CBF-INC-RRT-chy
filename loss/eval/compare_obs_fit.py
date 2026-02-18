from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from loss.metrics.obs_metrics import compute_obs_metrics
from loss.utils.config import load_config
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
    plot_sweep_with_bands,
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
    p.add_argument("--min_hit_count", type=int, default=128)
    p.add_argument("--sweep_trials_per_point", type=int, default=10)
    p.add_argument("--max_attempts_per_point", type=int, default=200)
    p.add_argument("--sweep_seed", type=int, default=0)
    return p.parse_args()


def _single_sample_errors(
    p_gt: torch.Tensor,
    n_gt: torch.Tensor,
    m: torch.Tensor,
    p_pred: torch.Tensor,
    n_pred: torch.Tensor,
) -> tuple[float, float, float, float, float, float, int]:
    hit = m > 0.5
    hit_count = int(hit.sum().item())
    if hit_count == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0.0, 0
    pos_vec = p_pred - p_gt
    pos_l2 = pos_vec.norm(dim=-1)
    rmse = torch.sqrt((pos_l2[hit].pow(2)).mean()).item()
    mae = pos_vec[hit].abs().mean().item()
    n_gt = torch.nn.functional.normalize(n_gt, dim=-1, eps=1e-6)
    n_pred = torch.nn.functional.normalize(n_pred, dim=-1, eps=1e-6)
    cos = (n_gt * n_pred).sum(dim=-1).clamp(-1.0, 1.0).abs()
    ang_all = torch.rad2deg(torch.acos(cos))[hit]
    ang = ang_all.mean().item()
    ang_p90 = torch.quantile(ang_all, 0.9).item()
    ang_p95 = torch.quantile(ang_all, 0.95).item()
    return float(rmse), float(mae), float(ang), float(ang_p90), float(ang_p95), float(hit.float().mean().item()), hit_count


def _build_sweep_pool(
    payload: dict[str, Any],
    ref_index: int,
    sweep_pool: int,
):
    q_ego = payload["q_ego"].float()
    q_obs = payload["q_obs"].float()
    N = q_ego.shape[0]
    ref_index = int(np.clip(ref_index, 0, N - 1))
    q_obs_ref = q_obs[ref_index]
    qobs_dist = (q_obs - q_obs_ref.unsqueeze(0)).norm(dim=-1)
    pool_k = min(int(sweep_pool), N)
    pool_idx = torch.topk(-qobs_dist, k=pool_k).indices
    return pool_idx, qobs_dist


def sweep_collect_trials(
    payload: dict[str, Any],
    model,
    device: torch.device,
    joint_idx: int,
    q_val: float,
    pool_idx: torch.Tensor,
    qobs_dist: torch.Tensor,
    min_hit_count: int,
    trials_per_point: int,
    max_attempts_per_point: int,
    qobs_weight: float,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    q_ego = payload["q_ego"].float()
    q_obs = payload["q_obs"].float()
    p_gt = payload["p_gt"].float()
    n_gt = payload["n_gt"].float()
    m = payload["m"].float()

    if pool_idx.numel() == 0:
        return []

    cand = pool_idx.cpu().numpy().astype(np.int64)
    if qobs_weight > 0.0:
        d = qobs_dist[pool_idx].cpu().numpy().astype(np.float64)
        d = d - d.min()
        prob = np.exp(-qobs_weight * d)
        prob = prob / (prob.sum() + 1e-12)
    else:
        prob = None

    trials: list[dict[str, float]] = []
    attempts = 0
    model.eval()
    with torch.no_grad():
        while len(trials) < int(trials_per_point) and attempts < int(max_attempts_per_point):
            attempts += 1
            idx = int(rng.choice(cand, p=prob))
            hit = m[idx].squeeze(-1) if m[idx].ndim == 2 else m[idx]
            hit_count = int((hit > 0.5).sum().item())
            hit_ratio = float((hit > 0.5).float().mean().item())
            if hit_count < int(min_hit_count):
                continue

            qe_mod = q_ego[idx].clone()
            qe_mod[int(joint_idx)] = float(q_val)
            qe_i = qe_mod.unsqueeze(0).to(device)
            qo_i = q_obs[idx : idx + 1].to(device)
            pred = model(qe_i, qo_i)
            p_pred_i = pred["p_hat"][0].cpu()
            n_pred_i = pred["n_hat"][0].cpu()

            rmse_i, mae_i, ang_i, ang_p90_i, ang_p95_i, _, _ = _single_sample_errors(
                p_gt[idx], n_gt[idx], hit, p_pred_i, n_pred_i
            )
            trials.append(
                {
                    "sample_idx": float(idx),
                    "hit_count": float(hit_count),
                    "hit_ratio": float(hit_ratio),
                    "rmse_hit": float(rmse_i),
                    "pos_mae_hit": float(mae_i),
                    "angle_mean_deg_hit": float(ang_i),
                    "normal_angle_p90_deg_hit": float(ang_p90_i),
                    "normal_angle_p95_deg_hit": float(ang_p95_i),
                }
            )
    # Keep attempts summary in a sentinel row if none accepted is handled by caller.
    trials_meta = {"attempts": float(attempts)}
    trials.append({"_meta_attempts_only": 1.0, **trials_meta})
    return trials


def sweep_aggregate_stats(
    q_val: float,
    trials: list[dict[str, float]],
    trials_target: int,
) -> dict[str, float]:
    attempts = 0
    valid = []
    for t in trials:
        if "_meta_attempts_only" in t:
            attempts = int(t["attempts"])
        else:
            valid.append(t)
    accepted = len(valid)
    rejected = max(attempts - accepted, 0)

    def _ms(key: str) -> tuple[float, float]:
        if accepted == 0:
            return float("nan"), float("nan")
        arr = np.asarray([float(v[key]) for v in valid], dtype=np.float64)
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    hit_m, hit_s = _ms("hit_count")
    rm_m, rm_s = _ms("rmse_hit")
    mae_m, mae_s = _ms("pos_mae_hit")
    ang_m, ang_s = _ms("angle_mean_deg_hit")

    return {
        "q_val": float(q_val),
        "accepted": float(accepted),
        "attempts": float(attempts),
        "rejected": float(rejected),
        "insufficient": float(1 if accepted < int(trials_target) else 0),
        "hit_count_mean": float(hit_m),
        "hit_count_std": float(hit_s),
        "rmse_mean": float(rm_m),
        "rmse_std": float(rm_s),
        "pos_mae_mean": float(mae_m),
        "pos_mae_std": float(mae_s),
        "angle_mean": float(ang_m),
        "angle_std": float(ang_s),
    }


def run_sweep(
    payload: dict[str, Any],
    model,
    device: torch.device,
    joint_idx: int,
    num_points: int,
    ref_index: int,
    sweep_pool: int,
    qobs_weight: float,
    min_hit_count: int,
    sweep_trials_per_point: int,
    max_attempts_per_point: int,
    sweep_seed: int,
) -> list[dict[str, float]]:
    q_ego = payload["q_ego"].float()
    _, n = q_ego.shape
    assert 0 <= joint_idx < n

    pool_idx, qobs_dist = _build_sweep_pool(payload, ref_index=ref_index, sweep_pool=sweep_pool)
    q_min = q_ego[:, joint_idx].min().item()
    q_max = q_ego[:, joint_idx].max().item()
    targets = torch.linspace(q_min, q_max, steps=num_points)

    rng = np.random.default_rng(int(sweep_seed))
    rows: list[dict[str, float]] = []
    for t in targets:
        q_val = float(t.item())
        trials = sweep_collect_trials(
            payload=payload,
            model=model,
            device=device,
            joint_idx=joint_idx,
            q_val=q_val,
            pool_idx=pool_idx,
            qobs_dist=qobs_dist,
            min_hit_count=min_hit_count,
            trials_per_point=sweep_trials_per_point,
            max_attempts_per_point=max_attempts_per_point,
            qobs_weight=qobs_weight,
            rng=rng,
        )
        rows.append(sweep_aggregate_stats(q_val, trials, trials_target=sweep_trials_per_point))
    return rows


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    cfg = load_config(args.config)

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

    unsigned_normal = bool(cfg.get("loss", {}).get("obs", {}).get("unsigned_normal", True))
    huber_delta = float(cfg.get("loss", {}).get("obs", {}).get("huber_delta", 0.05))
    metric_out = compute_obs_metrics(
        p_gt=p_gt,
        n_gt=n_gt,
        m=m,
        p_pred=p_pred,
        n_pred=n_pred,
        m_pred=m_pred,
        huber_delta=huber_delta,
        unsigned_normal=unsigned_normal,
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
            min_hit_count=args.min_hit_count,
            sweep_trials_per_point=args.sweep_trials_per_point,
            max_attempts_per_point=args.max_attempts_per_point,
            sweep_seed=args.sweep_seed,
        )
        save_csv(out_dir / "sweep_stats.csv", sweep_rows)
        save_csv(out_dir / f"sweep_joint_{args.joint_idx}.csv", sweep_rows)
        save_json(
            out_dir / "sweep_stats.json",
            {
                "joint_idx": args.joint_idx,
                "num_points": args.num_points,
                "num_rows": len(sweep_rows),
                "min_hit_count": int(args.min_hit_count),
                "sweep_trials_per_point": int(args.sweep_trials_per_point),
                "max_attempts_per_point": int(args.max_attempts_per_point),
                "sweep_seed": int(args.sweep_seed),
            },
        )
        qv = np.array([r["q_val"] for r in sweep_rows], dtype=np.float32)
        rv_m = np.array([r["rmse_mean"] for r in sweep_rows], dtype=np.float32)
        rv_s = np.array([r["rmse_std"] for r in sweep_rows], dtype=np.float32)
        av_m = np.array([r["angle_mean"] for r in sweep_rows], dtype=np.float32)
        av_s = np.array([r["angle_std"] for r in sweep_rows], dtype=np.float32)
        hv_m = np.array([r["hit_count_mean"] for r in sweep_rows], dtype=np.float32)
        acc = np.array([r["accepted"] for r in sweep_rows], dtype=np.float32)
        plot_sweep_with_bands(
            qv,
            rv_m,
            rv_s,
            av_m,
            av_s,
            hv_m,
            acc,
            int(args.sweep_trials_per_point),
            args.joint_idx,
            fig_dir / f"sweep_joint_{args.joint_idx}.png",
            save_pdf=args.save_pdf,
        )
        insufficient = int(np.sum(acc < float(args.sweep_trials_per_point)))
        acc_min = float(np.nanmin(acc)) if acc.size > 0 else float("nan")
        acc_max = float(np.nanmax(acc)) if acc.size > 0 else float("nan")
        print(
            f"[compare_obs_fit][sweep] accepted range={acc_min:.0f}..{acc_max:.0f}, "
            f"insufficient_points={insufficient}/{len(sweep_rows)}"
        )

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
