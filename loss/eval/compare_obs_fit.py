from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from loss.metrics.obs_metrics import compute_obs_metrics
from loss.metrics.pointset import compute_pointset_metrics_single
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
    plot_sweep_scalar_with_band,
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
    p.add_argument("--debug_one_sample", action="store_true")
    p.add_argument("--debug_index", type=int, default=-1)
    return p.parse_args()


def _tensor_stats_1d(x: torch.Tensor) -> dict[str, float]:
    if x.numel() == 0:
        return {"count": 0.0, "mean": float("nan"), "std": float("nan"), "p95": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "count": float(x.numel()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "p95": float(torch.quantile(x, 0.95).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def _tensor_stats_xyz(x: torch.Tensor) -> dict[str, list[float]]:
    # x: (M,3)
    if x.numel() == 0:
        nan3 = [float("nan"), float("nan"), float("nan")]
        return {"min_xyz": nan3, "max_xyz": nan3, "mean_xyz": nan3}
    return {
        "min_xyz": [float(v) for v in x.min(dim=0).values.tolist()],
        "max_xyz": [float(v) for v in x.max(dim=0).values.tolist()],
        "mean_xyz": [float(v) for v in x.mean(dim=0).tolist()],
    }


def run_debug_one_sample(
    args: argparse.Namespace,
    payload: dict[str, Any],
    model,
    device: torch.device,
    out_dir: Path,
) -> bool:
    if not args.debug_one_sample and args.debug_index < 0:
        return False

    N = int(payload["q_ego"].shape[0])
    idx = 0 if args.debug_one_sample else int(np.clip(args.debug_index, 0, N - 1))

    qe = payload["q_ego"][idx : idx + 1].float().to(device)
    qo = payload["q_obs"][idx : idx + 1].float().to(device)
    p_gt = payload["p_gt"][idx].float().cpu()
    n_gt = payload["n_gt"][idx].float().cpu()
    m = payload["m"][idx].float().cpu()
    if m.ndim == 2:
        m = m.squeeze(-1)

    with torch.no_grad():
        pred = model(qe, qo)
        p_pred = pred["p_hat"][0].float().cpu()
        n_pred = torch.nn.functional.normalize(pred["n_hat"][0].float().cpu(), dim=-1, eps=1e-6)

    hit = m > 0.5
    hit_count = int(hit.sum().item())

    # Step 1: range/scale checks.
    p_gt_hit = p_gt[hit]
    p_pred_hit = p_pred[hit]
    n_gt_hit = n_gt[hit]
    n_pred_hit = n_pred[hit]
    e_hit = (p_pred_hit - p_gt_hit).norm(dim=-1) if hit_count > 0 else torch.zeros((0,))
    pgt_norm = p_gt_hit.norm(dim=-1) if hit_count > 0 else torch.zeros((0,))
    ppd_norm = p_pred_hit.norm(dim=-1) if hit_count > 0 else torch.zeros((0,))

    step1 = {
        "hit_count": float(hit_count),
        "p_gt_xyz": _tensor_stats_xyz(p_gt_hit),
        "p_pred_xyz": _tensor_stats_xyz(p_pred_hit),
        "p_gt_norm": _tensor_stats_1d(pgt_norm),
        "p_pred_norm": _tensor_stats_1d(ppd_norm),
        "pos_error_norm": _tensor_stats_1d(e_hit),
    }

    # Step 2: normal diagnostics.
    if hit_count > 0:
        n_gt_n = torch.nn.functional.normalize(n_gt_hit, dim=-1, eps=1e-6)
        dot = (n_pred_hit * n_gt_n).sum(dim=-1).clamp(-1.0, 1.0)
        absdot = dot.abs()
        angle_signed = torch.rad2deg(torch.acos(dot))
        angle_unsigned = torch.rad2deg(torch.acos(absdot))
    else:
        dot = absdot = angle_signed = angle_unsigned = torch.zeros((0,))

    step2 = {
        "dot": _tensor_stats_1d(dot),
        "absdot": _tensor_stats_1d(absdot),
        "angle_signed_deg": _tensor_stats_1d(angle_signed),
        "angle_unsigned_deg": _tensor_stats_1d(angle_unsigned),
    }

    # Step 3: index-vs-NN alignment.
    if hit_count > 0:
        rmse_index = float(torch.sqrt(((p_pred_hit - p_gt_hit).norm(dim=-1).pow(2)).mean()).item())
        dmat = torch.cdist(p_pred_hit, p_gt_hit)  # (M, M)
        nn_min = dmat.min(dim=1).values
        rmse_nn = float(torch.sqrt((nn_min.pow(2)).mean()).item())
    else:
        rmse_index = float("nan")
        rmse_nn = float("nan")

    step3 = {
        "rmse_index": rmse_index,
        "rmse_nn": rmse_nn,
        "rmse_nn_over_index": float(rmse_nn / (rmse_index + 1e-12)) if np.isfinite(rmse_index) else float("nan"),
    }

    # Step 4: no-hit sanity.
    no_hit_idx = torch.where(~hit)[0]
    sel = no_hit_idx[:5]
    no_hit_preview = []
    for j in sel.tolist():
        no_hit_preview.append(
            {
                "ray_idx": float(j),
                "p_gt": [float(v) for v in p_gt[j].tolist()],
                "n_gt": [float(v) for v in n_gt[j].tolist()],
                "m": float(m[j].item()),
            }
        )
    step4 = {
        "mask_shape": list(m.shape),
        "mask_dtype": str(m.dtype),
        "hit_count": float(hit_count),
        "hit_count_recomputed": float((m > 0.5).sum().item()),
        "no_hit_preview": no_hit_preview,
        "single_sample_error_formula": "rmse = sqrt(mean(||p_pred-p_gt||^2 over hit rays only))",
    }

    # Step 5: conclusions.
    reasons = []
    pgt_mean = step1["p_gt_norm"]["mean"]
    ppd_mean = step1["p_pred_norm"]["mean"]
    scale_mismatch = np.isfinite(pgt_mean) and np.isfinite(ppd_mean) and (
        (pgt_mean > 0.5 and ppd_mean < 0.2) or (ppd_mean > 0.5 and pgt_mean < 0.2) or (max(pgt_mean, ppd_mean) / (min(pgt_mean, ppd_mean) + 1e-12) > 3.0)
    )
    if scale_mismatch:
        reasons.append("尺度/归一化或坐标系不一致（Top-1）")

    nn_ratio = step3["rmse_nn_over_index"]
    if np.isfinite(nn_ratio) and nn_ratio < 0.7:
        reasons.append("ray 索引/排列不一致（Top-1/Top-2）")

    ang_s = step2["angle_signed_deg"]["mean"]
    ang_u = step2["angle_unsigned_deg"]["mean"]
    if np.isfinite(ang_s) and np.isfinite(ang_u) and (ang_s - ang_u) > 10.0:
        reasons.append("法向符号歧义明显（建议评估/训练统一无符号）")

    if len(reasons) == 0:
        reasons.append("更可能是坐标系旋转未对齐或模型拟合不足（需检查 world/sensor frame 一致性）")

    frame_assumption = {
        "p_gt": "world (from pybullet raycast)",
        "n_gt": "world (from pybullet raycast)",
        "p_pred": "assumed world for direct compare",
        "n_pred": "assumed world for direct compare",
        "fix_option_A_world": "若模型输出在sensor/ee系，先用 T_WE(q_ego) 变换 p_pred/n_pred 到 world 再评估",
        "fix_option_B_sensor": "将 p_gt/n_gt 用 T_EW(q_ego) 变换到 sensor 再评估",
        "needed_inputs": "FK 提供 T_WE（可从 environment robot fk 获取）",
        "suggested_files": ["loss/eval/compare_obs_fit.py", "loss/data/collect_dataset_real.py"],
    }

    report = {
        "debug_index": float(idx),
        "meta": payload.get("meta", {}),
        "step1_range_scale": step1,
        "step2_normals": step2,
        "step3_alignment": step3,
        "step4_mask_nohit": step4,
        "frame_assumption": frame_assumption,
        "top_reasons": reasons[:2],
    }

    save_json(out_dir / "debug_report.json", report)
    print("[compare_obs_fit][debug] saved:", out_dir / "debug_report.json")
    print("[compare_obs_fit][debug] Top-1:", reasons[0])
    if len(reasons) > 1:
        print("[compare_obs_fit][debug] Top-2:", reasons[1])
    return True


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


def _single_sample_rmse_nn(
    p_gt: torch.Tensor,
    m: torch.Tensor,
    p_pred: torch.Tensor,
) -> float:
    hit = m > 0.5
    if int(hit.sum().item()) == 0:
        return float("nan")
    gt = p_gt[hit]
    pr = p_pred[hit]
    dmat = torch.cdist(pr, gt)
    nn_min = dmat.min(dim=1).values
    return float(torch.sqrt((nn_min.pow(2)).mean()).item())


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
    pred_mask_mode: str,
    pred_hit_tau: float,
    max_points: int,
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
            rmse_nn_i = _single_sample_rmse_nn(p_gt[idx], hit, p_pred_i)
            psm = compute_pointset_metrics_single(
                p_gt=p_gt[idx],
                n_gt=n_gt[idx],
                m_gt=hit,
                p_pred=p_pred_i,
                n_pred=n_pred_i,
                m_pred=None,
                pred_mask_mode=pred_mask_mode,
                pred_hit_tau=pred_hit_tau,
                max_points=max_points,
            )
            trials.append(
                {
                    "sample_idx": float(idx),
                    "hit_count": float(hit_count),
                    "hit_ratio": float(hit_ratio),
                    "rmse_hit": float(rmse_i),
                    "rmse_index": float(rmse_i),
                    "rmse_nn": float(rmse_nn_i),
                    "chamfer": float(psm.chamfer),
                    "pos_mae_hit": float(mae_i),
                    "angle_mean_deg_hit": float(ang_i),
                    "angle_unsigned_mean_deg": float(psm.angle_unsigned_mean_deg),
                    "normal_angle_p90_deg_hit": float(ang_p90_i),
                    "angle_unsigned_p90_deg": float(psm.angle_unsigned_p90_deg),
                    "normal_angle_p95_deg_hit": float(ang_p95_i),
                    "normal_absdot_mean": float(psm.normal_absdot_mean),
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
    rm_idx_m, rm_idx_s = _ms("rmse_index")
    rm_nn_m, rm_nn_s = _ms("rmse_nn")
    mae_m, mae_s = _ms("pos_mae_hit")
    ang_m, ang_s = _ms("angle_unsigned_mean_deg")
    ch_m, ch_s = _ms("chamfer")

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
        "rmse_index_mean": float(rm_idx_m),
        "rmse_index_std": float(rm_idx_s),
        "rmse_nn_mean": float(rm_nn_m),
        "rmse_nn_std": float(rm_nn_s),
        "chamfer_mean": float(ch_m),
        "chamfer_std": float(ch_s),
        "pos_mae_mean": float(mae_m),
        "pos_mae_std": float(mae_s),
        "angle_unsigned_mean": float(ang_m),
        "angle_unsigned_std": float(ang_s),
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
    pred_mask_mode: str,
    pred_hit_tau: float,
    max_points: int,
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
            pred_mask_mode=pred_mask_mode,
            pred_hit_tau=pred_hit_tau,
            max_points=max_points,
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
    if run_debug_one_sample(args, payload, model, device, out_dir):
        return

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
    obs_cfg = cfg.get("loss", {}).get("obs", {})
    pred_mask_mode_eval = str(obs_cfg.get("pred_mask_mode", "gt_hit"))
    pred_hit_tau_eval = float(obs_cfg.get("pred_hit_tau", 0.5))
    max_points_eval = int(obs_cfg.get("max_points", 256))
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

    # Unordered point-set metrics (default report to avoid index-mismatch bias).
    vals_rmse_nn, vals_ch, vals_ang_u, vals_absdot = [], [], [], []
    N_eval = p_gt.shape[0]
    for i in range(N_eval):
        m_pred_i = None if m_pred is None else m_pred[i]
        ps = compute_pointset_metrics_single(
            p_gt=p_gt[i],
            n_gt=n_gt[i],
            m_gt=m[i],
            p_pred=p_pred[i],
            n_pred=n_pred[i],
            m_pred=m_pred_i,
            pred_mask_mode=pred_mask_mode_eval,
            pred_hit_tau=pred_hit_tau_eval,
            max_points=max_points_eval,
        )
        if np.isfinite(ps.rmse_nn):
            vals_rmse_nn.append(ps.rmse_nn)
        if np.isfinite(ps.chamfer):
            vals_ch.append(ps.chamfer)
        if np.isfinite(ps.angle_unsigned_mean_deg):
            vals_ang_u.append(ps.angle_unsigned_mean_deg)
        if np.isfinite(ps.normal_absdot_mean):
            vals_absdot.append(ps.normal_absdot_mean)

    if len(vals_rmse_nn) > 0:
        metric_out.summary["pos_rmse_nn_hit"] = float(np.mean(vals_rmse_nn))
    if len(vals_ch) > 0:
        metric_out.summary["pos_chamfer_hit"] = float(np.mean(vals_ch))
    if len(vals_ang_u) > 0:
        metric_out.summary["normal_angle_unsigned_mean_deg_hit"] = float(np.mean(vals_ang_u))
    if len(vals_absdot) > 0:
        metric_out.summary["normal_absdot_mean_hit"] = float(np.mean(vals_absdot))

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
            pred_mask_mode=pred_mask_mode_eval,
            pred_hit_tau=pred_hit_tau_eval,
            max_points=max_points_eval,
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
        rv_m = np.array([r["rmse_index_mean"] for r in sweep_rows], dtype=np.float32)
        rv_s = np.array([r["rmse_index_std"] for r in sweep_rows], dtype=np.float32)
        rv_nn_m = np.array([r["rmse_nn_mean"] for r in sweep_rows], dtype=np.float32)
        rv_nn_s = np.array([r["rmse_nn_std"] for r in sweep_rows], dtype=np.float32)
        av_m = np.array([r["angle_unsigned_mean"] for r in sweep_rows], dtype=np.float32)
        av_s = np.array([r["angle_unsigned_std"] for r in sweep_rows], dtype=np.float32)
        hv_m = np.array([r["hit_count_mean"] for r in sweep_rows], dtype=np.float32)
        acc = np.array([r["accepted"] for r in sweep_rows], dtype=np.float32)
        ch_m = np.array([r["chamfer_mean"] for r in sweep_rows], dtype=np.float32)
        ch_s = np.array([r["chamfer_std"] for r in sweep_rows], dtype=np.float32)
        plot_sweep_with_bands(
            qv,
            rv_m,
            rv_s,
            rv_nn_m,
            rv_nn_s,
            av_m,
            av_s,
            hv_m,
            acc,
            int(args.sweep_trials_per_point),
            args.joint_idx,
            fig_dir / f"sweep_joint_{args.joint_idx}.png",
            save_pdf=args.save_pdf,
        )
        plot_sweep_scalar_with_band(
            q_values=qv,
            y_mean=ch_m,
            y_std=ch_s,
            joint_idx=args.joint_idx,
            title="Sweep Chamfer (unordered)",
            ylabel="Chamfer L2",
            out_path=fig_dir / f"sweep_joint_{args.joint_idx}_chamfer.png",
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
