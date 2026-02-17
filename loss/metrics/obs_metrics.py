from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ObsMetricOutputs:
    summary: dict[str, float]
    per_ray_rows: list[dict[str, Any]]
    per_sample_rows: list[dict[str, Any]]
    flat_hit_pos_l2: np.ndarray
    flat_hit_angle_deg: np.ndarray


def _safe_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _binary_metrics(y_true: torch.Tensor, y_prob: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    y_hat = (y_prob >= threshold).to(y_true.dtype)
    tp = ((y_hat == 1) & (y_true == 1)).sum().item()
    tn = ((y_hat == 0) & (y_true == 0)).sum().item()
    fp = ((y_hat == 1) & (y_true == 0)).sum().item()
    fn = ((y_hat == 0) & (y_true == 1)).sum().item()

    total = max(tp + tn + fp + fn, 1)
    acc = (tp + tn) / total
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "hit_acc": float(acc),
        "hit_precision": float(prec),
        "hit_recall": float(rec),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def compute_obs_metrics(
    p_gt: torch.Tensor,
    n_gt: torch.Tensor,
    m: torch.Tensor,
    p_pred: torch.Tensor,
    n_pred: torch.Tensor,
    m_pred: torch.Tensor | None = None,
    huber_delta: float = 0.02,
    eps: float = 1e-6,
) -> ObsMetricOutputs:
    """Compute overall/per-ray/per-sample metrics.

    Shapes:
        p_gt, p_pred: (N, R, 3)
        n_gt, n_pred: (N, R, 3)
        m: (N, R) or (N, R, 1)
        m_pred: (N, R) or (N, R, 1) optional
    """
    assert p_gt.ndim == 3 and p_gt.shape[-1] == 3
    assert p_pred.shape == p_gt.shape
    assert n_gt.shape == p_gt.shape
    assert n_pred.shape == p_gt.shape

    if m.ndim == 3:
        assert m.shape[-1] == 1
        m = m.squeeze(-1)
    assert m.ndim == 2 and m.shape[:2] == p_gt.shape[:2]

    n_gt = _safe_normalize(n_gt, eps=eps)
    n_pred = _safe_normalize(n_pred, eps=eps)

    hit = m > 0.5  # (N, R)
    hit_count = hit.sum().item()

    pos_diff = p_pred - p_gt
    pos_l2 = pos_diff.norm(dim=-1)  # (N, R)
    pos_abs = pos_diff.abs()  # (N, R, 3)

    cos = (n_gt * n_pred).sum(dim=-1).clamp(-1.0, 1.0)  # (N, R)
    angle_deg = torch.rad2deg(torch.acos(cos))

    if hit_count > 0:
        mae = pos_abs[hit].mean().item()
        rmse = torch.sqrt((pos_l2[hit].pow(2)).mean()).item()
        huber = F.smooth_l1_loss(p_pred[hit], p_gt[hit], beta=huber_delta, reduction="mean").item()
        cos_mean = cos[hit].mean().item()
        angle_mean = angle_deg[hit].mean().item()
        angle_p50 = torch.quantile(angle_deg[hit], 0.5).item()
        angle_p90 = torch.quantile(angle_deg[hit], 0.9).item()
        angle_p95 = torch.quantile(angle_deg[hit], 0.95).item()
    else:
        mae = rmse = huber = cos_mean = angle_mean = angle_p50 = angle_p90 = angle_p95 = float("nan")

    summary = {
        "num_samples": float(p_gt.shape[0]),
        "num_rays": float(p_gt.shape[1]),
        "hit_ratio": float(hit.float().mean().item()),
        "pos_mae_hit": float(mae),
        "pos_rmse_hit": float(rmse),
        "pos_huber_hit": float(huber),
        "normal_cos_mean_hit": float(cos_mean),
        "normal_angle_mean_deg_hit": float(angle_mean),
        "normal_angle_p50_deg_hit": float(angle_p50),
        "normal_angle_p90_deg_hit": float(angle_p90),
        "normal_angle_p95_deg_hit": float(angle_p95),
    }

    if m_pred is not None:
        if m_pred.ndim == 3:
            assert m_pred.shape[-1] == 1
            m_pred = m_pred.squeeze(-1)
        assert m_pred.shape == m.shape
        bce = F.binary_cross_entropy(m_pred.clamp(1e-6, 1 - 1e-6), m.float()).item()
        summary["hit_bce"] = float(bce)
        summary.update(_binary_metrics(m.float(), m_pred.float(), threshold=0.5))

    N, R = p_gt.shape[:2]
    per_ray_rows: list[dict[str, Any]] = []
    for r in range(R):
        hr = hit[:, r]
        cnt = int(hr.sum().item())
        if cnt > 0:
            rmse_r = torch.sqrt((pos_l2[:, r][hr].pow(2)).mean()).item()
            ang_r = angle_deg[:, r][hr].mean().item()
            cos_r = cos[:, r][hr].mean().item()
        else:
            rmse_r = float("nan")
            ang_r = float("nan")
            cos_r = float("nan")
        per_ray_rows.append(
            {
                "ray_idx": r,
                "hit_count": cnt,
                "hit_ratio": float(hr.float().mean().item()),
                "rmse_hit": float(rmse_r),
                "angle_mean_deg_hit": float(ang_r),
                "cos_mean_hit": float(cos_r),
            }
        )

    per_sample_rows: list[dict[str, Any]] = []
    for i in range(N):
        hi = hit[i]
        cnt = int(hi.sum().item())
        if cnt > 0:
            rmse_i = torch.sqrt((pos_l2[i][hi].pow(2)).mean()).item()
            ang_i = angle_deg[i][hi].mean().item()
            cos_i = cos[i][hi].mean().item()
        else:
            rmse_i = float("nan")
            ang_i = float("nan")
            cos_i = float("nan")
        per_sample_rows.append(
            {
                "sample_idx": i,
                "hit_count": cnt,
                "hit_ratio": float(hi.float().mean().item()),
                "rmse_hit": float(rmse_i),
                "angle_mean_deg_hit": float(ang_i),
                "cos_mean_hit": float(cos_i),
            }
        )

    flat_hit_pos_l2 = pos_l2[hit].detach().cpu().numpy() if hit_count > 0 else np.array([], dtype=np.float32)
    flat_hit_angle = angle_deg[hit].detach().cpu().numpy() if hit_count > 0 else np.array([], dtype=np.float32)

    return ObsMetricOutputs(
        summary=summary,
        per_ray_rows=per_ray_rows,
        per_sample_rows=per_sample_rows,
        flat_hit_pos_l2=flat_hit_pos_l2,
        flat_hit_angle_deg=flat_hit_angle,
    )
