from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class PointSetMetricsSingle:
    hit_count_gt: int
    pred_count: int
    rmse_nn: float
    chamfer: float
    chamfer_p2q: float
    chamfer_q2p: float
    angle_unsigned_mean_deg: float
    angle_unsigned_p90_deg: float
    normal_absdot_mean: float


def _mask_to_bool(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    return mask > 0.5


def _take_max_points(x: torch.Tensor, max_points: int) -> torch.Tensor:
    if max_points > 0 and x.shape[0] > max_points:
        return x[:max_points]
    return x


def _build_point_sets_single(
    p_gt: torch.Tensor,
    p_pred: torch.Tensor,
    m_gt: torch.Tensor,
    m_pred: Optional[torch.Tensor],
    n_gt: Optional[torch.Tensor] = None,
    n_pred: Optional[torch.Tensor] = None,
    pred_mask_mode: str = "gt_hit",
    pred_hit_tau: float = 0.5,
    max_points: int = 256,
):
    gt_mask = _mask_to_bool(m_gt)
    if pred_mask_mode == "pred_hit" and m_pred is not None:
        if m_pred.ndim == 2 and m_pred.shape[-1] == 1:
            m_pred = m_pred.squeeze(-1)
        pr_mask = m_pred > float(pred_hit_tau)
    else:
        pr_mask = gt_mask

    P = _take_max_points(p_gt[gt_mask], max_points)
    Q = _take_max_points(p_pred[pr_mask], max_points)
    NG = None if n_gt is None else _take_max_points(n_gt[gt_mask], max_points)
    NQ = None if n_pred is None else _take_max_points(n_pred[pr_mask], max_points)
    return P, Q, NG, NQ


def one_way_nn_rmse_single(P: torch.Tensor, Q: torch.Tensor) -> float:
    rmse_t = one_way_nn_rmse_tensor_single(P, Q)
    return float(rmse_t.item()) if torch.isfinite(rmse_t) else float("nan")


def chamfer_l2_single(P: torch.Tensor, Q: torch.Tensor) -> float:
    p2q_t, q2p_t = chamfer_l2_parts_tensor_single(P, Q)
    if not torch.isfinite(p2q_t) or not torch.isfinite(q2p_t):
        return float("nan")
    return float((p2q_t + q2p_t).item())


def chamfer_l2_parts_single(P: torch.Tensor, Q: torch.Tensor) -> tuple[float, float]:
    p2q_t, q2p_t = chamfer_l2_parts_tensor_single(P, Q)
    if not torch.isfinite(p2q_t) or not torch.isfinite(q2p_t):
        return float("nan"), float("nan")
    return float(p2q_t.item()), float(q2p_t.item())


def matched_absdot_loss_single(
    P: torch.Tensor,
    Q: torch.Tensor,
    NG: torch.Tensor,
    NQ: torch.Tensor,
    max_match_dist: float | None = None,
    bidirectional: bool = False,
) -> tuple[float, float, float]:
    loss_t, mean_t, p90_t = matched_absdot_loss_tensor_single(
        P, Q, NG, NQ, max_match_dist=max_match_dist, bidirectional=bidirectional
    )
    if not torch.isfinite(loss_t):
        return float("nan"), float("nan"), float("nan")
    return (
        float(loss_t.item()),
        float(mean_t.item()),
        float(p90_t.item()),
    )


def one_way_nn_rmse_tensor_single(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    if P.shape[0] == 0 or Q.shape[0] == 0:
        return torch.tensor(float("nan"), device=P.device, dtype=P.dtype)
    d = torch.cdist(P, Q)
    nn = d.min(dim=1).values
    return torch.sqrt((nn.pow(2)).mean())


def chamfer_l2_parts_tensor_single(P: torch.Tensor, Q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if P.shape[0] == 0 or Q.shape[0] == 0:
        nan = torch.tensor(float("nan"), device=P.device, dtype=P.dtype)
        return nan, nan
    d = torch.cdist(P, Q)
    p2q = d.min(dim=1).values.pow(2).mean()
    q2p = d.min(dim=0).values.pow(2).mean()
    return p2q, q2p


def _matched_absdot_oneway_tensor(
    P: torch.Tensor,
    Q: torch.Tensor,
    NG: torch.Tensor,
    NQ: torch.Tensor,
    max_match_dist: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if P.shape[0] == 0 or Q.shape[0] == 0 or NG is None or NQ is None:
        nan = torch.tensor(float("nan"), device=P.device, dtype=P.dtype)
        return nan, nan, nan
    d = torch.cdist(P, Q)
    nn = d.min(dim=1)
    nn_idx = nn.indices
    nn_dist = nn.values
    n_gt = F.normalize(NG, dim=-1, eps=1e-6)
    n_pr = F.normalize(NQ[nn_idx], dim=-1, eps=1e-6)
    absdot = (n_gt * n_pr).sum(dim=-1).clamp(-1.0, 1.0).abs()

    if max_match_dist is not None and float(max_match_dist) > 0.0:
        valid = nn_dist <= float(max_match_dist)
        if not torch.any(valid):
            nan = torch.tensor(float("nan"), device=P.device, dtype=P.dtype)
            return nan, nan, nan
        absdot = absdot[valid]

    angle = torch.rad2deg(torch.acos(absdot))
    return (1.0 - absdot).mean(), angle.mean(), torch.quantile(angle, 0.9)


def matched_absdot_loss_tensor_single(
    P: torch.Tensor,
    Q: torch.Tensor,
    NG: torch.Tensor,
    NQ: torch.Tensor,
    max_match_dist: float | None = None,
    bidirectional: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    l1, a1, p1 = _matched_absdot_oneway_tensor(P, Q, NG, NQ, max_match_dist=max_match_dist)
    if not bidirectional:
        return l1, a1, p1

    l2, a2, p2 = _matched_absdot_oneway_tensor(Q, P, NQ, NG, max_match_dist=max_match_dist)
    if torch.isfinite(l1) and torch.isfinite(l2):
        return 0.5 * (l1 + l2), 0.5 * (a1 + a2), 0.5 * (p1 + p2)
    if torch.isfinite(l1):
        return l1, a1, p1
    return l2, a2, p2


def compute_pointset_metrics_single(
    p_gt: torch.Tensor,
    n_gt: torch.Tensor,
    m_gt: torch.Tensor,
    p_pred: torch.Tensor,
    n_pred: torch.Tensor,
    m_pred: Optional[torch.Tensor] = None,
    pred_mask_mode: str = "gt_hit",
    pred_hit_tau: float = 0.5,
    max_points: int = 256,
) -> PointSetMetricsSingle:
    P, Q, NG, NQ = _build_point_sets_single(
        p_gt=p_gt,
        p_pred=p_pred,
        m_gt=m_gt,
        m_pred=m_pred,
        n_gt=n_gt,
        n_pred=n_pred,
        pred_mask_mode=pred_mask_mode,
        pred_hit_tau=pred_hit_tau,
        max_points=max_points,
    )
    rmse_nn = one_way_nn_rmse_single(P, Q)
    chamfer_p2q, chamfer_q2p = chamfer_l2_parts_single(P, Q)
    chamfer = float(chamfer_p2q + chamfer_q2p) if (torch.isfinite(torch.tensor(chamfer_p2q)) and torch.isfinite(torch.tensor(chamfer_q2p))) else float("nan")
    _, ang_mean, ang_p90 = matched_absdot_loss_single(P, Q, NG, NQ)
    if P.shape[0] == 0 or Q.shape[0] == 0:
        absdot_mean = float("nan")
    else:
        d = torch.cdist(P, Q)
        nn_idx = d.min(dim=1).indices
        n_gt_u = F.normalize(NG, dim=-1, eps=1e-6)
        n_pr_u = F.normalize(NQ[nn_idx], dim=-1, eps=1e-6)
        absdot = (n_gt_u * n_pr_u).sum(dim=-1).clamp(-1.0, 1.0).abs()
        absdot_mean = float(absdot.mean().item())
    return PointSetMetricsSingle(
        hit_count_gt=int(P.shape[0]),
        pred_count=int(Q.shape[0]),
        rmse_nn=float(rmse_nn),
        chamfer=float(chamfer),
        chamfer_p2q=float(chamfer_p2q),
        chamfer_q2p=float(chamfer_q2p),
        angle_unsigned_mean_deg=float(ang_mean),
        angle_unsigned_p90_deg=float(ang_p90),
        normal_absdot_mean=float(absdot_mean),
    )
