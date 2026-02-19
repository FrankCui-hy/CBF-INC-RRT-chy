from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn.functional as F

from loss.metrics.pointset import (
    _build_point_sets_single,
    chamfer_l2_parts_tensor_single,
    matched_absdot_loss_tensor_single,
    one_way_nn_rmse_tensor_single,
)


def compute_observation_losses(pred: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Observation supervision loss with hit-mask handling.

    Args:
        pred: output from g_phi
        batch: dataset batch
        cfg: cfg["loss"]["obs"]
    Returns:
        dict containing total and each component
    """
    eps = float(cfg.get("eps", 1e-6))
    delta = float(cfg.get("huber_delta", 0.02))
    unsigned_normal = bool(cfg.get("unsigned_normal", True))
    use_hard_weight = bool(cfg.get("hard_example_weighting", False))
    hard_w_max = float(cfg.get("hard_w_max", 5.0))
    hard_residual_clip = float(cfg.get("hard_residual_clip", 0.0))
    normal_loss_mode = str(cfg.get("normal_loss_mode", "dot2")).lower()
    match_mode = str(cfg.get("match_mode", "index")).lower()
    q2p_weight = float(cfg.get("chamfer_q2p_weight", 1.0))
    p2q_weight = float(cfg.get("chamfer_p2q_weight", 1.0))
    pred_mask_mode = str(cfg.get("pred_mask_mode", "gt_hit")).lower()
    pred_hit_tau = float(cfg.get("pred_hit_tau", 0.5))
    max_points = int(cfg.get("max_points", 256))
    normal_match_max_dist = float(cfg.get("normal_match_max_dist", 0.0))
    normal_bidirectional = bool(cfg.get("normal_bidirectional", True))
    pointset_empty_as_zero = bool(cfg.get("pointset_empty_as_zero", True))

    p_hat = pred["p_hat"]  # (B, R, 3)
    n_hat = pred["n_hat"]  # (B, R, 3)
    m_hat = pred["m_hat"]  # (B, R, 1) or None

    p_gt = batch["p_gt"]
    n_gt = batch["n_gt"]
    m = batch["m"]
    if m.dim() == 2:
        m_col = m.unsqueeze(-1)
    else:
        m_col = m

    num_hit = m_col.sum() + eps

    if match_mode == "index":
        # Position loss (hit only): index-wise Huber
        l1 = F.smooth_l1_loss(p_hat, p_gt, beta=delta, reduction="none")
        if use_hard_weight:
            pos_e = (p_hat - p_gt).norm(dim=-1, keepdim=True)
            if hard_residual_clip > 0.0:
                pos_e = torch.clamp(pos_e, max=hard_residual_clip)
            mean_e = (m_col * pos_e).sum() / num_hit
            w = torch.clamp(pos_e / (mean_e + eps), min=1.0, max=hard_w_max)
        else:
            w = torch.ones_like(m_col)
        l_p = (m_col * w * l1).sum() / num_hit
    elif match_mode in ("nn", "chamfer"):
        # Unordered point-set position loss.
        vals = []
        for b in range(p_gt.shape[0]):
            mb = m[b].squeeze(-1) if m[b].ndim == 2 else m[b]
            mpb = None if m_hat is None else (m_hat[b].squeeze(-1) if m_hat[b].ndim == 2 else m_hat[b])
            P, Q, _, _ = _build_point_sets_single(
                p_gt[b], p_hat[b], mb, mpb,
                pred_mask_mode=pred_mask_mode, pred_hit_tau=pred_hit_tau, max_points=max_points
            )
            if P.shape[0] == 0 or Q.shape[0] == 0:
                if pointset_empty_as_zero:
                    vals.append(torch.zeros((), device=p_gt.device, dtype=p_gt.dtype))
                continue
            if match_mode == "nn":
                v = one_way_nn_rmse_tensor_single(P, Q)
                vals.append(v * v)
            else:
                p2q, q2p = chamfer_l2_parts_tensor_single(P, Q)
                vals.append(p2q_weight * p2q + q2p_weight * q2p)
        l_p = torch.stack(vals).mean() if len(vals) > 0 else torch.zeros((), device=p_gt.device, dtype=p_gt.dtype)
    else:
        raise ValueError(f"Unknown match_mode={match_mode}. Use 'index'|'nn'|'chamfer'.")

    normal_mode = str(cfg.get("normal_mode", "index")).lower()
    if normal_mode == "matched_absdot":
        vals_n = []
        for b in range(p_gt.shape[0]):
            mb = m[b].squeeze(-1) if m[b].ndim == 2 else m[b]
            mpb = None if m_hat is None else (m_hat[b].squeeze(-1) if m_hat[b].ndim == 2 else m_hat[b])
            P, Q, NG, NQ = _build_point_sets_single(
                p_gt[b], p_hat[b], mb, mpb, n_gt[b], n_hat[b],
                pred_mask_mode=pred_mask_mode, pred_hit_tau=pred_hit_tau, max_points=max_points
            )
            loss_n, _, _ = matched_absdot_loss_tensor_single(
                P,
                Q,
                NG,
                NQ,
                max_match_dist=(normal_match_max_dist if normal_match_max_dist > 0.0 else None),
                bidirectional=normal_bidirectional,
            )
            if torch.isfinite(loss_n):
                vals_n.append(loss_n)
            elif pointset_empty_as_zero:
                vals_n.append(torch.zeros((), device=p_gt.device, dtype=p_gt.dtype))
        l_n = torch.stack(vals_n).mean() if len(vals_n) > 0 else torch.zeros((), device=p_gt.device, dtype=p_gt.dtype)
    elif normal_mode == "none":
        l_n = torch.zeros((), device=p_hat.device, dtype=p_hat.dtype)
    else:
        # Backward-compatible index-wise normal loss.
        n_gt_norm = F.normalize(n_gt, dim=-1, eps=1e-6)
        cos = (n_gt_norm * n_hat).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        if unsigned_normal:
            if normal_loss_mode == "dot2":
                n_term = 1.0 - cos.pow(2)
            elif normal_loss_mode == "abs_cos":
                n_term = 1.0 - cos.abs()
            else:
                raise ValueError(f"Unknown normal_loss_mode={normal_loss_mode}. Use 'dot2' or 'abs_cos'.")
        else:
            n_term = 1.0 - cos
        l_n = (m_col * n_term).sum() / num_hit

    # Hit classification loss (optional)
    if m_hat is not None:
        l_m = F.binary_cross_entropy(m_hat, m_col)
    else:
        l_m = torch.zeros((), device=p_hat.device, dtype=p_hat.dtype)

    # Ray consistency loss (hit only)
    s = batch["ray_origin"]
    r = F.normalize(batch["ray_dir"], dim=-1, eps=1e-6)
    d_hat = ((p_hat - s) * r).sum(dim=-1, keepdim=True)
    p_proj = s + d_hat * r
    l_ray = (m_col * (p_hat - p_proj).pow(2)).sum() / num_hit

    w_p = float(cfg.get("lambda_p", 1.0))
    w_n = float(cfg.get("lambda_n", 0.3))
    w_m = float(cfg.get("lambda_m", 1.0))
    w_r = float(cfg.get("lambda_r", 0.2))

    total = w_p * l_p + w_n * l_n + w_m * l_m + w_r * l_ray
    return {
        "total": total,
        "L_p": l_p,
        "L_n": l_n,
        "L_m": l_m,
        "L_ray": l_ray,
    }
