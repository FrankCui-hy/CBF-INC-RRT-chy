from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn.functional as F


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

    # Position loss (hit only): Huber
    l1 = F.smooth_l1_loss(p_hat, p_gt, beta=delta, reduction="none")
    if use_hard_weight:
        pos_e = (p_hat - p_gt).norm(dim=-1, keepdim=True)
        mean_e = (m_col * pos_e).sum() / num_hit
        w = torch.clamp(pos_e / (mean_e + eps), min=1.0, max=hard_w_max)
    else:
        w = torch.ones_like(m_col)
    l_p = (m_col * w * l1).sum() / num_hit

    # Normal cosine loss (hit only)
    n_gt_norm = F.normalize(n_gt, dim=-1, eps=1e-6)
    cos = (n_gt_norm * n_hat).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    if unsigned_normal:
        cos = cos.abs()
    l_n = (m_col * (1.0 - cos)).sum() / num_hit

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
