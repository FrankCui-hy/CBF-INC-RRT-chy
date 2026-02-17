from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn.functional as F


def compute_cbf_losses(h: torch.Tensor, hdot: torch.Tensor, y: torch.Tensor, cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """CBF + classification hinge losses.

    Args:
        h: (B,)
        hdot: (B,)
        y: (B,) in {-1, +1}
    """
    alpha = float(cfg["alpha"])
    gamma_s = float(cfg["gamma_s"])
    gamma_u = float(cfg["gamma_u"])
    sign_mode = str(cfg.get("sign_mode", "standard"))

    margin = hdot + alpha * h
    if sign_mode == "flipped":
        margin = -margin
    elif sign_mode != "standard":
        raise ValueError(f"Unknown sign_mode: {sign_mode}")

    l_cbf = F.relu(-margin).mean()

    safe = (y > 0).float()
    unsafe = (y <= 0).float()
    l_cls = (safe * F.relu(gamma_s - h) + unsafe * F.relu(h - gamma_u)).mean()

    w_cbf = float(cfg.get("w_cbf", 1.0))
    w_cls = float(cfg.get("w_cls", 1.0))
    total = w_cbf * l_cbf + w_cls * l_cls
    return {"total": total, "L_cbf": l_cbf, "L_cls": l_cls}
