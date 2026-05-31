from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn.functional as F


def compute_cbf_losses(h: torch.Tensor, hdot: torch.Tensor, y: torch.Tensor, cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """CBF + classification hinge losses using the Efficient CBF-INC sign convention.

    Args:
        h: (B,)
        hdot: (B,), normally the minimized CBF derivative over admissible ego controls
        y: (B,) in {-1, +1}, where +1 is safe and -1 is unsafe
    """
    alpha = float(cfg["alpha"])
    gamma_s = abs(float(cfg["gamma_s"]))
    gamma_u = abs(float(cfg["gamma_u"]))
    eps = float(cfg.get("epsilon", cfg.get("eps", 0.0)))
    sign_mode = str(cfg.get("sign_mode", "standard")).lower()

    margin = hdot + alpha * h
    if sign_mode == "standard":
        # Paper convention: safe h <= -gamma, unsafe h > gamma, CBF residual <= -epsilon.
        l_cbf = F.relu(eps + margin).mean()
    elif sign_mode == "flipped":
        # Legacy/debug convention retained for old checkpoints only.
        l_cbf = F.relu(eps - margin).mean()
    else:
        raise ValueError(f"Unknown sign_mode: {sign_mode}")

    safe = (y > 0).float()
    unsafe = (y <= 0).float()
    if sign_mode == "standard":
        l_cls = (safe * F.relu(gamma_s + h) + unsafe * F.relu(gamma_u - h)).mean()
    else:
        l_cls = (safe * F.relu(gamma_s - h) + unsafe * F.relu(h - gamma_u)).mean()

    w_cbf = float(cfg.get("w_cbf", 1.0))
    w_cls = float(cfg.get("w_cls", 1.0))
    total = w_cbf * l_cbf + w_cls * l_cls
    return {"total": total, "L_cbf": l_cbf, "L_cls": l_cls}
