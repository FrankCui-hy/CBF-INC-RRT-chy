from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")


class SurrogateObservationNet(nn.Module):
    """g_phi: differentiable surrogate for raycast observation.

    Input:
        q_ego: (B, n_ego)
        q_obs: (B, n_obs)
    Output dict:
        p_hat: (B, R, 3)
        n_hat: (B, R, 3) normalized
        m_hat: (B, R, 1) or None
        o_hat: (B, R, 6 or 7)
    """

    def __init__(
        self,
        n_ego: int,
        n_obs: int,
        rays: int,
        hidden_dims: List[int],
        activation: str = "silu",
        predict_hit_prob: bool = True,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_ego = n_ego
        self.n_obs = n_obs
        self.rays = rays
        self.predict_hit_prob = predict_hit_prob
        self.norm_eps = norm_eps

        act = _make_activation(activation)
        layers: list[nn.Module] = []
        in_dim = n_ego + n_obs
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), act.__class__()])
            in_dim = h

        out_per_ray = 3 + 3 + (1 if predict_hit_prob else 0)
        out_dim = rays * out_per_ray
        layers.append(nn.Linear(in_dim, out_dim))
        self.backbone = nn.Sequential(*layers)

    def forward(self, q_ego: torch.Tensor, q_obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = torch.cat([q_ego, q_obs], dim=-1)
        raw = self.backbone(x)
        B = raw.shape[0]
        out_per_ray = 3 + 3 + (1 if self.predict_hit_prob else 0)
        raw = raw.view(B, self.rays, out_per_ray)

        p_hat = raw[..., 0:3]
        n_tilde = raw[..., 3:6]
        n_hat = n_tilde / (n_tilde.norm(dim=-1, keepdim=True) + self.norm_eps)

        if self.predict_hit_prob:
            m_logit = raw[..., 6:7]
            m_hat = torch.sigmoid(m_logit)
            o_hat = torch.cat([p_hat, n_hat, m_hat], dim=-1)
        else:
            m_hat = None
            o_hat = torch.cat([p_hat, n_hat], dim=-1)

        return {"p_hat": p_hat, "n_hat": n_hat, "m_hat": m_hat, "o_hat": o_hat}
