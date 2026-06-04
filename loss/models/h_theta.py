from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    """Simple PointNet-style encoder for per-ray features."""

    def __init__(self, in_dim: int, point_feat_dim: int, global_feat_dim: int, activation: str) -> None:
        super().__init__()
        act = _make_activation(activation)
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, point_feat_dim),
            act.__class__(),
            nn.Linear(point_feat_dim, point_feat_dim),
            act.__class__(),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(point_feat_dim, global_feat_dim),
            act.__class__(),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: (B, R, C)
        feat = self.point_mlp(points)
        pooled = feat.max(dim=1).values
        return self.global_mlp(pooled)


class FlattenEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: str) -> None:
        super().__init__()
        act = _make_activation(activation)
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            act.__class__(),
            nn.Linear(out_dim, out_dim),
            act.__class__(),
        )

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        return self.net(flat)


class NeuralCBF(nn.Module):
    """h_theta(q_ego, o_hat) -> scalar."""

    def __init__(
        self,
        n_ego: int,
        rays: int,
        obs_dim_per_ray: int,
        encoder: str,
        point_feat_dim: int,
        global_feat_dim: int,
        hidden_dims: List[int],
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.n_ego = n_ego
        self.rays = rays
        self.obs_dim_per_ray = obs_dim_per_ray

        if encoder == "pointnet":
            self.encoder = PointNetEncoder(obs_dim_per_ray, point_feat_dim, global_feat_dim, activation)
            z_dim = global_feat_dim
        elif encoder == "flatten":
            self.encoder = FlattenEncoder(rays * obs_dim_per_ray, global_feat_dim, activation)
            z_dim = global_feat_dim
        else:
            raise ValueError(f"Unknown encoder: {encoder}")
        self.encoder_type = encoder

        act = _make_activation(activation)
        layers: list[nn.Module] = []
        in_dim = n_ego + z_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), act.__class__()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, q_ego: torch.Tensor, o_hat: torch.Tensor) -> torch.Tensor:
        # q_ego: (B, n_ego), o_hat: (B, R, C)
        if self.encoder_type == "pointnet":
            z = self.encoder(o_hat)
        else:
            z = self.encoder(o_hat.reshape(o_hat.shape[0], -1))
        x = torch.cat([q_ego, z], dim=-1)
        return self.head(x).squeeze(-1)


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")
