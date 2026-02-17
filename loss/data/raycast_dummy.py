from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class RaycastConfig:
    range_max: float = 2.5
    no_hit_point_fill: float = 0.0


def fibonacci_sphere_dirs(num_rays: int, device: torch.device | None = None) -> torch.Tensor:
    """Generate approximately uniform unit vectors on a sphere.

    Returns:
        dirs: (R, 3) unit vectors.
    """
    i = torch.arange(num_rays, dtype=torch.float32, device=device)
    phi = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0, device=device)))
    y = 1.0 - (2.0 * i + 1.0) / num_rays
    radius = torch.sqrt(torch.clamp(1.0 - y * y, min=0.0))
    theta = phi * i
    x = torch.cos(theta) * radius
    z = torch.sin(theta) * radius
    dirs = torch.stack([x, y, z], dim=-1)
    return torch.nn.functional.normalize(dirs, dim=-1)


def ego_sensor_origin(q_ego: torch.Tensor) -> torch.Tensor:
    """Dummy differentiable sensor origin in world frame from ego joint angles.

    Args:
        q_ego: (n,)
    Returns:
        origin: (3,)
    """
    x = 0.4 + 0.22 * torch.sin(q_ego[0]) + 0.08 * torch.sin(q_ego[3])
    y = -0.1 + 0.20 * torch.sin(q_ego[1])
    z = 0.55 + 0.18 * torch.cos(q_ego[2])
    return torch.stack([x, y, z], dim=0)


def obstacle_spheres_from_q(q_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map obstacle arm joints to a compact set of moving spheres.

    Args:
        q_obs: (m,)
    Returns:
        centers: (K, 3)
        radii: (K,)
    """
    c0 = torch.stack(
        [
            0.65 + 0.25 * torch.sin(q_obs[0]),
            0.15 + 0.20 * torch.cos(q_obs[1]),
            0.50 + 0.18 * torch.sin(q_obs[2]),
        ]
    )
    c1 = torch.stack(
        [
            0.55 + 0.18 * torch.sin(q_obs[3]),
            -0.20 + 0.25 * torch.sin(q_obs[4]),
            0.62 + 0.12 * torch.cos(q_obs[5]),
        ]
    )
    c2 = torch.stack(
        [
            0.48 + 0.16 * torch.sin(q_obs[6]),
            0.30 + 0.12 * torch.cos(q_obs[0]),
            0.42 + 0.14 * torch.sin(q_obs[1]),
        ]
    )
    centers = torch.stack([c0, c1, c2], dim=0)
    radii = torch.tensor([0.13, 0.10, 0.09], dtype=q_obs.dtype, device=q_obs.device)
    return centers, radii


def raycast_spheres(
    ray_origins: torch.Tensor,
    ray_dirs: torch.Tensor,
    sphere_centers: torch.Tensor,
    sphere_radii: torch.Tensor,
    cfg: RaycastConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ray-sphere intersection for dummy non-differentiable sensor supervision.

    Args:
        ray_origins: (R, 3)
        ray_dirs: (R, 3), unit vectors
        sphere_centers: (K, 3)
        sphere_radii: (K,)
    Returns:
        p_gt: (R, 3)
        n_gt: (R, 3)
        m: (R,) in {0,1}
        hit_dist: (R,) distance along ray, range_max when no-hit
    """
    R = ray_origins.shape[0]
    K = sphere_centers.shape[0]

    o = ray_origins[:, None, :].expand(R, K, 3)
    d = ray_dirs[:, None, :].expand(R, K, 3)
    c = sphere_centers[None, :, :].expand(R, K, 3)
    r = sphere_radii[None, :].expand(R, K)

    oc = o - c
    a = (d * d).sum(dim=-1)
    b = 2.0 * (oc * d).sum(dim=-1)
    cterm = (oc * oc).sum(dim=-1) - r * r
    disc = b * b - 4.0 * a * cterm

    valid = disc >= 0.0
    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
    t0 = (-b - sqrt_disc) / (2.0 * a)
    t1 = (-b + sqrt_disc) / (2.0 * a)

    big = torch.full_like(t0, 1e9)
    t0 = torch.where(valid & (t0 > 0.0), t0, big)
    t1 = torch.where(valid & (t1 > 0.0), t1, big)
    t = torch.minimum(t0, t1)

    t_hit, k_hit = t.min(dim=1)
    hit = t_hit < cfg.range_max

    p_hit = ray_origins + t_hit[:, None] * ray_dirs
    c_hit = sphere_centers[k_hit]
    n_hit = torch.nn.functional.normalize(p_hit - c_hit, dim=-1)

    p_no = torch.full_like(p_hit, cfg.no_hit_point_fill)
    n_no = torch.zeros_like(n_hit)
    p_gt = torch.where(hit[:, None], p_hit, p_no)
    n_gt = torch.where(hit[:, None], n_hit, n_no)
    m = hit.to(ray_origins.dtype)
    dist = torch.where(hit, t_hit, torch.full_like(t_hit, cfg.range_max))
    return p_gt, n_gt, m, dist
