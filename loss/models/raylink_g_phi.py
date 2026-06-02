from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from loss.models.torch_panda_fk import TorchPandaFK


def _activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def _mlp(dims: Iterable[int], activation: str) -> nn.Sequential:
    dims = list(int(x) for x in dims)
    layers: List[nn.Module] = []
    for idx in range(len(dims) - 1):
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            layers.append(_activation(activation))
    return nn.Sequential(*layers)


def _tensor4(value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    t = torch.tensor(value, dtype=dtype)
    if t.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transform, got shape {tuple(t.shape)}")
    return t


def _translation4(x: float, y: float, z: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    t = torch.eye(4, dtype=dtype)
    t[0, 3] = float(x)
    t[1, 3] = float(y)
    t[2, 3] = float(z)
    return t


class RayLinkMLPGPhi(nn.Module):
    """FK-aware MLP raycast surrogate for fixed Panda dual-arm geometry."""

    # PyBullet stores/resets the floating base at the base-link inertial frame.
    # In this Panda URDF, panda_link0 has inertial origin xyz="0 0 0.05".
    # PyBullet getLinkState(...)[4:6] reports link frames, so metadata base poses
    # must be shifted by -5 cm along local z before multiplying URDF FK.
    PANDA_LINK0_FROM_PYBULLET_BASE_Z = -0.05

    def __init__(
        self,
        metadata: Dict[str, Any],
        pair_hidden_dim: int = 128,
        head_hidden_dims: Optional[List[int]] = None,
        link_embed_dim: int = 8,
        anchor_embed_dim: int = 8,
        activation: str = "silu",
        finger_open: float = 0.04,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.r_max = float(metadata["r_max"])
        self.anchor_link_ids = [int(x) for x in metadata["anchor_link_ids"]]
        self.anchor_T_L_S = torch.tensor(metadata["anchor_T_L_S"], dtype=torch.float32)
        self.local_ray_dirs = torch.tensor(metadata["local_ray_dirs"], dtype=torch.float32)
        if self.local_ray_dirs.ndim != 3:
            raise ValueError("metadata local_ray_dirs must have shape [num_anchors, rays_per_anchor, 3]")
        self.num_anchors = int(self.local_ray_dirs.shape[0])
        self.rays_per_anchor = int(self.local_ray_dirs.shape[1])
        self.num_rays = int(self.num_anchors * self.rays_per_anchor)

        robot_meta = metadata.get("robot_model", {})
        obs_links = robot_meta.get("obstacle_collision_link_ids", metadata.get("obstacle_collision_link_ids", []))
        if not obs_links:
            raise KeyError("metadata must include robot_model.obstacle_collision_link_ids")
        self.obs_link_ids = [int(x) for x in obs_links]
        self.num_obs_links = len(self.obs_link_ids)

        self.register_buffer("T_W_Bego", _tensor4(metadata["T_W_Bego"]), persistent=False)
        self.register_buffer("T_W_Bobs", _tensor4(metadata["T_W_Bobs"]), persistent=False)
        self.register_buffer(
            "T_pybullet_base_link0",
            _translation4(0.0, 0.0, self.PANDA_LINK0_FROM_PYBULLET_BASE_Z),
            persistent=False,
        )
        self.register_buffer("anchor_T_L_S_buf", self.anchor_T_L_S, persistent=False)
        self.register_buffer("local_ray_dirs_buf", self.local_ray_dirs, persistent=False)
        anchor_index = torch.arange(self.num_anchors, dtype=torch.long).repeat_interleave(self.rays_per_anchor)
        self.register_buffer("ray_anchor_index", anchor_index, persistent=False)

        self.fk = TorchPandaFK(finger_open=finger_open)
        self.link_embedding = nn.Embedding(self.num_obs_links, int(link_embed_dim))
        self.anchor_embedding = nn.Embedding(self.num_anchors, int(anchor_embed_dim))

        feature_dim = 3 + 3 + 1 + 1 + 1 + int(link_embed_dim) + int(anchor_embed_dim)
        self.pair_mlp = _mlp([feature_dim, int(pair_hidden_dim), int(pair_hidden_dim), int(pair_hidden_dim)], activation)
        head_hidden_dims = head_hidden_dims or [128, 64]
        self.ray_head = _mlp([2 * int(pair_hidden_dim)] + list(head_hidden_dims) + [2], activation)

    def _world_from_base(self, T_W_B: torch.Tensor, T_B_L: torch.Tensor) -> torch.Tensor:
        T_W_pybullet_base = T_W_B.to(device=T_B_L.device, dtype=T_B_L.dtype).view(1, 1, 4, 4)
        T_pybullet_base_link0 = self.T_pybullet_base_link0.to(device=T_B_L.device, dtype=T_B_L.dtype).view(1, 1, 4, 4)
        return T_W_pybullet_base @ T_pybullet_base_link0 @ T_B_L

    def compute_geometry(self, q_ego: torch.Tensor, q_obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = int(q_ego.shape[0])
        T_Bego_L = self.fk(q_ego, self.anchor_link_ids)
        T_W_L = self._world_from_base(self.T_W_Bego, T_Bego_L)
        T_L_S = self.anchor_T_L_S_buf.to(device=q_ego.device, dtype=q_ego.dtype).view(1, self.num_anchors, 4, 4)
        T_W_S = T_W_L @ T_L_S

        origins = T_W_S[:, :, :3, 3]
        R_W_S = T_W_S[:, :, :3, :3]
        local_dirs = self.local_ray_dirs_buf.to(device=q_ego.device, dtype=q_ego.dtype)
        ray_dirs = torch.einsum("baij,akj->baki", R_W_S, local_dirs)
        ray_dirs = ray_dirs / (ray_dirs.norm(dim=-1, keepdim=True) + 1e-8)
        ray_origins = origins[:, :, None, :].expand(B, self.num_anchors, self.rays_per_anchor, 3)
        ray_origins = ray_origins.reshape(B, self.num_rays, 3)
        ray_dirs = ray_dirs.reshape(B, self.num_rays, 3)

        T_Bobs_L = self.fk(q_obs, self.obs_link_ids)
        T_W_obs = self._world_from_base(self.T_W_Bobs, T_Bobs_L)
        return {
            "ray_origins_W": ray_origins,
            "ray_dirs_W": ray_dirs,
            "obs_link_pos_W": T_W_obs[:, :, :3, 3],
            "obs_link_rot_W": T_W_obs[:, :, :3, :3],
        }

    def build_ray_link_features(self, geometry: Dict[str, torch.Tensor]) -> torch.Tensor:
        o = geometry["ray_origins_W"][:, :, None, :]
        d = geometry["ray_dirs_W"][:, :, None, :]
        p = geometry["obs_link_pos_W"][:, None, :, :]
        R = geometry["obs_link_rot_W"][:, None, :, :, :]

        rel_o = o - p
        link_to_ray = p - o
        R_t = R.transpose(-1, -2)
        o_local = torch.matmul(R_t, rel_o.unsqueeze(-1)).squeeze(-1)
        d_local = torch.matmul(R_t, d.expand_as(rel_o).unsqueeze(-1)).squeeze(-1)
        d_local = d_local / (d_local.norm(dim=-1, keepdim=True) + 1e-8)

        proj = (link_to_ray * d).sum(dim=-1, keepdim=True)
        perp = link_to_ray - proj * d
        perp_dist = perp.norm(dim=-1, keepdim=True)
        link_dist = link_to_ray.norm(dim=-1, keepdim=True)

        r_max = max(float(self.r_max), 1e-6)
        o_local = o_local / r_max
        proj = proj / r_max
        perp_dist = perp_dist / r_max
        link_dist = link_dist / r_max

        B, Rn, L, _ = o_local.shape
        link_idx = torch.arange(L, device=o_local.device, dtype=torch.long)
        link_emb = self.link_embedding(link_idx).view(1, 1, L, -1).expand(B, Rn, L, -1)
        anchor_idx = self.ray_anchor_index.to(device=o_local.device)
        anchor_emb = self.anchor_embedding(anchor_idx).view(1, Rn, 1, -1).expand(B, Rn, L, -1)
        return torch.cat([o_local, d_local, proj, perp_dist, link_dist, link_emb, anchor_emb], dim=-1)

    def forward(self, q_ego: torch.Tensor, q_obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        geometry = self.compute_geometry(q_ego, q_obs)
        phi = self.build_ray_link_features(geometry)
        pair_feat = self.pair_mlp(phi)
        max_feat = pair_feat.max(dim=2).values
        mean_feat = pair_feat.mean(dim=2)
        ray_feat = torch.cat([max_feat, mean_feat], dim=-1)
        out = self.ray_head(ray_feat)
        hit_logits = out[..., 0]
        depth_norm = torch.sigmoid(out[..., 1])
        return {
            "hit_logits": hit_logits,
            "depth_norm": depth_norm,
            "geometry": geometry,
        }
