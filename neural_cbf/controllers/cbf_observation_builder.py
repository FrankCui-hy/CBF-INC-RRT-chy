import math
from typing import Any, Dict, Optional

import numpy as np
import torch

from neural_cbf.tools.metadata_fingerprint import compute_metadata_fingerprint


def _translation_from_T(T: Any):
    if T is None:
        return None
    return [float(T[0][3]), float(T[1][3]), float(T[2][3])]


def _quat_xyzw_from_T(T: Any):
    if T is None:
        return None
    r00 = float(T[0][0])
    r01 = float(T[0][1])
    r02 = float(T[0][2])
    r10 = float(T[1][0])
    r11 = float(T[1][1])
    r12 = float(T[1][2])
    r20 = float(T[2][0])
    r21 = float(T[2][1])
    r22 = float(T[2][2])
    trace = r00 + r11 + r22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r21 - r12) / s
        qy = (r02 - r20) / s
        qz = (r10 - r01) / s
    elif r00 > r11 and r00 > r22:
        s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        qw = (r21 - r12) / s
        qx = 0.25 * s
        qy = (r01 + r10) / s
        qz = (r02 + r20) / s
    elif r11 > r22:
        s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        qw = (r02 - r20) / s
        qx = (r01 + r10) / s
        qy = 0.25 * s
        qz = (r12 + r21) / s
    else:
        s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        qw = (r10 - r01) / s
        qx = (r02 + r20) / s
        qy = (r12 + r21) / s
        qz = 0.25 * s
    return [float(qx), float(qy), float(qz), float(qw)]


def raylink_cbf_metadata(gphi_metadata: Dict[str, Any], gphi_ckpt: str = "", add_normal: bool = False, point_dim: int = 3) -> Dict[str, Any]:
    robot_meta = gphi_metadata.get("robot_model", {})
    num_anchors = int(gphi_metadata.get("num_anchors", len(gphi_metadata.get("anchor_link_ids", []))))
    rays_per_anchor = int(gphi_metadata.get("num_rays_per_anchor", 0))
    num_rays = int(gphi_metadata.get("num_rays_total", num_anchors * rays_per_anchor))
    T_W_Bego = gphi_metadata.get("T_W_Bego")
    T_W_Bobs = gphi_metadata.get("T_W_Bobs")
    return {
        "schema": "raylink_cbf_v1",
        "robot": "dual_panda",
        "ego_base_pos": gphi_metadata.get("ego_base_pos", _translation_from_T(T_W_Bego)),
        "obs_base_pos": gphi_metadata.get("obs_base_pos", _translation_from_T(T_W_Bobs)),
        "obs_base_quat": gphi_metadata.get("obs_base_quat", _quat_xyzw_from_T(T_W_Bobs)),
        "T_W_Bego": T_W_Bego,
        "T_W_Bobs": T_W_Bobs,
        "anchor_links": gphi_metadata.get("anchor_link_ids", []),
        "anchor_link_names": gphi_metadata.get("anchor_link_names", []),
        "num_anchors": num_anchors,
        "rays_per_anchor": rays_per_anchor,
        "num_rays": num_rays,
        "ray_direction_type": "fibonacci" if gphi_metadata.get("shared_local_ray_dirs", False) else "metadata",
        "ray_seed": gphi_metadata.get("random_seed", None),
        "ray_order": gphi_metadata.get("ray_ordering_rule", "anchor_major"),
        "r_max": float(gphi_metadata.get("r_max", 5.0)),
        "add_normal": bool(add_normal),
        "point_dim": int(point_dim),
        "gphi_ckpt": str(gphi_ckpt or ""),
        "ego_collision_link_ids": robot_meta.get("ego_collision_link_ids", []),
        "obstacle_collision_link_ids": robot_meta.get("obstacle_collision_link_ids", []),
    }


class CBFObservationBuilder:
    def __init__(
        self,
        mode: str,
        q_dim: int,
        obs_dim: int,
        aux_dim: int,
        g_phi=None,
        gphi_metadata: Optional[Dict[str, Any]] = None,
        r_max: float = 5.0,
        hit_threshold: float = 0.5,
        hit_temp: float = 0.1,
        add_normal: bool = False,
        point_dim: int = 3,
        device=None,
        raycast_env=None,
        raycast_ego_robot=None,
        raycast_obstacle_robot=None,
    ):
        self.mode = str(mode)
        self.q_dim = int(q_dim)
        self.obs_dim = int(obs_dim)
        self.aux_dim = int(aux_dim)
        self.g_phi = g_phi
        self.gphi_metadata = gphi_metadata or {}
        self.r_max = float(r_max)
        self.hit_threshold = float(hit_threshold)
        self.hit_temp = float(hit_temp)
        self.add_normal = bool(add_normal)
        self.point_dim = int(point_dim)
        self.device = device
        self.raycast_env = raycast_env
        self.raycast_ego_robot = raycast_ego_robot
        self.raycast_obstacle_robot = raycast_obstacle_robot

        if self.mode not in ("legacy_oracle", "gphi", "raylink_oracle"):
            raise ValueError(f"Unknown CBF observation mode: {self.mode}")
        if self.mode in ("gphi", "raylink_oracle"):
            if self.g_phi is None:
                raise ValueError(f"cbf_obs_mode='{self.mode}' requires RayLink metadata/FK from a g_phi checkpoint.")
            if self.add_normal:
                raise ValueError(f"cbf_obs_mode='{self.mode}' is point-only. Normal prediction is not implemented.")
            if self.point_dim != 3:
                raise ValueError(f"cbf_obs_mode='{self.mode}' currently requires point_dim=3.")
        if self.mode == "gphi":
            if self.hit_temp <= 0:
                raise ValueError("gphi hit_temp must be positive.")

    def _module_device(self) -> Optional[torch.device]:
        if self.g_phi is None:
            return None
        try:
            return next(self.g_phi.parameters()).device
        except StopIteration:
            try:
                return next(self.g_phi.buffers()).device
            except StopIteration:
                return None

    def _ensure_module_device(self, device: torch.device) -> None:
        if self.g_phi is None:
            return
        module_device = self._module_device()
        if module_device is not None and module_device != device:
            self.g_phi.to(device)

    def build(
        self,
        datax: Optional[torch.Tensor] = None,
        q_ego: Optional[torch.Tensor] = None,
        q_obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.mode == "legacy_oracle":
            if datax is None:
                raise ValueError("legacy_oracle observation mode requires datax.")
            if self.aux_dim <= 0:
                return datax[:, self.q_dim :]
            return datax[:, self.q_dim : -self.aux_dim]

        if q_ego is None or q_obs is None:
            raise ValueError(f"{self.mode} observation mode requires q_ego and q_obs.")
        if q_ego.ndim != 2 or q_obs.ndim != 2:
            raise ValueError(f"Expected q_ego/q_obs to be rank-2, got {tuple(q_ego.shape)} and {tuple(q_obs.shape)}.")

        self._ensure_module_device(q_ego.device)
        if self.mode == "raylink_oracle":
            return self._build_raylink_oracle_points(q_ego, q_obs)

        pred = self.g_phi(q_ego, q_obs)
        hit_logits = pred["hit_logits"]
        depth_hit = pred["depth_norm"] * float(self.r_max)
        geometry = pred["geometry"]
        ray_origins = geometry["ray_origins_W"]
        ray_dirs = geometry["ray_dirs_W"]

        threshold = min(max(float(self.hit_threshold), 1e-6), 1.0 - 1e-6)
        logit_threshold = math.log(threshold / (1.0 - threshold))
        w = torch.sigmoid((hit_logits - logit_threshold) / float(self.hit_temp))
        depth_eff = w * depth_hit + (1.0 - w) * float(self.r_max)
        points_w = ray_origins + depth_eff.unsqueeze(-1) * ray_dirs
        return points_w.reshape(points_w.shape[0], -1)

    def _raycast_handles(self):
        env = self.raycast_env
        if env is None or not hasattr(env, "p"):
            raise ValueError("cbf_obs_mode='raylink_oracle' requires an ArmEnv/PyBullet environment.")
        obstacle_robot = getattr(env, "obstacle_robot", None) or self.raycast_obstacle_robot
        if obstacle_robot is None:
            raise ValueError("cbf_obs_mode='raylink_oracle' requires env.obstacle_robot for obstacle-only raycast.")
        robots = getattr(env, "robot_list", [])
        ego_robot = robots[0] if len(robots) > 0 else self.raycast_ego_robot
        return env, env.p, ego_robot, obstacle_robot

    def _metadata_base_pose(self, prefix: str):
        if prefix == "ego":
            pos = self.gphi_metadata.get("ego_base_pos", _translation_from_T(self.gphi_metadata.get("T_W_Bego")))
            quat = self.gphi_metadata.get("ego_base_quat", _quat_xyzw_from_T(self.gphi_metadata.get("T_W_Bego")))
        elif prefix == "obs":
            pos = self.gphi_metadata.get("obs_base_pos", _translation_from_T(self.gphi_metadata.get("T_W_Bobs")))
            quat = self.gphi_metadata.get("obs_base_quat", _quat_xyzw_from_T(self.gphi_metadata.get("T_W_Bobs")))
        else:
            raise ValueError(f"Unknown base pose prefix: {prefix}")
        if pos is None or quat is None:
            raise ValueError(f"RayLink metadata is missing {prefix} base pose/quaternion.")
        return tuple(float(x) for x in pos), tuple(float(x) for x in quat)

    def _set_obstacle_joint_state(self, obstacle_robot, q_obs_b: torch.Tensor) -> None:
        q_np = q_obs_b.detach().cpu().numpy().astype(np.float64).reshape(-1)
        joints = list(obstacle_robot.body_joints)
        if q_np.shape[0] != len(joints):
            raise ValueError(
                "cbf_obs_mode='raylink_oracle' expected q_obs width "
                f"{len(joints)}, got {q_np.shape[0]}."
            )
        obstacle_robot.set_joint_position(joints, q_np)

    def _temporarily_move_body(self, p_client, body_id: Optional[int], pos=(1000.0, 1000.0, 1000.0)):
        if body_id is None:
            return None
        try:
            old_pos, old_orn = p_client.getBasePositionAndOrientation(int(body_id))
            p_client.resetBasePositionAndOrientation(int(body_id), pos, (0.0, 0.0, 0.0, 1.0))
            return int(body_id), old_pos, old_orn
        except Exception:
            return None

    def _restore_moved_body(self, p_client, record) -> None:
        if record is None:
            return
        body_id, old_pos, old_orn = record
        try:
            p_client.resetBasePositionAndOrientation(int(body_id), old_pos, old_orn)
        except Exception:
            pass

    def _oracle_raycast_obstacle(
        self,
        q_obs: torch.Tensor,
        ray_origins: torch.Tensor,
        ray_dirs: torch.Tensor,
    ):
        env, p_client, ego_robot, obstacle_robot = self._raycast_handles()
        obs_base_pos, obs_base_quat = self._metadata_base_pose("obs")
        device = ray_origins.device
        dtype = ray_origins.dtype
        B = int(ray_origins.shape[0])
        R = int(ray_origins.shape[1])
        r_max = float(self.r_max)
        hit_mask = torch.zeros((B, R), device=device, dtype=dtype)
        depth = torch.full((B, R), r_max, device=device, dtype=dtype)

        origins_cpu = ray_origins.detach().cpu().float()
        dirs_cpu = ray_dirs.detach().cpu().float()
        ray_ends_cpu = origins_cpu + r_max * dirs_cpu

        ego_body_id = int(getattr(ego_robot, "robotId", -1)) if ego_robot is not None else None
        plane_body_id = int(getattr(env, "plane", -1)) if hasattr(env, "plane") else None
        obstacle_body_id = int(obstacle_robot.robotId)

        for b in range(B):
            p_client.resetBasePositionAndOrientation(obstacle_body_id, obs_base_pos, obs_base_quat)
            self._set_obstacle_joint_state(obstacle_robot, q_obs[b])

            moved_ego = self._temporarily_move_body(p_client, ego_body_id)
            moved_plane = self._temporarily_move_body(p_client, plane_body_id, pos=(0.0, 0.0, -1000.0))
            try:
                raw = p_client.rayTestBatch(
                    origins_cpu[b].tolist(),
                    ray_ends_cpu[b].tolist(),
                    numThreads=0,
                )
            finally:
                self._restore_moved_body(p_client, moved_plane)
                self._restore_moved_body(p_client, moved_ego)

            depth_b = depth[b]
            hit_b = hit_mask[b]
            for i, result in enumerate(raw):
                hit_uid = int(result[0])
                if hit_uid != obstacle_body_id:
                    continue
                dist = float(result[2]) * r_max
                if 0.0 <= dist < r_max:
                    hit_b[i] = 1.0
                    depth_b[i] = dist
        return hit_mask, depth

    def _build_raylink_oracle_points(self, q_ego: torch.Tensor, q_obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            geometry = self.g_phi.compute_geometry(q_ego, q_obs)
            ray_origins = geometry["ray_origins_W"]
            ray_dirs = geometry["ray_dirs_W"]
            _, depth = self._oracle_raycast_obstacle(q_obs, ray_origins, ray_dirs)
            points_w = ray_origins + depth.unsqueeze(-1) * ray_dirs
        return points_w.reshape(points_w.shape[0], -1)
