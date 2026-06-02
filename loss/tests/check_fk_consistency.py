from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

import numpy as np
import pybullet
import pybullet_utils.bullet_client as bc
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment.franka_panda import FrankaPanda
from loss.data.raycast_hdf5_dataset import load_raycast_metadata
from loss.models.raylink_g_phi import RayLinkMLPGPhi
from loss.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check TorchPandaFK against the PyBullet FK used by dataset sampling.")
    p.add_argument("--dataset_dir", default="dataset/neural_raycast_full")
    p.add_argument("--config", default="loss/configs/config_raylink_g_phi.yaml")
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--position_tol", type=float, default=1e-4)
    p.add_argument("--direction_tol", type=float, default=1e-4)
    p.add_argument("--rotation_tol", type=float, default=1e-4)
    return p.parse_args()


def quat_from_matrix(R: np.ndarray) -> Tuple[float, float, float, float]:
    m = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.asarray([x, y, z, w], dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    return tuple(float(v) for v in q)


def transform_from_pos_quat(p, pos: Iterable[float], quat: Iterable[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(p.getMatrixFromQuaternion(quat), dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def reset_base_from_transform(p, robot: FrankaPanda, T_W_B: np.ndarray) -> None:
    p.resetBasePositionAndOrientation(robot.robotId, T_W_B[:3, 3].tolist(), quat_from_matrix(T_W_B[:3, :3]))


def set_body_and_fingers(p, robot: FrankaPanda, q: np.ndarray, finger_open: float) -> None:
    robot.set_joint_position(robot.body_joints, q.astype(np.float64))
    for joint_id in robot.ee_joints:
        p.resetJointState(robot.robotId, int(joint_id), float(finger_open))
    p.performCollisionDetection()


def link_transform(p, robot: FrankaPanda, link_id: int) -> np.ndarray:
    if int(link_id) == -1:
        pos, quat = p.getBasePositionAndOrientation(robot.robotId)
    else:
        state = p.getLinkState(robot.robotId, int(link_id), computeForwardKinematics=True)
        pos, quat = state[4], state[5]
    return transform_from_pos_quat(p, pos, quat)


def pybullet_rays(p, robot: FrankaPanda, metadata: Dict) -> Tuple[np.ndarray, np.ndarray]:
    anchor_link_ids = [int(x) for x in metadata["anchor_link_ids"]]
    anchor_T_L_S = np.asarray(metadata["anchor_T_L_S"], dtype=np.float64)
    local_ray_dirs = np.asarray(metadata["local_ray_dirs"], dtype=np.float64)
    if local_ray_dirs.ndim == 2:
        local_ray_dirs = np.repeat(local_ray_dirs[None, :, :], len(anchor_link_ids), axis=0)

    origins = []
    dirs = []
    for anchor_idx, link_id in enumerate(anchor_link_ids):
        T_W_L = link_transform(p, robot, link_id)
        T_W_S = T_W_L @ anchor_T_L_S[anchor_idx]
        R_W_S = T_W_S[:3, :3]
        ray_dirs = (R_W_S @ local_ray_dirs[anchor_idx].T).T
        ray_dirs = ray_dirs / (np.linalg.norm(ray_dirs, axis=1, keepdims=True) + 1e-12)
        origins.append(np.repeat(T_W_S[:3, 3][None, :], ray_dirs.shape[0], axis=0))
        dirs.append(ray_dirs)
    return np.concatenate(origins, axis=0), np.concatenate(dirs, axis=0)


def pybullet_obs_link_poses(p, robot: FrankaPanda, link_ids: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
    transforms = np.stack([link_transform(p, robot, int(link_id)) for link_id in link_ids], axis=0)
    return transforms[:, :3, 3], transforms[:, :3, :3]


def check_gradients(
    model: RayLinkMLPGPhi,
    q_low: np.ndarray,
    q_high: np.ndarray,
    device: torch.device,
) -> Dict[str, Union[float, bool]]:
    mid = 0.5 * (q_low + q_high)
    q_ego = torch.tensor(mid[None, :], dtype=torch.float32, device=device, requires_grad=True)
    q_obs = torch.tensor((mid * 0.9)[None, :], dtype=torch.float32, device=device, requires_grad=True)
    geom = model.compute_geometry(q_ego, q_obs)
    scalar = (
        geom["ray_origins_W"].sum()
        + geom["ray_dirs_W"].sum()
        + geom["obs_link_pos_W"].sum()
        + geom["obs_link_rot_W"].sum()
    )
    scalar.backward()
    ego_grad = q_ego.grad.detach()
    obs_grad = q_obs.grad.detach()
    return {
        "ego_grad_finite": bool(torch.isfinite(ego_grad).all().item()),
        "obs_grad_finite": bool(torch.isfinite(obs_grad).all().item()),
        "ego_grad_norm": float(ego_grad.norm().item()),
        "obs_grad_norm": float(obs_grad.norm().item()),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    metadata = load_raycast_metadata(args.dataset_dir)
    model_cfg = cfg.get("model", {})
    finger_open = float(model_cfg.get("finger_open", 0.04))
    device = torch.device(args.device)

    model = RayLinkMLPGPhi(
        metadata,
        pair_hidden_dim=int(model_cfg.get("pair_hidden_dim", 128)),
        head_hidden_dims=list(model_cfg.get("head_hidden_dims", [128, 64])),
        link_embed_dim=int(model_cfg.get("link_embed_dim", 8)),
        anchor_embed_dim=int(model_cfg.get("anchor_embed_dim", 8)),
        activation=str(model_cfg.get("activation", "silu")),
        finger_open=finger_open,
    ).to(device)
    model.eval()

    robot_meta = metadata["robot_model"]
    q_low = np.asarray(robot_meta["joint_limits_low"], dtype=np.float64)
    q_high = np.asarray(robot_meta["joint_limits_high"], dtype=np.float64)
    obs_link_ids = [int(x) for x in robot_meta["obstacle_collision_link_ids"]]
    rng = np.random.default_rng(int(args.seed))

    client = bc.BulletClient(connection_mode=pybullet.DIRECT)
    try:
        ego = FrankaPanda(client)
        obs = FrankaPanda(client)
        reset_base_from_transform(client, ego, np.asarray(metadata["T_W_Bego"], dtype=np.float64))
        reset_base_from_transform(client, obs, np.asarray(metadata["T_W_Bobs"], dtype=np.float64))

        origin_errs = []
        dir_errs = []
        obs_pos_errs = []
        obs_rot_errs = []
        for _ in range(int(args.num_samples)):
            q_ego_np = rng.uniform(q_low, q_high)
            q_obs_np = rng.uniform(q_low, q_high)
            set_body_and_fingers(client, ego, q_ego_np, finger_open)
            set_body_and_fingers(client, obs, q_obs_np, finger_open)

            torch_geom = model.compute_geometry(
                torch.tensor(q_ego_np[None, :], dtype=torch.float32, device=device),
                torch.tensor(q_obs_np[None, :], dtype=torch.float32, device=device),
            )
            ray_o_t = torch_geom["ray_origins_W"][0].detach().cpu().numpy()
            ray_d_t = torch_geom["ray_dirs_W"][0].detach().cpu().numpy()
            obs_p_t = torch_geom["obs_link_pos_W"][0].detach().cpu().numpy()
            obs_R_t = torch_geom["obs_link_rot_W"][0].detach().cpu().numpy()

            ray_o_pb, ray_d_pb = pybullet_rays(client, ego, metadata)
            obs_p_pb, obs_R_pb = pybullet_obs_link_poses(client, obs, obs_link_ids)

            origin_errs.append(float(np.max(np.abs(ray_o_t - ray_o_pb))))
            dir_errs.append(float(np.max(np.abs(ray_d_t - ray_d_pb))))
            obs_pos_errs.append(float(np.max(np.abs(obs_p_t - obs_p_pb))))
            obs_rot_errs.append(float(np.max(np.abs(obs_R_t - obs_R_pb))))
    finally:
        client.disconnect()

    grad_summary = check_gradients(model, q_low, q_high, device)
    summary = {
        "num_samples": int(args.num_samples),
        "max_ray_origin_abs_error": float(max(origin_errs) if origin_errs else 0.0),
        "mean_ray_origin_abs_error": float(np.mean(origin_errs) if origin_errs else 0.0),
        "max_ray_dir_abs_error": float(max(dir_errs) if dir_errs else 0.0),
        "mean_ray_dir_abs_error": float(np.mean(dir_errs) if dir_errs else 0.0),
        "max_obs_link_pos_abs_error": float(max(obs_pos_errs) if obs_pos_errs else 0.0),
        "mean_obs_link_pos_abs_error": float(np.mean(obs_pos_errs) if obs_pos_errs else 0.0),
        "max_obs_link_rot_abs_error": float(max(obs_rot_errs) if obs_rot_errs else 0.0),
        "mean_obs_link_rot_abs_error": float(np.mean(obs_rot_errs) if obs_rot_errs else 0.0),
        **grad_summary,
    }
    summary["passed"] = bool(
        summary["max_ray_origin_abs_error"] <= float(args.position_tol)
        and summary["max_obs_link_pos_abs_error"] <= float(args.position_tol)
        and summary["max_ray_dir_abs_error"] <= float(args.direction_tol)
        and summary["max_obs_link_rot_abs_error"] <= float(args.rotation_tol)
        and summary["ego_grad_finite"]
        and summary["obs_grad_finite"]
        and summary["ego_grad_norm"] > 0.0
        and summary["obs_grad_norm"] > 0.0
    )
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
