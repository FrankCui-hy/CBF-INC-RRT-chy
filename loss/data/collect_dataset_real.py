from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from environment import ArmEnv
from loss.data.raycast_dummy import fibonacci_sphere_dirs
from loss.utils.config import load_config
from loss.utils.seed import set_seed


def _sample_uniform(low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return low + torch.rand_like(low) * (high - low)


def _random_walk_step(q: torch.Tensor, sigma: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    q_next = q + sigma * torch.randn_like(q)
    return torch.clamp(q_next, min=low, max=high)


def _link_pose_world(robot, link_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    ls = robot.p.getLinkState(robot.robotId, int(link_idx))
    pos = np.asarray(ls[4], dtype=np.float32)
    quat = ls[5]
    rot = np.asarray(robot.p.getMatrixFromQuaternion(quat), dtype=np.float32).reshape(3, 3)
    return pos, rot


def _raycast_obstacle_robot(
    env: ArmEnv,
    ray_origin: np.ndarray,  # (R,3)
    ray_dir_world: np.ndarray,  # (R,3), unit
    range_max: float,
    no_hit_fill: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert ray_origin.ndim == 2 and ray_origin.shape[1] == 3
    assert ray_dir_world.ndim == 2 and ray_dir_world.shape == ray_origin.shape

    ray_to = ray_origin + range_max * ray_dir_world
    raw = env.p.rayTestBatch(ray_origin.tolist(), ray_to.tolist(), numThreads=0)

    R = ray_origin.shape[0]
    p_gt = np.full((R, 3), no_hit_fill, dtype=np.float32)
    n_gt = np.zeros((R, 3), dtype=np.float32)
    m = np.zeros((R,), dtype=np.float32)
    hit_dist = np.full((R,), range_max, dtype=np.float32)

    valid_uid = env.obstacle_robot.robotId if env.obstacle_robot is not None else None
    for i, r in enumerate(raw):
        hit_uid = int(r[0])
        if valid_uid is None or hit_uid != int(valid_uid):
            continue
        hit_fraction = float(r[2])
        hit_pos = np.asarray(r[3], dtype=np.float32)
        hit_nrm = np.asarray(r[4], dtype=np.float32)
        nrm_norm = np.linalg.norm(hit_nrm)
        if nrm_norm > 1e-8:
            hit_nrm = hit_nrm / nrm_norm
        p_gt[i] = hit_pos
        n_gt[i] = hit_nrm
        m[i] = 1.0
        hit_dist[i] = np.clip(hit_fraction, 0.0, 1.0) * range_max

    return p_gt, n_gt, m, hit_dist


def _cone_dirs_local(num_rays: int, max_angle_deg: float) -> np.ndarray:
    """Generate local-frame unit rays within a cone around +Z."""
    max_angle_rad = np.deg2rad(max(1e-3, float(max_angle_deg)))
    cos_max = float(np.cos(max_angle_rad))
    idx = np.arange(num_rays, dtype=np.float32)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden * idx
    u = (idx + 0.5) / float(num_rays)
    cos_phi = 1.0 - u * (1.0 - cos_max)
    sin_phi = np.sqrt(np.clip(1.0 - cos_phi * cos_phi, 0.0, 1.0))
    x = np.cos(theta) * sin_phi
    y = np.sin(theta) * sin_phi
    z = cos_phi
    d = np.stack([x, y, z], axis=1).astype(np.float32)
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
    return d


def _robot_center_world(robot) -> np.ndarray:
    """Approximate obstacle robot center by averaging base and body-link positions."""
    pts = []
    base_pos, _ = robot.p.getBasePositionAndOrientation(robot.robotId)
    pts.append(np.asarray(base_pos, dtype=np.float32))
    for link in robot.body_joints:
        ls = robot.p.getLinkState(robot.robotId, int(link))
        pts.append(np.asarray(ls[4], dtype=np.float32))
    return np.mean(np.stack(pts, axis=0), axis=0).astype(np.float32)


def _cone_dirs_world_towards_target(template_cone_dirs: np.ndarray, forward_world: np.ndarray) -> np.ndarray:
    """Rotate +Z cone template to world forward direction."""
    fwd = np.asarray(forward_world, dtype=np.float32)
    fwd = fwd / (np.linalg.norm(fwd) + 1e-8)

    # Pick a stable helper axis not parallel to forward.
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(fwd, helper))) > 0.95:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = np.cross(helper, fwd)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    y_axis = np.cross(fwd, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

    # Local template uses +Z as cone axis.
    R = np.stack([x_axis, y_axis, fwd], axis=1).astype(np.float32)  # (3,3)
    dirs = (R @ template_cone_dirs.T).T
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
    return dirs.astype(np.float32)


def _joint_limits_from_cfg_or_robot(cfg: Dict[str, Any], robot, which: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    key = f"q_limits_{which}"
    if key in cfg["system"]:
        low = torch.tensor(cfg["system"][key]["low"], dtype=torch.float32, device=device)
        high = torch.tensor(cfg["system"][key]["high"], dtype=torch.float32, device=device)
        if low.numel() == robot.body_dim and high.numel() == robot.body_dim:
            return low, high

    br = np.asarray(robot.body_range, dtype=np.float32)
    low = torch.tensor(br[:, 0], dtype=torch.float32, device=device)
    high = torch.tensor(br[:, 1], dtype=torch.float32, device=device)
    return low, high


def build_episode_samples_real(cfg: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    sys_cfg = cfg["system"]
    data_cfg = cfg["data"]
    dt = float(sys_cfg["dt"])
    R = int(sys_cfg["rays"])

    range_max = float(data_cfg["ray"]["range_max"])
    no_hit_fill = float(data_cfg["ray"].get("no_hit_point_fill", 0.0))
    safe_dist = float(data_cfg["safe_dist"])
    unsafe_dist = float(data_cfg["unsafe_dist"])

    num_episodes = int(data_cfg["n_episodes"])
    T = int(data_cfg["episode_len"])
    ratio_dynamic = float(data_cfg["dynamic_episode_ratio"])
    sigma_dyn = float(data_cfg["sigma_dynamic"])
    sigma_static_obs = float(data_cfg["sigma_static_obs"])
    sigma_static_ego = float(data_cfg.get("sigma_static_ego", 0.0))

    real_cfg = data_cfg.get("real", {})
    robot_name = str(real_cfg.get("robot_name", "panda"))
    obstacle_robot_name = str(real_cfg.get("obstacle_robot_name", "panda"))
    obstacle_traj_path = real_cfg.get("obstacle_traj_path", None)
    gui = bool(real_cfg.get("gui", False))
    sensor_link_mode = str(real_cfg.get("sensor_link_mode", "ee"))
    ray_mode = str(real_cfg.get("ray_mode", data_cfg.get("ray_mode", "sphere"))).lower()
    cone_half_angle_deg = float(real_cfg.get("cone_half_angle_deg", data_cfg.get("cone_half_angle_deg", 35.0)))
    obstacle_base_pos = tuple(real_cfg.get("obstacle_robot_base_pos", (0.3, 0.0, 0.0)))
    obstacle_base_orn = tuple(real_cfg.get("obstacle_robot_base_orn", (0.0, 0.0, 0.0, 1.0)))
    near_episode_ratio = float(real_cfg.get("near_episode_ratio", 0.0))
    near_obstacle_base_pos = tuple(real_cfg.get("near_obstacle_base_pos", obstacle_base_pos))
    near_obstacle_base_pos_jitter = tuple(real_cfg.get("near_obstacle_base_pos_jitter", (0.0, 0.0, 0.0)))
    near_obstacle_base_orn = tuple(real_cfg.get("near_obstacle_base_orn", obstacle_base_orn))

    env = ArmEnv(
        [robot_name],
        GUI=gui,
        config_file="",
        obstacle_robot_name=obstacle_robot_name,
        obstacle_traj_path=obstacle_traj_path,
        obstacle_robot_base_pos=obstacle_base_pos,
        obstacle_robot_base_orn=obstacle_base_orn,
    )
    ego_robot = env.robot_list[0]
    obs_robot = env.obstacle_robot
    if obs_robot is None:
        raise RuntimeError("Real sampler requires obstacle_robot_name to be set.")

    n_ego = int(ego_robot.body_dim)
    n_obs = int(obs_robot.body_dim)

    ql_e_low, ql_e_high = _joint_limits_from_cfg_or_robot(cfg, ego_robot, "ego", device)
    ql_o_low, ql_o_high = _joint_limits_from_cfg_or_robot(cfg, obs_robot, "obs", device)
    assert ql_e_low.numel() == n_ego and ql_o_low.numel() == n_obs

    if ray_mode == "cone":
        ray_dirs_local = _cone_dirs_local(R, max_angle_deg=cone_half_angle_deg)
    elif ray_mode == "sphere":
        ray_dirs_local = fibonacci_sphere_dirs(R, device=torch.device("cpu")).cpu().numpy().astype(np.float32)
    else:
        raise ValueError(f"Unsupported ray_mode={ray_mode}. Use 'sphere' or 'cone'.")
    if sensor_link_mode == "mid":
        sensor_link = int(ego_robot.body_joints[len(ego_robot.body_joints) // 2])
    else:
        sensor_link = int(ego_robot.body_joints[-1])

    keys = [
        "q_ego",
        "qdot_ego",
        "q_obs",
        "qdot_obs",
        "p_gt",
        "n_gt",
        "m",
        "ray_origin",
        "ray_dir",
        "y",
        "episode_type",
    ]
    buf: Dict[str, list[torch.Tensor]] = {k: [] for k in keys}

    for ep in range(num_episodes):
        is_dynamic = ep < int(num_episodes * ratio_dynamic)
        is_near = ep < int(num_episodes * near_episode_ratio)

        if is_near:
            jitter = np.random.uniform(low=-1.0, high=1.0, size=(3,)).astype(np.float32) * np.asarray(
                near_obstacle_base_pos_jitter, dtype=np.float32
            )
            base_pos_ep = tuple((np.asarray(near_obstacle_base_pos, dtype=np.float32) + jitter).tolist())
            base_orn_ep = near_obstacle_base_orn
        else:
            base_pos_ep = obstacle_base_pos
            base_orn_ep = obstacle_base_orn
        env.p.resetBasePositionAndOrientation(obs_robot.robotId, base_pos_ep, base_orn_ep)
        env.p.performCollisionDetection()

        q_ego = _sample_uniform(ql_e_low, ql_e_high)
        q_obs = _sample_uniform(ql_o_low, ql_o_high)

        if not is_dynamic:
            q_ego = _sample_uniform(ql_e_low, ql_e_high)

        for _ in range(T):
            if is_dynamic:
                q_ego_next = _random_walk_step(q_ego, sigma_dyn, ql_e_low, ql_e_high)
                q_obs_next = _random_walk_step(q_obs, sigma_dyn, ql_o_low, ql_o_high)
            else:
                q_ego_next = _random_walk_step(q_ego, sigma_static_ego, ql_e_low, ql_e_high)
                q_obs_next = _random_walk_step(q_obs, sigma_static_obs, ql_o_low, ql_o_high)

            qdot_ego = (q_ego_next - q_ego) / dt
            qdot_obs = (q_obs_next - q_obs) / dt

            ego_robot.set_joint_position(ego_robot.body_joints, q_ego.detach().cpu().numpy())
            obs_robot.set_joint_position(obs_robot.body_joints, q_obs.detach().cpu().numpy())

            origin, rot = _link_pose_world(ego_robot, sensor_link)
            ray_origin = np.repeat(origin[None, :], R, axis=0).astype(np.float32)
            if ray_mode == "cone":
                obs_center = _robot_center_world(obs_robot)
                fwd = obs_center - origin
                ray_dir_world = _cone_dirs_world_towards_target(ray_dirs_local, fwd)
            else:
                ray_dir_world = (rot @ ray_dirs_local.T).T
                ray_dir_world = ray_dir_world / (np.linalg.norm(ray_dir_world, axis=1, keepdims=True) + 1e-8)

            p_gt, n_gt, m, hit_dist = _raycast_obstacle_robot(
                env=env,
                ray_origin=ray_origin,
                ray_dir_world=ray_dir_world,
                range_max=range_max,
                no_hit_fill=no_hit_fill,
            )

            min_dist = float(hit_dist.min())
            if min_dist <= unsafe_dist:
                y = -1.0
            elif min_dist >= safe_dist:
                y = 1.0
            else:
                y = 1.0

            buf["q_ego"].append(q_ego.detach().cpu().clone())
            buf["qdot_ego"].append(qdot_ego.detach().cpu().clone())
            buf["q_obs"].append(q_obs.detach().cpu().clone())
            buf["qdot_obs"].append(qdot_obs.detach().cpu().clone())
            buf["p_gt"].append(torch.from_numpy(p_gt))
            buf["n_gt"].append(torch.from_numpy(n_gt))
            buf["m"].append(torch.from_numpy(m))
            buf["ray_origin"].append(torch.from_numpy(ray_origin))
            buf["ray_dir"].append(torch.from_numpy(ray_dir_world.astype(np.float32)))
            buf["y"].append(torch.tensor(y, dtype=torch.float32))
            buf["episode_type"].append(torch.tensor(1.0 if is_dynamic else 0.0, dtype=torch.float32))

            q_ego = q_ego_next
            q_obs = q_obs_next

    out = {k: torch.stack(v, dim=0) for k, v in buf.items()}
    out["meta"] = {
        "n_ego": n_ego,
        "n_obs": n_obs,
        "rays": R,
        "dt": dt,
        "num_episodes": num_episodes,
        "episode_len": T,
        "dynamic_episode_ratio": ratio_dynamic,
        "backend": "pybullet_raycast",
        "sensor_link_mode": sensor_link_mode,
        "ray_mode": ray_mode,
        "cone_axis_mode": "target_tracking" if ray_mode == "cone" else "sensor_frame",
        "cone_half_angle_deg": cone_half_angle_deg,
        "obstacle_robot_base_pos": obstacle_base_pos,
        "near_episode_ratio": near_episode_ratio,
        "near_obstacle_base_pos": near_obstacle_base_pos,
    }
    return out


def save_dataset(payload: Dict[str, torch.Tensor], cfg: Dict[str, Any], output_path: str | None) -> Path:
    dataset_path = Path(output_path) if output_path else Path(cfg["paths"]["dataset_path"])
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dataset_path)
    return dataset_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect real pybullet raycast dataset for loss/*.py pipelines.")
    p.add_argument("--config", type=str, default="loss/configs/config.yaml")
    p.add_argument("--output_path", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    payload = build_episode_samples_real(cfg, device=torch.device("cpu"))
    path = save_dataset(payload, cfg, args.output_path or None)

    n = payload["q_ego"].shape[0]
    dyn_ratio = payload["episode_type"].float().mean().item()
    safe_ratio = (payload["y"] > 0).float().mean().item()
    print(f"[collect_dataset_real] saved: {path}")
    print(f"[collect_dataset_real] samples={n}, dynamic_ratio={dyn_ratio:.3f}, safe_ratio={safe_ratio:.3f}")


if __name__ == "__main__":
    main()
