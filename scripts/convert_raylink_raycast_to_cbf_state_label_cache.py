#!/usr/bin/env python3
"""Convert a RayLink raycast HDF5 dataset into a CBF state-label cache.

The output intentionally discards ray observations. CBF training later rebuilds
observations online according to cbf_obs_mode:
  - gphi: frozen RayLinkMLPGPhi(q_ego, q_obs)
  - raylink_oracle: PyBullet oracle rays with the same RayLink metadata
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_csv_floats(text: Optional[str], n: int, name: str) -> Optional[Tuple[float, ...]]:
    if text is None or str(text).strip() == "":
        return None
    vals = tuple(float(x) for x in str(text).split(","))
    if len(vals) != int(n):
        raise ValueError(f"{name} must have {n} comma-separated floats, got {text!r}.")
    return vals


def translation_from_T(T: Any) -> Optional[Tuple[float, float, float]]:
    if T is None:
        return None
    return (float(T[0][3]), float(T[1][3]), float(T[2][3]))


def quat_xyzw_from_T(T: Any) -> Optional[Tuple[float, float, float, float]]:
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
    return (float(qx), float(qy), float(qz), float(qw))


def close_list(a: Iterable[Any], b: Iterable[Any], tol: float = 1e-6) -> bool:
    aa = list(a)
    bb = list(b)
    return len(aa) == len(bb) and all(abs(float(x) - float(y)) <= float(tol) for x, y in zip(aa, bb))


def compare_metadata(dataset_meta: Dict[str, Any], ckpt_meta: Dict[str, Any]) -> None:
    keys = ("r_max", "num_anchors", "num_rays_per_anchor", "num_rays_total")
    for key in keys:
        if key in dataset_meta and key in ckpt_meta and float(dataset_meta[key]) != float(ckpt_meta[key]):
            raise ValueError(f"RayLink metadata mismatch for {key}: dataset={dataset_meta[key]} ckpt={ckpt_meta[key]}")
    for key in ("anchor_link_ids", "anchor_T_L_S", "local_ray_dirs", "T_W_Bego", "T_W_Bobs"):
        if key not in dataset_meta or key not in ckpt_meta:
            continue
        ds_json = json.dumps(dataset_meta[key], sort_keys=True)
        ckpt_json = json.dumps(ckpt_meta[key], sort_keys=True)
        if ds_json != ckpt_json:
            raise ValueError(f"RayLink metadata mismatch for {key}. Refusing to mix dataset and g_phi checkpoint layouts.")


def load_torch_checkpoint_metadata(path: str) -> Dict[str, Any]:
    import torch

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if "metadata" not in ckpt:
        raise KeyError(f"g_phi checkpoint does not contain metadata: {path}")
    return ckpt["metadata"]


def read_split(hdf5_path: Path, d_safe: float, qdot_mode: str):
    import h5py
    import numpy as np
    import torch

    required = ("q_ego", "q_obs", "d_min", "collision")
    with h5py.File(hdf5_path, "r") as f:
        missing = [key for key in required if key not in f]
        if missing:
            raise KeyError(f"{hdf5_path} is missing required datasets: {missing}")
        q_ego = torch.from_numpy(f["q_ego"][:].astype(np.float32))
        q_obs = torch.from_numpy(f["q_obs"][:].astype(np.float32))
        d_min = torch.from_numpy(f["d_min"][:].astype(np.float32)).reshape(-1)
        collision = torch.from_numpy(f["collision"][:].astype(np.uint8)).bool().reshape(-1)

        if qdot_mode == "hdf5":
            if "qdot_obs" not in f:
                raise KeyError(f"{hdf5_path} does not contain qdot_obs but --qdot_mode hdf5 was requested.")
            qdot_obs = torch.from_numpy(f["qdot_obs"][:].astype(np.float32))
            traj_idx = torch.from_numpy(f["traj_idx"][:].astype(np.float32)).reshape(-1) if "traj_idx" in f else torch.zeros_like(d_min)
            step_idx = torch.from_numpy(f["step_idx"][:].astype(np.float32)).reshape(-1) if "step_idx" in f else torch.zeros_like(d_min)
        elif qdot_mode == "zeros":
            qdot_obs = torch.zeros_like(q_obs)
            traj_idx = torch.zeros_like(d_min)
            step_idx = torch.zeros_like(d_min)
        else:
            raise ValueError(f"Unknown qdot_mode={qdot_mode}.")

    if q_ego.ndim != 2 or q_obs.ndim != 2:
        raise ValueError(f"Expected q_ego/q_obs rank-2 in {hdf5_path}, got {tuple(q_ego.shape)} and {tuple(q_obs.shape)}.")
    if q_ego.shape[0] != q_obs.shape[0] or q_ego.shape[0] != d_min.shape[0] or q_ego.shape[0] != collision.shape[0]:
        raise ValueError(f"Split length mismatch in {hdf5_path}.")
    if qdot_obs.shape != q_obs.shape:
        raise ValueError(f"qdot_obs shape {tuple(qdot_obs.shape)} does not match q_obs shape {tuple(q_obs.shape)}.")

    unsafe_mask = collision
    safe_mask = torch.logical_and(d_min > float(d_safe), torch.logical_not(unsafe_mask))
    boundary_mask = torch.logical_not(torch.logical_or(safe_mask, unsafe_mask))
    return {
        "q_ego": q_ego,
        "q_obs": q_obs,
        "qdot_obs": qdot_obs,
        "traj_idx": traj_idx,
        "step_idx": step_idx,
        "safe_mask": safe_mask,
        "unsafe_mask": unsafe_mask,
        "boundary_mask": boundary_mask,
        "d_min": d_min,
        "collision": collision,
    }


def attach_jacobians(split: Dict[str, Any], dynamics_model: Any, chunk_size: int, split_name: str) -> Dict[str, Any]:
    import torch

    q_ego = split["q_ego"].float()
    n = int(q_ego.shape[0])
    num_sensors = len(dynamics_model.list_sensor)
    q_dim = int(dynamics_model.q_dims)
    J_P = torch.empty((n, num_sensors, 3, q_dim), dtype=torch.float32)
    J_R = torch.empty((n, num_sensors, 3, 3, q_dim), dtype=torch.float32)
    chunk_size = max(int(chunk_size), 1)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        jp, jr = dynamics_model.get_batch_jacobian(q_ego[start:end])
        J_P[start:end] = jp.float()
        J_R[start:end] = jr.float()
        print(f"[jacobian:{split_name}] {end}/{n}")
    out = {
        "q_ego": split["q_ego"],
        "q_obs": split["q_obs"],
        "qdot_obs": split["qdot_obs"],
        "traj_idx": split["traj_idx"],
        "step_idx": split["step_idx"],
        "safe_mask": split["safe_mask"],
        "unsafe_mask": split["unsafe_mask"],
        "boundary_mask": split["boundary_mask"],
        "J_P": J_P,
        "J_R": J_R,
    }
    return out


def build_dynamics_for_jacobian(args: argparse.Namespace, dataset_meta: Dict[str, Any], d_safe: float):
    import torch

    from environment import ArmEnv
    from neural_cbf.systems import ArmLidar

    ego_base_pos = parse_csv_floats(args.ego_base_pos, 3, "ego_base_pos") or translation_from_T(dataset_meta.get("T_W_Bego"))
    ego_base_quat = parse_csv_floats(args.ego_base_quat, 4, "ego_base_quat") or quat_xyzw_from_T(dataset_meta.get("T_W_Bego")) or (0.0, 0.0, 0.0, 1.0)
    obs_base_pos = parse_csv_floats(args.obs_base_pos, 3, "obs_base_pos") or translation_from_T(dataset_meta.get("T_W_Bobs"))
    obs_base_quat = parse_csv_floats(args.obs_base_quat, 4, "obs_base_quat") or quat_xyzw_from_T(dataset_meta.get("T_W_Bobs"))
    if ego_base_pos is None or obs_base_pos is None or obs_base_quat is None:
        raise ValueError("Base poses are missing. Pass --ego_base_pos, --obs_base_pos, and --obs_base_quat.")

    env = ArmEnv(
        [args.robot_name],
        GUI=False,
        config_file="",
        obstacle_robot_name=args.obstacle_robot_name,
        obstacle_robot_base_pos=tuple(obs_base_pos),
        obstacle_robot_base_orn=tuple(obs_base_quat),
    )
    robot = env.robot_list[0]
    env.p.resetBasePositionAndOrientation(int(robot.robotId), tuple(ego_base_pos), tuple(ego_base_quat))
    if env.obstacle_robot is not None:
        env.p.resetBasePositionAndOrientation(int(env.obstacle_robot.robotId), tuple(obs_base_pos), tuple(obs_base_quat))

    dynamics_model = ArmLidar(
        {},
        dt=float(args.simulation_dt),
        controller_dt=float(args.controller_period),
        dis_threshold=float(d_safe),
        env=env,
        robot=robot,
        n_obs=int(args.n_observation),
        point_in_dataset_pc=int(args.n_observation_dataset),
        list_sensor=robot.body_joints,
        observation_type="uniform_surface",
        point_dim=3,
        add_normal=False,
        include_point_velocity=False,
        obstacle_horizon_s=float(args.obstacle_horizon_s),
    )
    dynamics_model.set_goal(torch.tensor(robot.q0, dtype=torch.float32))
    return dynamics_model, {
        "ego_base_pos": [float(x) for x in ego_base_pos],
        "ego_base_quat": [float(x) for x in ego_base_quat],
        "obs_base_pos": [float(x) for x in obs_base_pos],
        "obs_base_quat": [float(x) for x in obs_base_quat],
    }


def summarize_split(name: str, split: Dict[str, Any]) -> Dict[str, int]:
    summary = {
        "n": int(split["q_ego"].shape[0]),
        "safe": int(split["safe_mask"].sum().item()),
        "unsafe": int(split["unsafe_mask"].sum().item()),
        "boundary": int(split["boundary_mask"].sum().item()),
    }
    print(f"[{name}] n={summary['n']} safe={summary['safe']} unsafe={summary['unsafe']} boundary={summary['boundary']}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert RayLink raycast HDF5 data to cbf_state_label_v1 cache.")
    p.add_argument("--dataset_dir", required=True, help="Directory containing train.hdf5, val.hdf5, metadata.json.")
    p.add_argument("--out", required=True, help="Output .pt cache path.")
    p.add_argument("--gphi_ckpt", default="", help="Optional g_phi checkpoint. If provided, metadata must match the dataset.")
    p.add_argument("--d_safe", type=float, default=None, help="Safety distance. Defaults to metadata.json d_safe.")
    p.add_argument("--qdot_mode", choices=["zeros", "hdf5"], default="zeros")
    p.add_argument("--assume_static_obstacle", action="store_true", default=False, help="Required with --qdot_mode zeros.")
    p.add_argument("--robot_name", default="panda")
    p.add_argument("--obstacle_robot_name", default="panda")
    p.add_argument("--ego_base_pos", default=None)
    p.add_argument("--ego_base_quat", default=None)
    p.add_argument("--obs_base_pos", default=None)
    p.add_argument("--obs_base_quat", default=None)
    p.add_argument("--n_observation", type=int, default=256)
    p.add_argument("--n_observation_dataset", type=int, default=256)
    p.add_argument("--simulation_dt", type=float, default=1.0 / 120.0)
    p.add_argument("--controller_period", type=float, default=1.0 / 30.0)
    p.add_argument("--obstacle_horizon_s", type=float, default=0.2)
    p.add_argument("--jacobian_chunk_size", type=int, default=512)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    from neural_cbf.tools.metadata_fingerprint import compute_metadata_fingerprint

    dataset_dir = Path(args.dataset_dir)
    out_path = Path(args.out)
    metadata_path = dataset_dir / "metadata.json"
    train_path = dataset_dir / "train.hdf5"
    val_path = dataset_dir / "val.hdf5"
    for path in (metadata_path, train_path, val_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.qdot_mode == "zeros" and not args.assume_static_obstacle:
        raise ValueError("--qdot_mode zeros requires --assume_static_obstacle so the static qdot assumption is explicit.")

    with open(metadata_path, "r", encoding="utf-8") as f:
        dataset_meta = json.load(f)
    d_safe = float(args.d_safe if args.d_safe is not None else dataset_meta.get("d_safe"))
    if not math.isfinite(d_safe) or d_safe <= 0.0:
        raise ValueError(f"Invalid d_safe={d_safe}. Pass --d_safe explicitly.")

    if args.gphi_ckpt:
        ckpt_meta = load_torch_checkpoint_metadata(args.gphi_ckpt)
        compare_metadata(dataset_meta, ckpt_meta)

    dynamics_model, base_meta = build_dynamics_for_jacobian(args, dataset_meta, d_safe)
    train_raw = read_split(train_path, d_safe=d_safe, qdot_mode=str(args.qdot_mode))
    val_raw = read_split(val_path, d_safe=d_safe, qdot_mode=str(args.qdot_mode))
    summarize_split("train/raw", train_raw)
    summarize_split("val/raw", val_raw)

    training = attach_jacobians(train_raw, dynamics_model, args.jacobian_chunk_size, "train")
    validation = attach_jacobians(val_raw, dynamics_model, args.jacobian_chunk_size, "val")
    train_summary = summarize_split("train/cache", training)
    val_summary = summarize_split("val/cache", validation)

    metadata = {
        "schema": "cbf_state_label_v1",
        "robot": "dual_panda",
        **base_meta,
        "source_dataset": str(dataset_dir),
        "source_train_hdf5": str(train_path),
        "source_val_hdf5": str(val_path),
        "source_metadata": str(metadata_path),
        "source_gphi_ckpt": str(args.gphi_ckpt or ""),
        "source_schema": "raylink_raycast_hdf5",
        "source_requires_manual_validation": False,
        "d_safe": float(d_safe),
        "label_rule": {
            "safe_mask": "d_min > d_safe and collision == False",
            "unsafe_mask": "collision == True",
            "boundary_mask": "not safe_mask and not unsafe_mask",
        },
        "qdot_mode": str(args.qdot_mode),
        "assume_static_obstacle": bool(args.assume_static_obstacle),
        "q_dim": int(training["q_ego"].shape[1]),
        "obstacle_q_dim": int(training["q_obs"].shape[1]),
        "obstacle_qdot_dim": int(training["qdot_obs"].shape[1]),
        "obs_dim_discarded": int(dynamics_model.o_dims_in_dataset),
        "sensor_aux_dims": int(dynamics_model.sensor_aux_dims),
        "n_observation": int(args.n_observation),
        "n_observation_dataset": int(args.n_observation_dataset),
        "point_dim": 3,
        "add_normal": False,
        "include_point_velocity": False,
        "raylink_metadata_subset": {
            "r_max": dataset_meta.get("r_max"),
            "num_anchors": dataset_meta.get("num_anchors"),
            "num_rays_per_anchor": dataset_meta.get("num_rays_per_anchor"),
            "num_rays_total": dataset_meta.get("num_rays_total"),
            "anchor_link_ids": dataset_meta.get("anchor_link_ids"),
            "ray_ordering_rule": dataset_meta.get("ray_ordering_rule"),
        },
        "split_summary": {
            "training": train_summary,
            "validation": val_summary,
        },
        "note": "Ray observations are intentionally discarded; gphi/raylink_oracle CBF modes rebuild observations online.",
    }
    metadata["metadata_fingerprint"] = compute_metadata_fingerprint(metadata)
    payload = {
        "schema": "cbf_state_label_v1",
        "training": training,
        "validation": validation,
        "metadata": metadata,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
