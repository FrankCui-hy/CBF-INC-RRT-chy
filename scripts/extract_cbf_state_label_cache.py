import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch

from neural_cbf.tools.metadata_fingerprint import compute_metadata_fingerprint


ASSUMED_ENVS = {
    "dual_panda_0p25_v1": {
        "robot": "dual_panda",
        "ego_base_pos": [0.0, -0.25, 0.0],
        "obs_base_pos": [0.0, 0.25, 0.0],
        "obs_base_quat": [0.0, 0.0, 1.0, 0.0],
    }
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract q/state labels from an old ArmLidar CBF .pt cache.")
    p.add_argument("--source_cache", required=True, help="Old EpisodicDataModule .pt cache.")
    p.add_argument("--out", required=True, help="Output state-label .pt path.")
    p.add_argument(
        "--assume_source_env",
        choices=sorted(ASSUMED_ENVS.keys()),
        default=None,
        help="Required when the old cache has no reliable geometry metadata.",
    )
    p.add_argument("--q_dim", type=int, default=7)
    p.add_argument("--obstacle_q_dim", type=int, default=7)
    p.add_argument("--obstacle_qdot_dim", type=int, default=7)
    p.add_argument("--num_sensors", type=int, default=7)
    p.add_argument("--sensor_aux_dims", type=int, default=None)
    p.add_argument("--obs_dim", type=int, default=None, help="Old oracle observation slice width. Overrides point settings.")
    p.add_argument("--n_observation_dataset", type=int, default=256)
    p.add_argument("--point_dim", type=int, default=3)
    p.add_argument("--include_point_velocity", action="store_true")
    p.add_argument("--add_normal", action="store_true")
    return p.parse_args()


def _mask(cache: Dict, split: str, name: str) -> torch.Tensor:
    flat_key = f"{name}_mask_{split}"
    if flat_key in cache:
        return cache[flat_key]
    dict_key = f"x_{split}_mask"
    if dict_key in cache and name in cache[dict_key]:
        return cache[dict_key][name]
    raise KeyError(f"Missing {name} mask for split '{split}'. Expected {flat_key} or {dict_key}['{name}'].")


def _lookahead(cache: Dict, split: str, key: str) -> torch.Tensor:
    flat_key = f"{key}_{split}"
    if flat_key in cache:
        return cache[flat_key]
    dict_key = f"x_{split}_lookahead"
    if dict_key in cache and key in cache[dict_key]:
        return cache[dict_key][key]
    raise KeyError(f"Missing lookahead {key} for split '{split}'. Expected {flat_key} or {dict_key}['{key}'].")


def _compute_obs_dim(args: argparse.Namespace) -> int:
    if args.obs_dim is not None:
        return int(args.obs_dim)
    point_width = int(args.point_dim) + 3 * int(bool(args.include_point_velocity)) + 3 * int(bool(args.add_normal))
    return int(args.n_observation_dataset) * point_width


def _parse_split(
    x: torch.Tensor,
    safe_mask: torch.Tensor,
    unsafe_mask: torch.Tensor,
    boundary_mask: torch.Tensor,
    j_p: torch.Tensor,
    j_r: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    if x.ndim != 2:
        raise ValueError(f"Expected x to be rank-2, got shape {tuple(x.shape)}.")

    q_dim = int(args.q_dim)
    obs_dim = _compute_obs_dim(args)
    sensor_aux_dims = int(args.sensor_aux_dims) if args.sensor_aux_dims is not None else int(args.num_sensors) * 12
    aux_dim = sensor_aux_dims + int(args.obstacle_q_dim) + int(args.obstacle_qdot_dim) + 2
    expected_dim = q_dim + obs_dim + aux_dim
    if int(x.shape[1]) != expected_dim:
        raise ValueError(
            "Old cache tensor width does not match auxv2 layout. "
            f"got={int(x.shape[1])}, expected={expected_dim} "
            f"(q_dim={q_dim}, obs_dim={obs_dim}, sensor_aux_dims={sensor_aux_dims}, "
            f"obstacle_q_dim={args.obstacle_q_dim}, obstacle_qdot_dim={args.obstacle_qdot_dim}). "
            "Pass the correct --obs_dim/--n_observation_dataset/--add_normal settings, or regenerate the old cache with auxv2."
        )

    aux = x[:, q_dim + obs_dim :]
    q_start = sensor_aux_dims
    qdot_start = q_start + int(args.obstacle_q_dim)
    q_obs = aux[:, q_start:qdot_start]
    qdot_obs = aux[:, qdot_start:qdot_start + int(args.obstacle_qdot_dim)]
    traj_idx = aux[:, -2]
    step_idx = aux[:, -1]
    if q_obs.shape[1] != int(args.obstacle_q_dim):
        raise ValueError(f"Failed to parse q_obs: got shape {tuple(q_obs.shape)}.")
    if not torch.isfinite(q_obs).all():
        raise ValueError("Parsed q_obs contains NaN/Inf; refusing to write state-label cache.")
    if torch.allclose(q_obs, torch.zeros_like(q_obs)):
        raise ValueError(
            "Parsed q_obs is all zeros. This usually means the source cache is auxv1 or the obs_dim settings are wrong. "
            "Refusing to silently use zero obstacle state."
        )

    n = int(x.shape[0])
    for mask_name, mask_value in (("safe", safe_mask), ("unsafe", unsafe_mask), ("boundary", boundary_mask)):
        if int(mask_value.shape[0]) != n:
            raise ValueError(f"{mask_name}_mask length {int(mask_value.shape[0])} does not match x length {n}.")

    return {
        "q_ego": x[:, :q_dim].clone(),
        "q_obs": q_obs.clone(),
        "qdot_obs": qdot_obs.clone(),
        "traj_idx": traj_idx.clone(),
        "step_idx": step_idx.clone(),
        "safe_mask": safe_mask.clone(),
        "unsafe_mask": unsafe_mask.clone(),
        "boundary_mask": boundary_mask.clone(),
        "J_P": j_p.clone(),
        "J_R": j_r.clone(),
    }


def main() -> None:
    args = parse_args()
    source = Path(args.source_cache)
    out = Path(args.out)
    if args.assume_source_env is None:
        raise ValueError(
            "Old CBF caches do not store RayLink-compatible geometry metadata. "
            "Pass --assume_source_env dual_panda_0p25_v1 after manually confirming the source cache was generated with that setup."
        )

    cache = torch.load(source, map_location="cpu")
    if "x_training" not in cache or "x_validation" not in cache:
        raise KeyError("Source cache must contain x_training and x_validation.")

    training = _parse_split(
        cache["x_training"],
        _mask(cache, "training", "safe"),
        _mask(cache, "training", "unsafe"),
        _mask(cache, "training", "boundary"),
        _lookahead(cache, "training", "J_P"),
        _lookahead(cache, "training", "J_R"),
        args,
    )
    validation = _parse_split(
        cache["x_validation"],
        _mask(cache, "validation", "safe"),
        _mask(cache, "validation", "unsafe"),
        _mask(cache, "validation", "boundary"),
        _lookahead(cache, "validation", "J_P"),
        _lookahead(cache, "validation", "J_R"),
        args,
    )

    env_meta = ASSUMED_ENVS[str(args.assume_source_env)]
    metadata = {
        "schema": "cbf_state_label_v1",
        **env_meta,
        "source_cache": str(source),
        "assume_source_env": str(args.assume_source_env),
        "source_requires_manual_validation": True,
        "q_dim": int(args.q_dim),
        "obstacle_q_dim": int(args.obstacle_q_dim),
        "obstacle_qdot_dim": int(args.obstacle_qdot_dim),
        "obs_dim_discarded": _compute_obs_dim(args),
        "sensor_aux_dims": int(args.sensor_aux_dims) if args.sensor_aux_dims is not None else int(args.num_sensors) * 12,
        "note": "Old oracle point cloud is intentionally discarded for gphi mode.",
    }
    metadata["metadata_fingerprint"] = compute_metadata_fingerprint(metadata)
    payload = {
        "schema": "cbf_state_label_v1",
        "training": training,
        "validation": validation,
        "metadata": metadata,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"Wrote state-label cache: {out}")
    print(f"training N={training['q_ego'].shape[0]} validation N={validation['q_ego'].shape[0]}")


if __name__ == "__main__":
    main()
