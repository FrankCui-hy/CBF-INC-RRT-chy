from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from loss.data.raycast_dummy import (
    RaycastConfig,
    ego_sensor_origin,
    fibonacci_sphere_dirs,
    obstacle_spheres_from_q,
    raycast_spheres,
)
from loss.utils.config import load_config
from loss.utils.seed import set_seed


def _sample_uniform(low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return low + torch.rand_like(low) * (high - low)


def _random_walk_step(q: torch.Tensor, sigma: float, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    q_next = q + sigma * torch.randn_like(q)
    return torch.clamp(q_next, min=low, max=high)


def build_episode_samples(cfg: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    sys_cfg = cfg["system"]
    data_cfg = cfg["data"]
    ray_cfg = RaycastConfig(**data_cfg["ray"])

    n_ego = int(sys_cfg["n_ego"])
    n_obs = int(sys_cfg["n_obs"])
    R = int(sys_cfg["rays"])
    dt = float(sys_cfg["dt"])

    ql_e_low = torch.tensor(sys_cfg["q_limits_ego"]["low"], dtype=torch.float32, device=device)
    ql_e_high = torch.tensor(sys_cfg["q_limits_ego"]["high"], dtype=torch.float32, device=device)
    ql_o_low = torch.tensor(sys_cfg["q_limits_obs"]["low"], dtype=torch.float32, device=device)
    ql_o_high = torch.tensor(sys_cfg["q_limits_obs"]["high"], dtype=torch.float32, device=device)

    num_episodes = int(data_cfg["n_episodes"])
    T = int(data_cfg["episode_len"])
    ratio_dynamic = float(data_cfg["dynamic_episode_ratio"])
    sigma_dyn = float(data_cfg["sigma_dynamic"])
    sigma_static_obs = float(data_cfg["sigma_static_obs"])
    sigma_static_ego = float(data_cfg.get("sigma_static_ego", 0.0))

    safe_dist = float(data_cfg["safe_dist"])
    unsafe_dist = float(data_cfg["unsafe_dist"])

    ray_dirs = fibonacci_sphere_dirs(R, device=device)

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
    buf: dict[str, list[torch.Tensor]] = {k: [] for k in keys}

    for ep in range(num_episodes):
        is_dynamic = ep < int(num_episodes * ratio_dynamic)
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

            origin = ego_sensor_origin(q_ego)
            ray_origin = origin[None, :].repeat(R, 1)
            centers, radii = obstacle_spheres_from_q(q_obs)
            p_gt, n_gt, m, hit_dist = raycast_spheres(ray_origin, ray_dirs, centers, radii, ray_cfg)

            min_dist = hit_dist.min()
            if min_dist <= unsafe_dist:
                y = torch.tensor(-1.0, device=device)
            elif min_dist >= safe_dist:
                y = torch.tensor(1.0, device=device)
            else:
                q_ego = q_ego_next
                q_obs = q_obs_next
                continue

            buf["q_ego"].append(q_ego.clone())
            buf["qdot_ego"].append(qdot_ego.clone())
            buf["q_obs"].append(q_obs.clone())
            buf["qdot_obs"].append(qdot_obs.clone())
            buf["p_gt"].append(p_gt.clone())
            buf["n_gt"].append(n_gt.clone())
            buf["m"].append(m.clone())
            buf["ray_origin"].append(ray_origin.clone())
            buf["ray_dir"].append(ray_dirs.clone())
            buf["y"].append(y)
            buf["episode_type"].append(torch.tensor(1.0 if is_dynamic else 0.0, device=device))

            q_ego = q_ego_next
            q_obs = q_obs_next

    if not buf["q_ego"]:
        raise RuntimeError(
            "No labeled samples were collected. Adjust safe_dist/unsafe_dist or sampling noise; "
            "the current config left every sample in the ambiguous band."
        )

    out = {k: torch.stack(v, dim=0).cpu() for k, v in buf.items()}
    out["meta"] = {
        "n_ego": n_ego,
        "n_obs": n_obs,
        "rays": R,
        "dt": dt,
        "num_episodes": num_episodes,
        "episode_len": T,
        "dynamic_episode_ratio": ratio_dynamic,
    }
    return out


def save_dataset(payload: dict[str, torch.Tensor], cfg: dict[str, Any]) -> Path:
    dataset_path = Path(cfg["paths"]["dataset_path"])
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = cfg["data"].get("save_format", "pt")

    if fmt != "pt":
        raise ValueError("This implementation currently saves torch .pt format only.")

    torch.save(payload, dataset_path)
    return dataset_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect episode dataset for surrogate observation + CBF training.")
    p.add_argument("--config", type=str, default="loss/configs/config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    device = torch.device("cpu")

    payload = build_episode_samples(cfg, device=device)
    path = save_dataset(payload, cfg)

    n = payload["q_ego"].shape[0]
    dyn_ratio = payload["episode_type"].float().mean().item()
    safe_ratio = (payload["y"] > 0).float().mean().item()
    print(f"[collect_dataset] saved: {path}")
    print(f"[collect_dataset] samples={n}, dynamic_ratio={dyn_ratio:.3f}, safe_ratio={safe_ratio:.3f}")


if __name__ == "__main__":
    main()
