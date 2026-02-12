import argparse
import os
from dataclasses import dataclass

import numpy as np

from environment import ArmEnv


@dataclass
class TrajConfig:
    num_traj: int
    num_steps: int
    dt: float
    num_terms: int
    min_freq_hz: float
    max_freq_hz: float
    amplitude_scale: float
    velocity_scale: float


def _build_time_vector(num_steps: int, dt: float) -> np.ndarray:
    return np.arange(num_steps, dtype=np.float32) * dt


def _random_smooth_traj(
    q_min: np.ndarray,
    q_max: np.ndarray,
    qd_max: np.ndarray,
    t: np.ndarray,
    cfg: TrajConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a smooth joint-space trajectory using random sinusoids.

    Returns:
        q_traj: (T, dof)
        qdot_traj: (T, dof)
    """
    dof = q_min.shape[0]
    q_center = (q_min + q_max) / 2.0
    q_range = (q_max - q_min) / 2.0

    # Sample frequencies (Hz), phases, and amplitudes for each joint and term
    freqs = np.random.uniform(cfg.min_freq_hz, cfg.max_freq_hz, size=(cfg.num_terms, dof))
    phases = np.random.uniform(0.0, 2.0 * np.pi, size=(cfg.num_terms, dof))
    amp_raw = np.random.uniform(0.1, 1.0, size=(cfg.num_terms, dof))

    # Limit amplitudes by joint range
    amp = cfg.amplitude_scale * q_range * amp_raw / amp_raw.max(axis=0, keepdims=True)

    # Initial construction
    q = np.zeros((t.shape[0], dof), dtype=np.float32)
    for k in range(cfg.num_terms):
        q += amp[k] * np.sin(2.0 * np.pi * freqs[k] * t[:, None] + phases[k])
    q += q_center

    # Ensure within joint limits by shrinking amplitude if needed
    over = np.maximum(q - q_max, 0.0)
    under = np.maximum(q_min - q, 0.0)
    violation = np.maximum(over.max(), under.max())
    if violation > 0.0:
        shrink = max(0.1, 1.0 - violation / (q_range.max() + 1e-6))
        q = (q - q_center) * shrink + q_center

    # Compute velocity and enforce limits by slowing down frequencies if needed
    qdot = np.gradient(q, cfg.dt, axis=0)
    max_ratio = (np.abs(qdot) / (qd_max[None, :] + 1e-6)).max()
    if max_ratio > 1.0:
        slow = cfg.velocity_scale / max_ratio
        freqs = freqs * slow
        q = np.zeros((t.shape[0], dof), dtype=np.float32)
        for k in range(cfg.num_terms):
            q += amp[k] * np.sin(2.0 * np.pi * freqs[k] * t[:, None] + phases[k])
        q += q_center
        qdot = np.gradient(q, cfg.dt, axis=0)

    # Final clamp to hard limits
    q = np.clip(q, q_min, q_max)
    qdot = np.clip(qdot, -qd_max, qd_max)
    return q.astype(np.float32), qdot.astype(np.float32)


def generate_trajectory_library(cfg: TrajConfig) -> dict:
    # Build a Panda robot to get joint limits and velocity limits
    env = ArmEnv(["panda"], GUI=False, config_file="")
    robot = env.robot_list[0]

    q_min = robot.body_range[:, 0].astype(np.float32)
    q_max = robot.body_range[:, 1].astype(np.float32)
    qd_max = np.array(
        [env.p.getJointInfo(robot.robotId, j)[11] for j in robot.body_joints],
        dtype=np.float32,
    )

    t = _build_time_vector(cfg.num_steps, cfg.dt)
    q_trajs = np.zeros((cfg.num_traj, cfg.num_steps, robot.body_dim), dtype=np.float32)
    qdot_trajs = np.zeros_like(q_trajs)

    for idx in range(cfg.num_traj):
        q, qdot = _random_smooth_traj(q_min, q_max, qd_max, t, cfg)
        q_trajs[idx] = q
        qdot_trajs[idx] = qdot

    return {
        "q_trajs": q_trajs,
        "qdot_trajs": qdot_trajs,
        "dt": np.array(cfg.dt, dtype=np.float32),
        "joint_limits": np.stack([q_min, q_max], axis=0),
        "vel_limits": qd_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_traj", type=int, default=200)
    parser.add_argument("--num_steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=1 / 60)
    parser.add_argument("--num_terms", type=int, default=3)
    parser.add_argument("--min_freq_hz", type=float, default=0.05)
    parser.add_argument("--max_freq_hz", type=float, default=0.6)
    parser.add_argument("--amplitude_scale", type=float, default=0.6)
    parser.add_argument("--velocity_scale", type=float, default=0.95)
    parser.add_argument(
        "--out",
        type=str,
        default="data/obstacle_trajs/panda_trajs.npz",
    )
    args = parser.parse_args()

    cfg = TrajConfig(
        num_traj=args.num_traj,
        num_steps=args.num_steps,
        dt=args.dt,
        num_terms=args.num_terms,
        min_freq_hz=args.min_freq_hz,
        max_freq_hz=args.max_freq_hz,
        amplitude_scale=args.amplitude_scale,
        velocity_scale=args.velocity_scale,
    )

    data = generate_trajectory_library(cfg)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **data)
    print(f"Saved obstacle trajectory library to {out_path}")


if __name__ == "__main__":
    main()
