from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError as exc:
    raise ImportError("h5py is required for this dataset writer. Install it with: pip install h5py") from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment import ArmEnv


SAMPLE_TYPES = {
    "far_random": 0,
    "medium_close": 1,
    "near_boundary": 2,
    "collision_unsafe": 3,
}
SAMPLE_TYPE_NAMES = {v: k for k, v in SAMPLE_TYPES.items()}


@dataclass
class StateRecord:
    q_ego: np.ndarray
    q_obs: np.ndarray
    d_min: float
    h: float
    link_distance_matrix: np.ndarray
    closest_ego_link: int
    closest_obs_link: int
    sample_type: int
    collision: bool
    source: str = ""
    guided_ego_link: int = -1
    guided_obs_link: int = -1
    guided_pair_name: str = ""


GUIDED_PAIR_WEIGHTS = [
    ("panda_hand", "panda_hand", 4.0),
    ("panda_hand", "panda_link5", 3.0),
    ("panda_link5", "panda_hand", 3.0),
    ("panda_hand", "panda_link6", 3.0),
    ("panda_link6", "panda_hand", 3.0),
    ("panda_hand", "panda_link7", 2.5),
    ("panda_link7", "panda_hand", 2.5),
    ("panda_link5", "panda_link5", 2.0),
    ("panda_link5", "panda_link6", 2.0),
    ("panda_link6", "panda_link5", 2.0),
    ("panda_link6", "panda_link6", 2.0),
    ("panda_link7", "panda_link5", 1.5),
    ("panda_link5", "panda_link7", 1.5),
    ("panda_link3", "panda_link5", 1.0),
    ("panda_link4", "panda_link5", 1.0),
    ("panda_link5", "panda_link3", 1.0),
    ("panda_link5", "panda_link4", 1.0),
    ("panda_link4", "panda_link6", 1.0),
    ("panda_link6", "panda_link4", 1.0),
]


def parse_vec(text: str, n: int) -> Tuple[float, ...]:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != n:
        raise ValueError(f"Expected {n} comma-separated values, got {len(vals)}: {text}")
    return tuple(vals)


def quat_to_matrix(p, quat: Tuple[float, float, float, float]) -> np.ndarray:
    return np.asarray(p.getMatrixFromQuaternion(quat), dtype=np.float32).reshape(3, 3)


def make_transform(pos, quat, p) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = quat_to_matrix(p, quat)
    T[:3, 3] = np.asarray(pos, dtype=np.float32)
    return T


def fibonacci_sphere(num: int, seed: int = 0) -> np.ndarray:
    # Deterministic full-sphere directions. The seed only rotates the starting phase.
    idx = np.arange(num, dtype=np.float32)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    phase = float(seed % 997) / 997.0 * 2.0 * math.pi
    z = 1.0 - 2.0 * (idx + 0.5) / float(num)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = golden * idx + phase
    dirs = np.stack((np.cos(theta) * radius, np.sin(theta) * radius, z), axis=1)
    return (dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def choose_anchor_links(robot, num_anchors: int) -> List[int]:
    if num_anchors < 1:
        raise ValueError("num_anchors must be positive.")
    preferred_names = [
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
        "panda_link7",
        "panda_hand",
    ]
    name_to_link = getattr(robot, "arm_link_name", {})
    preferred_links = []
    for name in preferred_names:
        link_id = name_to_link.get(name)
        if link_id is not None and link_id >= 0 and link_id not in preferred_links:
            preferred_links.append(int(link_id))

    body_joints = list(robot.body_joints)
    available_links = preferred_links if len(preferred_links) >= num_anchors else body_joints
    if len(available_links) == 0:
        raise ValueError("Robot has no usable links for anchors.")
    if num_anchors > len(available_links):
        raise ValueError(f"num_anchors={num_anchors} exceeds available anchor links={len(available_links)}.")
    if num_anchors == 1:
        return [available_links[len(available_links) // 2]]
    raw = np.linspace(0, len(available_links) - 1, num_anchors).round().astype(int)
    links = []
    for idx in raw:
        link_id = available_links[int(np.clip(idx, 0, len(available_links) - 1))]
        if link_id not in links:
            links.append(link_id)
    cursor = len(available_links) - 1
    while len(links) < num_anchors and cursor >= 0:
        link_id = available_links[cursor]
        if link_id not in links:
            links.append(link_id)
        cursor -= 1
    return links


def link_name(robot, link_id: int) -> str:
    if int(link_id) == -1:
        return robot.p.getBodyInfo(robot.robotId)[0].decode("utf-8")
    return robot.p.getJointInfo(robot.robotId, int(link_id))[12].decode("utf-8")


def collision_link_ids(robot, include_base: bool = False) -> List[int]:
    links = []
    if include_base and robot.p.getCollisionShapeData(robot.robotId, -1):
        links.append(-1)
    invalid_links = set(int(x) for x in getattr(robot, "invalid_link", []))
    for link_id in range(int(robot.n_joints)):
        if link_id in invalid_links:
            continue
        if robot.p.getCollisionShapeData(robot.robotId, int(link_id)):
            links.append(int(link_id))
    if not links:
        links = [int(x) for x in robot.body_joints]
    return links


def resolve_guided_pairs(ego_robot, obs_robot) -> List[Dict]:
    pairs = []
    ego_names = getattr(ego_robot, "arm_link_name", {})
    obs_names = getattr(obs_robot, "arm_link_name", {})
    for ego_name, obs_name, weight in GUIDED_PAIR_WEIGHTS:
        ego_link = ego_names.get(ego_name)
        obs_link = obs_names.get(obs_name)
        if ego_link is None or obs_link is None:
            continue
        if ego_link < 0 or obs_link < 0:
            continue
        pair_name = f"{ego_name}:{obs_name}"
        pairs.append(
            {
                "ego_name": ego_name,
                "obs_name": obs_name,
                "ego_link": int(ego_link),
                "obs_link": int(obs_link),
                "weight": float(weight),
                "pair_name": pair_name,
            }
        )
    if not pairs:
        raise RuntimeError("No guided IK link pairs could be resolved from robot link names.")
    return pairs


def guided_local_anchor_candidates(link_name_text: str) -> np.ndarray:
    if "hand" in link_name_text:
        anchors = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.04), (0.0, 0.0, 0.08)]
    elif "link7" in link_name_text or "link6" in link_name_text:
        anchors = [(0.0, 0.0, 0.0), (0.04, 0.0, 0.0), (-0.04, 0.0, 0.0)]
    elif "link5" in link_name_text:
        anchors = [(0.0, 0.0, 0.0), (0.0, 0.06, 0.0), (0.0, -0.06, 0.0)]
    else:
        anchors = [(0.0, 0.0, 0.0), (0.0, 0.05, 0.0), (0.0, -0.05, 0.0)]
    return np.asarray(anchors, dtype=np.float32)


def sample_guided_pair(rng: np.random.Generator, guided_pairs: List[Dict], success_counts: Counter) -> Dict:
    weights = []
    for pair in guided_pairs:
        # Keep the hand-heavy prior, but softly prefer pairs that have succeeded less often.
        weights.append(pair["weight"] / math.sqrt(1.0 + float(success_counts[pair["pair_name"]])))
    probs = np.asarray(weights, dtype=np.float64)
    probs = probs / probs.sum()
    return guided_pairs[int(rng.choice(len(guided_pairs), p=probs))]


def sample_shared_workspace(rng: np.random.Generator, x_range, y_range, z_range) -> np.ndarray:
    return np.asarray(
        [
            rng.uniform(float(x_range[0]), float(x_range[1])),
            rng.uniform(float(y_range[0]), float(y_range[1])),
            rng.uniform(float(z_range[0]), float(z_range[1])),
        ],
        dtype=np.float32,
    )


def random_offset(rng: np.random.Generator, min_norm: float, max_norm: float) -> np.ndarray:
    direction = rng.normal(0.0, 1.0, size=3)
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    scale = rng.uniform(float(min_norm), float(max_norm))
    return (direction * scale).astype(np.float32)


def current_link_rotation(env: ArmEnv, robot, link_id: int) -> np.ndarray:
    link_state = env.p.getLinkState(robot.robotId, int(link_id))
    return quat_to_matrix(env.p, link_state[5])


def solve_link_ik(
    env: ArmEnv,
    robot,
    link_id: int,
    target_pos: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
    rest_q: np.ndarray,
) -> Optional[np.ndarray]:
    joint_ranges = np.maximum(q_high - q_low, 1e-4).astype(np.float32)
    kwargs = dict(
        lowerLimits=q_low.astype(float).tolist(),
        upperLimits=q_high.astype(float).tolist(),
        jointRanges=joint_ranges.astype(float).tolist(),
        restPoses=rest_q.astype(float).tolist(),
        maxNumIterations=160,
        residualThreshold=1e-5,
    )
    try:
        sol = env.p.calculateInverseKinematics(
            robot.robotId,
            int(link_id),
            targetPosition=np.asarray(target_pos, dtype=float).tolist(),
            **kwargs,
        )
    except TypeError:
        sol = env.p.calculateInverseKinematics(
            robot.robotId,
            int(link_id),
            targetPosition=np.asarray(target_pos, dtype=float).tolist(),
            maxNumIterations=160,
            residualThreshold=1e-5,
        )
    if len(sol) < robot.body_dim:
        return None
    return clip_q(np.asarray(sol[: robot.body_dim], dtype=np.float32), q_low, q_high)


def set_robot_state(robot, q: np.ndarray) -> None:
    robot.set_joint_position(robot.body_joints, np.asarray(q, dtype=np.float32))


def sample_uniform(rng: np.random.Generator, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return rng.uniform(low=low, high=high).astype(np.float32)


def clip_q(q: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.clip(q, low, high).astype(np.float32)


def interpolate_q(q_a: np.ndarray, q_b: np.ndarray, t: float) -> np.ndarray:
    # Panda joint limits here are not periodic full revolute joints, so linear interpolation
    # inside limits is the conservative shortest path.
    return ((1.0 - t) * q_a + t * q_b).astype(np.float32)


def compute_distance_record(
    env: ArmEnv,
    ego_robot,
    obs_robot,
    q_ego: np.ndarray,
    q_obs: np.ndarray,
    d_safe: float,
    delta: float,
    distance_query_range: float,
) -> Tuple[float, float, np.ndarray, int, int, bool]:
    set_robot_state(ego_robot, q_ego)
    set_robot_state(obs_robot, q_obs)
    env.p.performCollisionDetection()

    ego_links = list(getattr(ego_robot, "collision_link_ids", ego_robot.body_joints))
    obs_links = list(getattr(obs_robot, "collision_link_ids", obs_robot.body_joints))
    probe_distance = max(float(distance_query_range), float(d_safe + delta + 0.2), float(d_safe + 0.3), 0.5)
    D = np.full((len(ego_links), len(obs_links)), probe_distance, dtype=np.float32)
    collision = len(env.p.getContactPoints(ego_robot.robotId, obs_robot.robotId)) > 0

    for i, ego_link in enumerate(ego_links):
        for j, obs_link in enumerate(obs_links):
            pts = env.p.getClosestPoints(
                ego_robot.robotId,
                obs_robot.robotId,
                distance=probe_distance,
                linkIndexA=int(ego_link),
                linkIndexB=int(obs_link),
            )
            if pts:
                D[i, j] = min(float(pt[8]) for pt in pts)
                if D[i, j] <= 0.0:
                    collision = True

    flat_idx = int(np.argmin(D))
    i_min, j_min = np.unravel_index(flat_idx, D.shape)
    d_min = float(D[i_min, j_min])
    if collision and d_min > 0.0:
        d_min = min(d_min, 0.0)
    h = float(d_min - d_safe)
    return d_min, h, D, int(ego_links[i_min]), int(obs_links[j_min]), bool(collision)


def classify_sample(h: float, collision: bool, epsilon: float, delta: float) -> int:
    if collision or h < 0.0:
        return SAMPLE_TYPES["collision_unsafe"]
    if h <= epsilon:
        return SAMPLE_TYPES["near_boundary"]
    if h <= delta:
        return SAMPLE_TYPES["medium_close"]
    return SAMPLE_TYPES["far_random"]


def make_state_record(
    env: ArmEnv,
    ego_robot,
    obs_robot,
    q_ego: np.ndarray,
    q_obs: np.ndarray,
    d_safe: float,
    epsilon: float,
    delta: float,
    distance_query_range: float,
) -> StateRecord:
    d_min, h, D, closest_ego, closest_obs, collision = compute_distance_record(
        env,
        ego_robot,
        obs_robot,
        q_ego,
        q_obs,
        d_safe=d_safe,
        delta=delta,
        distance_query_range=distance_query_range,
    )
    sample_type = classify_sample(h, collision, epsilon=epsilon, delta=delta)
    return StateRecord(
        q_ego=q_ego.astype(np.float32),
        q_obs=q_obs.astype(np.float32),
        d_min=d_min,
        h=h,
        link_distance_matrix=D,
        closest_ego_link=closest_ego,
        closest_obs_link=closest_obs,
        sample_type=sample_type,
        collision=collision,
    )


def find_random_record(
    rng: np.random.Generator,
    env: ArmEnv,
    ego_robot,
    obs_robot,
    q_low: np.ndarray,
    q_high: np.ndarray,
    d_safe: float,
    epsilon: float,
    delta: float,
    distance_query_range: float,
    wanted_type: Optional[int],
    max_attempts: int,
) -> Tuple[Optional[StateRecord], int]:
    for attempt in range(1, max_attempts + 1):
        record = make_state_record(
            env,
            ego_robot,
            obs_robot,
            sample_uniform(rng, q_low, q_high),
            sample_uniform(rng, q_low, q_high),
            d_safe=d_safe,
            epsilon=epsilon,
            delta=delta,
            distance_query_range=distance_query_range,
        )
        if wanted_type is None or record.sample_type == wanted_type:
            return record, attempt
    return None, max_attempts


def generate_guided_unsafe_record(
    rng: np.random.Generator,
    env: ArmEnv,
    ego_robot,
    obs_robot,
    q_low: np.ndarray,
    q_high: np.ndarray,
    d_safe: float,
    epsilon: float,
    delta: float,
    distance_query_range: float,
    guided_pairs: List[Dict],
    guided_pair_attempt_counts: Counter,
    guided_pair_success_counts: Counter,
    x_range,
    y_range,
    z_range,
    offset_min: float,
    offset_max: float,
    ik_noise_std: float,
    max_attempts: int,
) -> Tuple[Optional[StateRecord], int, int]:
    discarded = 0
    for attempt in range(1, max_attempts + 1):
        pair = sample_guided_pair(rng, guided_pairs, guided_pair_success_counts)
        guided_pair_attempt_counts[pair["pair_name"]] += 1

        q_ego_seed = sample_uniform(rng, q_low, q_high)
        q_obs_seed = sample_uniform(rng, q_low, q_high)
        set_robot_state(ego_robot, q_ego_seed)
        set_robot_state(obs_robot, q_obs_seed)

        p_mid = sample_shared_workspace(rng, x_range, y_range, z_range)
        ego_anchor_pool = guided_local_anchor_candidates(pair["ego_name"])
        obs_anchor_pool = guided_local_anchor_candidates(pair["obs_name"])
        ego_anchor = ego_anchor_pool[int(rng.integers(0, ego_anchor_pool.shape[0]))]
        obs_anchor = obs_anchor_pool[int(rng.integers(0, obs_anchor_pool.shape[0]))]
        ego_target = p_mid - current_link_rotation(env, ego_robot, pair["ego_link"]) @ ego_anchor
        obs_target = p_mid + random_offset(rng, offset_min, offset_max)
        obs_target = obs_target - current_link_rotation(env, obs_robot, pair["obs_link"]) @ obs_anchor

        q_ego = solve_link_ik(env, ego_robot, pair["ego_link"], ego_target, q_low, q_high, q_ego_seed)
        q_obs = solve_link_ik(env, obs_robot, pair["obs_link"], obs_target, q_low, q_high, q_obs_seed)
        if q_ego is None or q_obs is None:
            discarded += 1
            continue

        q_ego = clip_q(q_ego + rng.normal(0.0, ik_noise_std, size=q_ego.shape), q_low, q_high)
        q_obs = clip_q(q_obs + rng.normal(0.0, ik_noise_std, size=q_obs.shape), q_low, q_high)
        record = make_state_record(
            env,
            ego_robot,
            obs_robot,
            q_ego,
            q_obs,
            d_safe=d_safe,
            epsilon=epsilon,
            delta=delta,
            distance_query_range=distance_query_range,
        )
        record.source = "guided_ik"
        record.guided_ego_link = int(pair["ego_link"])
        record.guided_obs_link = int(pair["obs_link"])
        record.guided_pair_name = pair["pair_name"]
        if record.sample_type == SAMPLE_TYPES["collision_unsafe"]:
            guided_pair_success_counts[pair["pair_name"]] += 1
            return record, attempt, discarded
    return None, max_attempts, discarded


def find_safe_endpoint(
    rng: np.random.Generator,
    env: ArmEnv,
    ego_robot,
    obs_robot,
    q_low: np.ndarray,
    q_high: np.ndarray,
    d_safe: float,
    epsilon: float,
    delta: float,
    distance_query_range: float,
    max_attempts: int,
) -> Tuple[Optional[StateRecord], int]:
    best = None
    best_score = float("inf")
    for attempt in range(1, max_attempts + 1):
        record = make_state_record(
            env,
            ego_robot,
            obs_robot,
            sample_uniform(rng, q_low, q_high),
            sample_uniform(rng, q_low, q_high),
            d_safe=d_safe,
            epsilon=epsilon,
            delta=delta,
            distance_query_range=distance_query_range,
        )
        if record.collision or record.h <= epsilon:
            continue
        if record.sample_type == SAMPLE_TYPES["medium_close"]:
            record.source = "safe_endpoint_medium"
            return record, attempt
        score = abs(record.h - delta)
        if score < best_score:
            best = record
            best_score = score
    if best is not None:
        best.source = "safe_endpoint_far"
        return best, max_attempts
    return None, max_attempts


def refine_near_boundary_from_endpoints(
    rng: np.random.Generator,
    env: ArmEnv,
    ego_robot,
    obs_robot,
    q_low: np.ndarray,
    q_high: np.ndarray,
    safe_rec: StateRecord,
    unsafe_rec: StateRecord,
    d_safe: float,
    epsilon: float,
    delta: float,
    distance_query_range: float,
    interp_steps: int,
    bisect_steps: int,
    noise_std: float,
    perturbations: int,
) -> Tuple[List[StateRecord], int, int]:
    accepted: List[StateRecord] = []
    candidates = 0
    discarded = 0
    if safe_rec.collision or safe_rec.h <= epsilon or (not unsafe_rec.collision and unsafe_rec.h >= 0.0):
        return accepted, candidates, discarded + 1

    interval = None
    best_positive = None
    prev_t = 0.0
    prev_h = safe_rec.h
    for k in range(1, interp_steps + 1):
        t = float(k) / float(interp_steps)
        rec = make_state_record(
            env,
            ego_robot,
            obs_robot,
            interpolate_q(safe_rec.q_ego, unsafe_rec.q_ego, t),
            interpolate_q(safe_rec.q_obs, unsafe_rec.q_obs, t),
            d_safe,
            epsilon,
            delta,
            distance_query_range,
        )
        candidates += 1
        if rec.h >= 0.0 and (best_positive is None or rec.h < best_positive.h):
            best_positive = rec
        if prev_h > 0.0 and rec.h <= 0.0:
            interval = (prev_t, t)
            break
        prev_t = t
        prev_h = rec.h
    if interval is None:
        if best_positive is not None and best_positive.h <= epsilon:
            best_positive.source = "near_scan"
            best_positive.guided_ego_link = unsafe_rec.guided_ego_link
            best_positive.guided_obs_link = unsafe_rec.guided_obs_link
            best_positive.guided_pair_name = unsafe_rec.guided_pair_name
            accepted.append(best_positive)
            return accepted, candidates, discarded
        return accepted, candidates, discarded + 1

    lo_t, hi_t = interval
    near_rec = None
    for _ in range(bisect_steps):
        mid_t = 0.5 * (lo_t + hi_t)
        rec = make_state_record(
            env,
            ego_robot,
            obs_robot,
            interpolate_q(safe_rec.q_ego, unsafe_rec.q_ego, mid_t),
            interpolate_q(safe_rec.q_obs, unsafe_rec.q_obs, mid_t),
            d_safe,
            epsilon,
            delta,
            distance_query_range,
        )
        candidates += 1
        if rec.h >= 0.0 and (best_positive is None or rec.h < best_positive.h):
            best_positive = rec
        if 0.0 <= rec.h <= epsilon:
            near_rec = rec
            hi_t = mid_t
        elif rec.h < 0.0:
            hi_t = mid_t
        else:
            lo_t = mid_t

    if near_rec is None and best_positive is not None and best_positive.h <= epsilon:
        near_rec = best_positive
    if near_rec is None:
        return accepted, candidates, discarded + 1

    near_rec.source = "near_bisection"
    near_rec.guided_ego_link = unsafe_rec.guided_ego_link
    near_rec.guided_obs_link = unsafe_rec.guided_obs_link
    near_rec.guided_pair_name = unsafe_rec.guided_pair_name
    accepted.append(near_rec)

    # Gaussian perturbation after bisection. Every perturbed state is relabeled.
    for _ in range(int(perturbations)):
        q_e = clip_q(near_rec.q_ego + rng.normal(0.0, noise_std, size=near_rec.q_ego.shape), q_low, q_high)
        q_o = clip_q(near_rec.q_obs + rng.normal(0.0, noise_std, size=near_rec.q_obs.shape), q_low, q_high)
        rec = make_state_record(env, ego_robot, obs_robot, q_e, q_o, d_safe, epsilon, delta, distance_query_range)
        candidates += 1
        rec.source = "near_gaussian"
        rec.guided_ego_link = unsafe_rec.guided_ego_link
        rec.guided_obs_link = unsafe_rec.guided_obs_link
        rec.guided_pair_name = unsafe_rec.guided_pair_name
        if rec.sample_type in (
            SAMPLE_TYPES["near_boundary"],
            SAMPLE_TYPES["medium_close"],
            SAMPLE_TYPES["collision_unsafe"],
        ):
            accepted.append(rec)
    return accepted, candidates, discarded


def compute_rays_for_state(env: ArmEnv, ego_robot, anchor_link_ids: List[int], anchor_T_L_S: np.ndarray, local_ray_dirs: np.ndarray):
    origins = []
    dirs = []
    for anchor_idx, link_id in enumerate(anchor_link_ids):
        link_state = env.p.getLinkState(ego_robot.robotId, int(link_id))
        T_W_L = make_transform(link_state[4], link_state[5], env.p)
        T_W_S = T_W_L @ anchor_T_L_S[anchor_idx]
        origin = T_W_S[:3, 3]
        R_W_S = T_W_S[:3, :3]
        ray_dirs_world = (R_W_S @ local_ray_dirs.T).T
        ray_dirs_world = ray_dirs_world / (np.linalg.norm(ray_dirs_world, axis=1, keepdims=True) + 1e-8)
        origins.append(np.repeat(origin[None, :], local_ray_dirs.shape[0], axis=0))
        dirs.append(ray_dirs_world.astype(np.float32))
    return np.concatenate(origins, axis=0).astype(np.float32), np.concatenate(dirs, axis=0).astype(np.float32)


def raycast_obstacle_only(
    env: ArmEnv,
    ego_robot,
    obstacle_robot,
    origins: np.ndarray,
    dirs: np.ndarray,
    r_max: float,
    save_hit_points: bool,
    ego_base_pos: Tuple[float, float, float],
    ego_base_orn: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], int]:
    ray_to = origins + float(r_max) * dirs
    # PyBullet rayTestBatch returns the first body hit in the whole world. The
    # dataset target must be obstacle-only, so move ego away during raycast and
    # restore it immediately after. Ray origins/directions have already been
    # computed in world coordinates.
    env.p.resetBasePositionAndOrientation(ego_robot.robotId, (1000.0, 1000.0, 1000.0), (0.0, 0.0, 0.0, 1.0))
    try:
        raw = env.p.rayTestBatch(origins.tolist(), ray_to.tolist(), numThreads=0)
    finally:
        env.p.resetBasePositionAndOrientation(ego_robot.robotId, ego_base_pos, ego_base_orn)
    hit_mask = np.zeros((origins.shape[0],), dtype=np.uint8)
    depth = np.full((origins.shape[0],), float(r_max), dtype=np.float32)
    hit_points = np.zeros((origins.shape[0], 3), dtype=np.float32) if save_hit_points else None
    failures = 0

    for i, result in enumerate(raw):
        hit_uid = int(result[0])
        if hit_uid != int(obstacle_robot.robotId):
            continue
        frac = float(result[2])
        dist = frac * float(r_max)
        if 0.0 <= dist < float(r_max):
            hit_mask[i] = 1
            depth[i] = dist
            if hit_points is not None:
                hit_points[i] = origins[i] + dist * dirs[i]
        elif dist < 0.0:
            failures += 1
    return hit_mask, depth, hit_points, failures


def target_counts(num_samples: int, ratios: Dict[str, float]) -> Dict[int, int]:
    raw = {SAMPLE_TYPES[k]: int(round(num_samples * ratios[k])) for k in ratios}
    diff = int(num_samples - sum(raw.values()))
    raw[SAMPLE_TYPES["near_boundary"]] += diff
    return raw


def split_indices_by_type(sample_types: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}
    for type_code in sorted(set(sample_types.tolist())):
        idx = np.where(sample_types == type_code)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(0.8 * n))
        n_val = int(round(0.1 * n))
        splits["train"].extend(idx[:n_train])
        splits["val"].extend(idx[n_train:n_train + n_val])
        splits["test"].extend(idx[n_train + n_val:])
    out = {}
    for key, values in splits.items():
        arr = np.asarray(values, dtype=np.int64)
        rng.shuffle(arr)
        out[key] = arr
    return out


def write_hdf5(path: Path, data: Dict[str, np.ndarray], indices: np.ndarray) -> None:
    with h5py.File(path, "w") as f:
        for key, value in data.items():
            f.create_dataset(key, data=value[indices], compression="gzip", compression_opts=4)


def validate_dataset(data: Dict[str, np.ndarray], report: Dict, q_low: np.ndarray, q_high: np.ndarray, r_max: float, num_anchors: int, rays_per_anchor: int):
    q_ego = data["q_ego"]
    q_obs = data["q_obs"]
    hit_mask = data["hit_mask"]
    depth = data["depth"]
    sample_type = data["sample_type"]
    closest_pairs = list(zip(data["closest_ego_link"].tolist(), data["closest_obs_link"].tolist()))

    report["q_ego_in_limits"] = bool(np.all(q_ego >= q_low[None, :] - 1e-6) and np.all(q_ego <= q_high[None, :] + 1e-6))
    report["q_obs_in_limits"] = bool(np.all(q_obs >= q_low[None, :] - 1e-6) and np.all(q_obs <= q_high[None, :] + 1e-6))
    report["has_nan_or_inf"] = bool(any(not np.isfinite(v).all() for v in data.values() if np.issubdtype(v.dtype, np.floating)))
    report["hit_mask_binary"] = bool(np.all((hit_mask == 0) | (hit_mask == 1)))
    report["depth_in_range"] = bool(np.all(depth >= -1e-6) and np.all(depth <= float(r_max) + 1e-6))
    report["no_hit_depth_is_r_max"] = bool(np.allclose(depth[hit_mask == 0], float(r_max), atol=1e-5))
    report["hit_depth_lt_r_max"] = bool(np.all(depth[hit_mask == 1] < float(r_max)))
    report["sample_type_counts"] = {SAMPLE_TYPE_NAMES[int(k)]: int(v) for k, v in Counter(sample_type.tolist()).items()}
    report["near_boundary_count"] = int(np.sum(sample_type == SAMPLE_TYPES["near_boundary"]))
    report["collision_unsafe_count"] = int(np.sum(sample_type == SAMPLE_TYPES["collision_unsafe"]))

    hit_view = hit_mask.reshape(hit_mask.shape[0], num_anchors, rays_per_anchor)
    anchor_hit_rate = hit_view.mean(axis=(0, 2))
    ray_hit_rate = hit_mask.mean(axis=0)
    report["anchor_hit_rate"] = [float(x) for x in anchor_hit_rate]
    report["dead_anchors"] = [int(i) for i, x in enumerate(anchor_hit_rate) if x == 0.0]
    report["dead_rays_count"] = int(np.sum(ray_hit_rate == 0.0))
    report["closest_link_pair_counts"] = {f"{a}:{b}": int(c) for (a, b), c in Counter(closest_pairs).items()}
    report["closest_ego_link_counts"] = {str(int(k)): int(v) for k, v in Counter(data["closest_ego_link"].tolist()).items()}
    report["closest_obs_link_counts"] = {str(int(k)): int(v) for k, v in Counter(data["closest_obs_link"].tolist()).items()}
    return report


def build_metadata(args, env: ArmEnv, ego_robot, obs_robot, anchor_link_ids, anchor_T_L_S, local_ray_dirs, guided_pairs, mesh_type_used: str) -> Dict:
    T_W_Bego = make_transform(args.ego_base_pos, args.ego_base_orn, env.p)
    T_W_Bobs = make_transform(args.obs_base_pos, args.obs_base_orn, env.p)
    num_anchors = len(anchor_link_ids)
    rays_per_anchor = int(local_ray_dirs.shape[0])
    anchor_local_ray_dirs = np.repeat(local_ray_dirs[None, :, :], num_anchors, axis=0)
    ego_collision_links = list(getattr(ego_robot, "collision_link_ids", ego_robot.body_joints))
    obs_collision_links = list(getattr(obs_robot, "collision_link_ids", obs_robot.body_joints))
    return {
        "robot_model": {
            "ego": args.robot_name,
            "obstacle": args.obstacle_robot_name,
            "ego_body_joints": [int(x) for x in ego_robot.body_joints],
            "obstacle_body_joints": [int(x) for x in obs_robot.body_joints],
            "ego_collision_link_ids": [int(x) for x in ego_collision_links],
            "obstacle_collision_link_ids": [int(x) for x in obs_collision_links],
            "ego_collision_link_names": [link_name(ego_robot, int(x)) for x in ego_collision_links],
            "obstacle_collision_link_names": [link_name(obs_robot, int(x)) for x in obs_collision_links],
            "joint_limits_low": ego_robot.body_range[:, 0].astype(float).tolist(),
            "joint_limits_high": ego_robot.body_range[:, 1].astype(float).tolist(),
        },
        "d_safe": float(args.d_safe),
        "epsilon": float(args.epsilon),
        "delta": float(args.delta),
        "r_max": float(args.r_max),
        "distance_query_range": float(args.distance_query_range),
        "T_W_Bego": T_W_Bego.astype(float).tolist(),
        "T_W_Bobs": T_W_Bobs.astype(float).tolist(),
        "num_anchors": int(num_anchors),
        "num_rays_per_anchor": int(rays_per_anchor),
        "num_rays_total": int(num_anchors * rays_per_anchor),
        "anchor_link_ids": [int(x) for x in anchor_link_ids],
        "anchor_link_names": [link_name(ego_robot, int(x)) for x in anchor_link_ids],
        "anchor_T_L_S": anchor_T_L_S.astype(float).tolist(),
        "local_ray_dirs": anchor_local_ray_dirs.astype(float).tolist(),
        "shared_local_ray_dirs": True,
        "ray_ordering_rule": "anchor_id ascending, then local_ray_index ascending",
        "random_seed": int(args.seed),
        "mesh_type_used": mesh_type_used,
        "include_base_collision_links": bool(args.include_base_collision_links),
        "guided_sampling": {
            "enabled": True,
            "stage_order": "collision_unsafe -> near_boundary -> medium_close -> far_random",
            "pair_pool": [
                {
                    "ego_link_name": pair["ego_name"],
                    "obs_link_name": pair["obs_name"],
                    "ego_link_id": int(pair["ego_link"]),
                    "obs_link_id": int(pair["obs_link"]),
                    "weight": float(pair["weight"]),
                }
                for pair in guided_pairs
            ],
            "shared_workspace_x": [float(x) for x in args.shared_workspace_x],
            "shared_workspace_y": [float(x) for x in args.shared_workspace_y],
            "shared_workspace_z": [float(x) for x in args.shared_workspace_z],
            "guided_offset_range": [float(args.guided_offset_min), float(args.guided_offset_max)],
            "guided_ik_noise_std": float(args.guided_ik_noise_std),
            "near_gaussian_noise_std": float(args.near_noise_std),
            "near_perturbations": int(args.near_perturbations),
            "max_closest_pair_fraction": float(args.max_closest_pair_fraction),
            "overrepresented_pair_accept_prob": float(args.overrepresented_pair_accept_prob),
            "local_anchor_note": "IK targets link-local anchor approximations; final labels always come from the true distance checker.",
        },
        "sample_type_definition": {
            "collision_unsafe": "collision = True or h < 0",
            "near_boundary": "0 <= h <= epsilon",
            "medium_close": "epsilon < h <= delta",
            "far_random": "h > delta",
        },
        "sample_type_codes": SAMPLE_TYPES,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate all-rays-at-once neural raycast surrogate dataset for dual Panda arms.")
    p.add_argument("--out_dir", default="dataset/neural_raycast_debug")
    p.add_argument("--preset", choices=["debug", "full", "large"], default="debug")
    p.add_argument("--num_samples", type=int, default=0, help="Override preset sample count.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--robot_name", default="panda")
    p.add_argument("--obstacle_robot_name", default="panda")
    p.add_argument("--ego_base_pos", default="0.0,-0.25,0.0")
    p.add_argument("--ego_base_orn", default="0.0,0.0,0.0,1.0")
    p.add_argument("--obs_base_pos", default="0.0,0.25,0.0")
    p.add_argument("--obs_base_orn", default="0.0,0.0,1.0,0.0")
    p.add_argument("--num_anchors", type=int, default=6)
    p.add_argument("--rays_per_anchor", type=int, default=128)
    p.add_argument("--r_max", type=float, default=5.0)
    p.add_argument("--distance_query_range", type=float, default=2.0)
    p.add_argument("--d_safe", type=float, default=0.05)
    p.add_argument("--epsilon", type=float, default=0.04)
    p.add_argument("--delta", type=float, default=0.20)
    p.add_argument("--near_noise_std", type=float, default=0.01)
    p.add_argument("--near_perturbations", type=int, default=4)
    p.add_argument("--guided_ik_noise_std", type=float, default=0.04)
    p.add_argument("--guided_offset_min", type=float, default=0.02)
    p.add_argument("--guided_offset_max", type=float, default=0.08)
    p.add_argument("--shared_workspace_x", default="0.05,0.25")
    p.add_argument("--shared_workspace_y", default="-0.10,0.10")
    p.add_argument("--shared_workspace_z", default="0.25,0.80")
    p.add_argument("--max_closest_pair_fraction", type=float, default=0.35)
    p.add_argument("--overrepresented_pair_accept_prob", type=float, default=0.35)
    p.add_argument("--interp_steps", type=int, default=24)
    p.add_argument("--bisect_steps", type=int, default=16)
    p.add_argument("--max_attempts_per_accept", type=int, default=2000)
    p.add_argument("--max_candidate_multiplier", type=int, default=200)
    p.add_argument("--include_base_collision_links", action="store_true")
    p.add_argument("--save_hit_points", action="store_true")
    p.add_argument("--gui", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    preset_counts = {"debug": 10000, "full": 100000, "large": 300000}
    num_samples = int(args.num_samples or preset_counts[args.preset])
    args.ego_base_pos = parse_vec(args.ego_base_pos, 3)
    args.ego_base_orn = parse_vec(args.ego_base_orn, 4)
    args.obs_base_pos = parse_vec(args.obs_base_pos, 3)
    args.obs_base_orn = parse_vec(args.obs_base_orn, 4)
    args.shared_workspace_x = parse_vec(args.shared_workspace_x, 2)
    args.shared_workspace_y = parse_vec(args.shared_workspace_y, 2)
    args.shared_workspace_z = parse_vec(args.shared_workspace_z, 2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "visualizations").mkdir(exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    env = ArmEnv(
        [args.robot_name],
        GUI=bool(args.gui),
        config_file="",
        include_floor=False,
        obstacle_robot_name=args.obstacle_robot_name,
        obstacle_robot_base_pos=args.obs_base_pos,
        obstacle_robot_base_orn=args.obs_base_orn,
    )
    ego_robot = env.robot_list[0]
    obs_robot = env.obstacle_robot
    if obs_robot is None:
        raise RuntimeError("Obstacle robot was not created.")
    env.p.resetBasePositionAndOrientation(ego_robot.robotId, args.ego_base_pos, args.ego_base_orn)
    env.p.resetBasePositionAndOrientation(obs_robot.robotId, args.obs_base_pos, args.obs_base_orn)

    q_low = ego_robot.body_range[:, 0].astype(np.float32)
    q_high = ego_robot.body_range[:, 1].astype(np.float32)
    ego_collision_links = collision_link_ids(ego_robot, include_base=bool(args.include_base_collision_links))
    obs_collision_links = collision_link_ids(obs_robot, include_base=bool(args.include_base_collision_links))
    ego_robot.collision_link_ids = ego_collision_links
    obs_robot.collision_link_ids = obs_collision_links
    anchor_link_ids = choose_anchor_links(ego_robot, int(args.num_anchors))
    anchor_T_L_S = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], len(anchor_link_ids), axis=0)
    local_ray_dirs = fibonacci_sphere(int(args.rays_per_anchor), seed=int(args.seed))
    guided_pairs = resolve_guided_pairs(ego_robot, obs_robot)

    wanted = target_counts(
        num_samples,
        {"far_random": 0.20, "medium_close": 0.30, "near_boundary": 0.40, "collision_unsafe": 0.10},
    )
    records: List[StateRecord] = []
    counts = Counter()
    closest_pair_counts = Counter()
    sampling_stage_counts = Counter()
    guided_pair_attempt_counts = Counter()
    guided_pair_success_counts = Counter()
    guided_closest_pair_counts = Counter()
    total_candidates = 0
    discarded_invalid = 0
    rejected_overrepresented_pairs = 0
    max_total_candidates = int(num_samples * args.max_candidate_multiplier)
    max_closest_pair_count = max(10, int(math.ceil(num_samples * float(args.max_closest_pair_fraction))))

    def readable_counts() -> Dict[str, int]:
        return {SAMPLE_TYPE_NAMES[int(k)]: int(v) for k, v in counts.items()}

    def check_budget(stage_name: str) -> None:
        if total_candidates <= max_total_candidates:
            return
        raise RuntimeError(
            f"Sampling exceeded max candidates ({max_total_candidates}) while filling {stage_name}. "
            f"Current counts={readable_counts()}. Increase --max_candidate_multiplier or relax epsilon/delta."
        )

    def add_record(record: StateRecord, stage_name: str) -> bool:
        nonlocal rejected_overrepresented_pairs
        sample_type = int(record.sample_type)
        if counts[sample_type] >= wanted.get(sample_type, 0):
            return False
        closest_key = f"{int(record.closest_ego_link)}:{int(record.closest_obs_link)}"
        if closest_pair_counts[closest_key] >= max_closest_pair_count:
            if rng.random() > float(args.overrepresented_pair_accept_prob):
                rejected_overrepresented_pairs += 1
                return False
        records.append(record)
        counts[sample_type] += 1
        closest_pair_counts[closest_key] += 1
        sampling_stage_counts[stage_name] += 1
        if record.guided_pair_name:
            guided_closest_pair_counts[closest_key] += 1
        return True

    print(f"[sample] target={num_samples}, rays={len(anchor_link_ids) * int(args.rays_per_anchor)}")

    # Stage 1: actively build the saved unsafe quota with guided IK.
    while counts[SAMPLE_TYPES["collision_unsafe"]] < wanted[SAMPLE_TYPES["collision_unsafe"]]:
        rec, candidates, discarded = generate_guided_unsafe_record(
            rng,
            env,
            ego_robot,
            obs_robot,
            q_low,
            q_high,
            args.d_safe,
            args.epsilon,
            args.delta,
            args.distance_query_range,
            guided_pairs,
            guided_pair_attempt_counts,
            guided_pair_success_counts,
            args.shared_workspace_x,
            args.shared_workspace_y,
            args.shared_workspace_z,
            args.guided_offset_min,
            args.guided_offset_max,
            args.guided_ik_noise_std,
            args.max_attempts_per_accept,
        )
        total_candidates += candidates
        discarded_invalid += discarded
        check_budget("collision_unsafe")
        if rec is None:
            discarded_invalid += 1
            continue
        add_record(rec, "guided_unsafe")

    print(f"[sample] after unsafe counts={readable_counts()}")

    # Stage 2: near-boundary comes from guided unsafe endpoint + safe endpoint + path scan/bisection.
    while counts[SAMPLE_TYPES["near_boundary"]] < wanted[SAMPLE_TYPES["near_boundary"]]:
        unsafe_rec, candidates, discarded = generate_guided_unsafe_record(
            rng,
            env,
            ego_robot,
            obs_robot,
            q_low,
            q_high,
            args.d_safe,
            args.epsilon,
            args.delta,
            args.distance_query_range,
            guided_pairs,
            guided_pair_attempt_counts,
            guided_pair_success_counts,
            args.shared_workspace_x,
            args.shared_workspace_y,
            args.shared_workspace_z,
            args.guided_offset_min,
            args.guided_offset_max,
            args.guided_ik_noise_std,
            args.max_attempts_per_accept,
        )
        total_candidates += candidates
        discarded_invalid += discarded
        check_budget("near_boundary unsafe endpoint")
        if unsafe_rec is None:
            discarded_invalid += 1
            continue
        if counts[SAMPLE_TYPES["collision_unsafe"]] < wanted[SAMPLE_TYPES["collision_unsafe"]]:
            add_record(unsafe_rec, "guided_unsafe_extra")

        safe_rec, candidates = find_safe_endpoint(
            rng,
            env,
            ego_robot,
            obs_robot,
            q_low,
            q_high,
            args.d_safe,
            args.epsilon,
            args.delta,
            args.distance_query_range,
            args.max_attempts_per_accept,
        )
        total_candidates += candidates
        check_budget("near_boundary safe endpoint")
        if safe_rec is None:
            discarded_invalid += 1
            continue

        new_records, candidates, discarded = refine_near_boundary_from_endpoints(
            rng,
            env,
            ego_robot,
            obs_robot,
            q_low,
            q_high,
            safe_rec,
            unsafe_rec,
            args.d_safe,
            args.epsilon,
            args.delta,
            args.distance_query_range,
            args.interp_steps,
            args.bisect_steps,
            args.near_noise_std,
            args.near_perturbations,
        )
        total_candidates += candidates
        discarded_invalid += discarded
        check_budget("near_boundary bisection")
        accepted_any = False
        for rec in new_records:
            accepted_any = add_record(rec, "near_from_guided") or accepted_any
        if not accepted_any:
            discarded_invalid += 1
        if len(records) % max(100, num_samples // 20) == 0:
            print(f"[sample] accepted={len(records)}/{num_samples} counts={readable_counts()}")

    print(f"[sample] after near counts={readable_counts()}")

    # Stage 3: medium-close is easier; fill it with rejection sampling after near is covered.
    while counts[SAMPLE_TYPES["medium_close"]] < wanted[SAMPLE_TYPES["medium_close"]]:
        rec, candidates = find_random_record(
            rng,
            env,
            ego_robot,
            obs_robot,
            q_low,
            q_high,
            args.d_safe,
            args.epsilon,
            args.delta,
            args.distance_query_range,
            SAMPLE_TYPES["medium_close"],
            args.max_attempts_per_accept,
        )
        total_candidates += candidates
        check_budget("medium_close")
        if rec is None:
            discarded_invalid += 1
            continue
        rec.source = "random_medium"
        add_record(rec, "random_medium")

    print(f"[sample] after medium counts={readable_counts()}")

    # Stage 4: far-random is usually abundant, so fill it last.
    while counts[SAMPLE_TYPES["far_random"]] < wanted[SAMPLE_TYPES["far_random"]]:
        rec, candidates = find_random_record(
            rng,
            env,
            ego_robot,
            obs_robot,
            q_low,
            q_high,
            args.d_safe,
            args.epsilon,
            args.delta,
            args.distance_query_range,
            SAMPLE_TYPES["far_random"],
            args.max_attempts_per_accept,
        )
        total_candidates += candidates
        check_budget("far_random")
        if rec is None:
            discarded_invalid += 1
            continue
        rec.source = "random_far"
        add_record(rec, "random_far")

    print(f"[sample] final counts={readable_counts()}")

    n = len(records)
    n_rays = len(anchor_link_ids) * int(args.rays_per_anchor)
    data: Dict[str, np.ndarray] = {
        "q_ego": np.zeros((n, ego_robot.body_dim), dtype=np.float32),
        "q_obs": np.zeros((n, obs_robot.body_dim), dtype=np.float32),
        "hit_mask": np.zeros((n, n_rays), dtype=np.uint8),
        "depth": np.zeros((n, n_rays), dtype=np.float32),
        "depth_norm": np.zeros((n, n_rays), dtype=np.float32),
        "d_min": np.zeros((n,), dtype=np.float32),
        "h": np.zeros((n,), dtype=np.float32),
        "link_distance_matrix": np.zeros((n, len(ego_collision_links), len(obs_collision_links)), dtype=np.float32),
        "closest_ego_link": np.zeros((n,), dtype=np.int16),
        "closest_obs_link": np.zeros((n,), dtype=np.int16),
        "sample_type": np.zeros((n,), dtype=np.int8),
        "collision": np.zeros((n,), dtype=np.uint8),
    }
    if args.save_hit_points:
        data["hit_points_world"] = np.zeros((n, n_rays, 3), dtype=np.float32)

    raycast_failures = 0
    for idx, rec in enumerate(records):
        set_robot_state(ego_robot, rec.q_ego)
        set_robot_state(obs_robot, rec.q_obs)
        origins, dirs = compute_rays_for_state(env, ego_robot, anchor_link_ids, anchor_T_L_S, local_ray_dirs)
        hit_mask, depth, hit_points, failures = raycast_obstacle_only(
            env,
            ego_robot,
            obs_robot,
            origins,
            dirs,
            args.r_max,
            args.save_hit_points,
            args.ego_base_pos,
            args.ego_base_orn,
        )
        raycast_failures += failures

        data["q_ego"][idx] = rec.q_ego
        data["q_obs"][idx] = rec.q_obs
        data["hit_mask"][idx] = hit_mask
        data["depth"][idx] = depth
        data["depth_norm"][idx] = depth / float(args.r_max)
        data["d_min"][idx] = rec.d_min
        data["h"][idx] = rec.h
        data["link_distance_matrix"][idx] = rec.link_distance_matrix
        data["closest_ego_link"][idx] = rec.closest_ego_link
        data["closest_obs_link"][idx] = rec.closest_obs_link
        data["sample_type"][idx] = rec.sample_type
        data["collision"][idx] = int(rec.collision)
        if hit_points is not None:
            data["hit_points_world"][idx] = hit_points

        if (idx + 1) % max(100, n // 20) == 0:
            print(f"[raycast] {idx + 1}/{n}")

    metadata = build_metadata(args, env, ego_robot, obs_robot, anchor_link_ids, anchor_T_L_S, local_ray_dirs, guided_pairs, mesh_type_used="pybullet_collision_shapes")
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    splits = split_indices_by_type(data["sample_type"], seed=int(args.seed))
    for split_name, idx in splits.items():
        write_hdf5(out_dir / f"{split_name}.hdf5", data, idx)

    report = {
        "total_samples": int(n),
        "total_candidates": int(total_candidates),
        "acceptance_rate": float(n / max(total_candidates, 1)),
        "discarded_invalid_samples": int(discarded_invalid),
        "raycast_failure_count": int(raycast_failures),
        "split_counts": {k: int(v.shape[0]) for k, v in splits.items()},
        "sampling_stage_counts": {str(k): int(v) for k, v in sampling_stage_counts.items()},
        "closest_pair_rejected_overrepresented_count": int(rejected_overrepresented_pairs),
        "closest_pair_soft_cap": int(max_closest_pair_count),
        "guided_pair_attempt_counts": {str(k): int(v) for k, v in guided_pair_attempt_counts.items()},
        "guided_pair_success_counts": {str(k): int(v) for k, v in guided_pair_success_counts.items()},
        "guided_pair_success_rate": {
            str(k): float(guided_pair_success_counts[k] / max(v, 1))
            for k, v in guided_pair_attempt_counts.items()
        },
        "guided_closest_pair_counts": {str(k): int(v) for k, v in guided_closest_pair_counts.items()},
    }
    report = validate_dataset(data, report, q_low, q_high, args.r_max, len(anchor_link_ids), int(args.rays_per_anchor))
    with open(out_dir / "sampling_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"[done] wrote dataset to {out_dir}")
    print(json.dumps(report["sample_type_counts"], indent=2))


if __name__ == "__main__":
    main()
