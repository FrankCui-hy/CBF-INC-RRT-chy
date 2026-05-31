import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import contextlib

from neural_cbf.controllers import NeuralLidarCBFController
from environment import ArmEnv
from neural_cbf.systems import ArmLidar
from neural_cbf.experiments import ExperimentSuite, BFContourExperiment, LidarRolloutExperiment
from neural_cbf.evaluation.eval_arm_lidar import draw_environment


@contextlib.contextmanager
def _torch_load_weights_only_false():
    _orig = torch.load

    def _patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    torch.load = _patched
    try:
        yield
    finally:
        torch.load = _orig


def build_dynamics_and_suite(args):
    # env
    env = ArmEnv(
        [args.robot_name],
        GUI=0,
        config_file="",
        obstacle_robot_name=getattr(args, "obstacle_robot_name", "panda"),
        obstacle_traj_path=getattr(args, "obstacle_traj_path", "data/obstacle_trajs/panda_trajs.npz"),
    )
    robot = env.robot_list[0]

    # dynamics
    dm = ArmLidar(
        {},
        dis_threshold=args.dis_threshold,
        dt=args.simulation_dt,
        controller_dt=args.controller_period,
        n_obs=args.n_observation,
        point_dim=args.point_dim,
        add_normal=bool("norm" in args.dataset_name),
        point_in_dataset_pc=args.n_observation_dataset,
        list_sensor=robot.body_joints,
        env=env,
        robot=robot,
        observation_type=args.observation_type,
        include_point_velocity=getattr(args, "include_point_velocity", False),
        obstacle_horizon_s=getattr(args, "obstacle_horizon_s", 0.2),
    )
    dm.compute_linearized_controller(None)

    # goal（随便设一个即可，不影响下面的 safe_mask 统计）
    try:
        import pybullet as p

        ik = p.calculateInverseKinematics(robot.robotId, robot.body_joints[-1], [0.55, 0.0, 0.45])
        goal_state = torch.tensor(ik[:dm.n_dims]).float()
    except Exception:
        goal_state = torch.tensor(robot.q0).float()
    dm.set_goal(goal_state)

    # experiment suite（占位即可）
    start_q = dm.complete_sample_with_observations(goal_state.reshape(1, -1), num_samples=1)
    rollout_exp = LidarRolloutExperiment(
        "Rollout",
        start_q,
        0,
        "x",
        2,
        "y",
        scenarios=[{}],
        n_sims_per_start=1,
        t_sim=1.0,
        compare_nominal=False,
    )
    contour_exp = BFContourExperiment(
        "h_Contour",
        domain=[tuple(robot.body_range[0]), tuple(robot.body_range[2])],
        n_grid=10,
        x_axis_index=0,
        y_axis_index=2,
        x_axis_label="x",
        y_axis_label="y",
        default_state=start_q,
        plot_unsafe_region=True,
    )
    suite = ExperimentSuite([rollout_exp, contour_exp])
    return dm, suite


HP_KEYS = [
    "robot_name",
    "dataset_name",
    "dis_threshold",
    "simulation_dt",
    "controller_period",
    "n_observation",
    "point_dim",
    "n_observation_dataset",
    "observation_type",
    "include_point_velocity",
    "safe_classification_weight",
    "unsafe_classification_weight",
    "descent_violation_weight",
    "hdot_divergence_weight",
    "u_coef_in_training",
    "cbf_hidden_layers",
    "cbf_hidden_size",
    "cbf_alpha",
    "cbf_relaxation_penalty",
    "feature_dim",
    "per_feature_dim",
    "use_bn",
    "ab_mode",
    "baseline",
    "obs_backend",
    "gphi_ckpt",
    "train_use_fd",
]

DYNAMICS_COMPAT_KEYS = [
    "robot_name",
    "dataset_name",
    "dis_threshold",
    "simulation_dt",
    "controller_period",
    "n_observation",
    "point_dim",
    "n_observation_dataset",
    "observation_type",
    "include_point_velocity",
]

METHOD_KEYS = ["ab_mode", "baseline", "obs_backend", "gphi_ckpt", "train_use_fd"]


def load_hparams_from_ckpt(ckpt_path: str) -> dict:
    try:
        with _torch_load_weights_only_false():
            ck = torch.load(ckpt_path, map_location="cpu")
        return ck.get("hyper_parameters", {}) if isinstance(ck, dict) else {}
    except Exception:
        return {}


def args_for_ckpt(base_args: argparse.Namespace, ckpt_path: str, suffix: str) -> argparse.Namespace:
    out = argparse.Namespace(**vars(base_args))
    hp = load_hparams_from_ckpt(ckpt_path)
    for k in HP_KEYS:
        if k in hp and hp[k] is not None:
            setattr(out, k, hp[k])
    for k in METHOD_KEYS:
        override = getattr(base_args, f"{k}_{suffix}", None)
        if override is not None:
            setattr(out, k, override)
    setattr(out, "ckpt_path", ckpt_path)
    return out


def validate_pair_compatible(args_a: argparse.Namespace, args_b: argparse.Namespace):
    mismatches = []
    for k in DYNAMICS_COMPAT_KEYS:
        va = getattr(args_a, k, None)
        vb = getattr(args_b, k, None)
        if va != vb:
            mismatches.append(f"{k}: A={va!r}, B={vb!r}")
    if mismatches:
        joined = "\n  ".join(mismatches)
        raise ValueError(f"Cannot compare checkpoints with different dynamics/observation configs:\n  {joined}")


def load_controller(ckpt_path, dm, suite, args):
    loss_config = {
        "u_coef_in_training": getattr(args, "u_coef_in_training", 5e-1),
        "safe_classification_weight": getattr(args, "safe_classification_weight", 20.0),
        "unsafe_classification_weight": getattr(args, "unsafe_classification_weight", 20.0),
        "descent_violation_weight": getattr(args, "descent_violation_weight", 2.0),
        "hdot_divergence_weight": getattr(args, "hdot_divergence_weight", 2e-2),
        "epsilon": getattr(args, "epsilon", 0.0),
    }
    with _torch_load_weights_only_false():
        baseline_flag = bool(getattr(args, "baseline", False))
        ctrl = NeuralLidarCBFController.load_from_checkpoint(
            ckpt_path,
            dynamics_model=dm,
            scenarios=[{}],
            datamodule=None,
            experiment_suite=suite,
            use_bn=getattr(args, "use_bn", False),
            cbf_hidden_layers=getattr(args, "cbf_hidden_layers", 2),
            cbf_hidden_size=getattr(args, "cbf_hidden_size", 48),
            cbf_alpha=getattr(args, "cbf_alpha", 1.0),
            cbf_relaxation_penalty=getattr(args, "cbf_relaxation_penalty", 5000.0),
            feature_dim=getattr(args, "feature_dim", 32),
            per_feature_dim=getattr(args, "per_feature_dim", 64),
            loss_config=loss_config,
            controller_period=getattr(args, "controller_period", 1 / 30),
            all_hparams=args,
            use_neural_actor=0,
            ab_mode=getattr(args, "ab_mode", "B_with_normal"),
            baseline=baseline_flag,
            obs_backend=getattr(args, "obs_backend", "raw" if baseline_flag else "gphi"),
            gphi_ckpt=getattr(args, "gphi_ckpt", "loss/outputs_real_v2/checkpoints/g_phi_best.pt"),
            train_use_fd=getattr(args, "train_use_fd", baseline_flag),
            map_location="cpu",
        )
    ctrl.eval()
    return ctrl


def build_eval_batch(ctrl, dm, n: int):
    """Get a batch with masks either from datamodule or by on-the-fly sampling."""
    # Preferred path: use datamodule training_data if present.
    if getattr(ctrl, "datamodule", None) is not None:
        ctrl.datamodule.prepare_data()
        td = ctrl.datamodule.training_data
        N_total = len(td)
        n_use = min(int(n), int(N_total))
        idx = torch.randperm(N_total)[:n_use]
        data_x, goal_mask, safe_mask, unsafe_mask, boundary_mask, JP, JR = td[idx]
        return data_x, goal_mask.bool(), safe_mask.bool(), unsafe_mask.bool(), boundary_mask.bool(), idx

    # Fallback path: sample q uniformly and label with system masks.
    ul, ll = dm.state_limits
    ul = ul.detach().clone().float().reshape(1, -1)
    ll = ll.detach().clone().float().reshape(1, -1)
    hi = torch.maximum(ul, ll)
    lo = torch.minimum(ul, ll)
    q = lo + torch.rand(int(n), dm.n_dims) * (hi - lo)
    data_x = dm.complete_sample_with_observations(q, num_samples=int(n))
    safe_mask = dm.safe_mask(data_x).bool()
    unsafe_mask = dm.unsafe_mask(data_x).bool()
    goal_mask = torch.zeros_like(safe_mask)
    boundary_mask = torch.logical_not(torch.logical_or(safe_mask, unsafe_mask))
    idx = torch.arange(int(n), dtype=torch.long)
    return data_x, goal_mask, safe_mask, unsafe_mask, boundary_mask, idx


def summarize_violation_observation(name: str, dm, data_x_in: torch.Tensor, vio_idx: torch.Tensor):
    if vio_idx.numel() == 0:
        print(f"[{name}] no safe-violation samples.")
        return

    obs_seg = data_x_in[vio_idx, dm.n_dims : dm.n_dims + dm.o_dims]
    obs_len = int(obs_seg.shape[1])
    pd = int(getattr(dm, "point_dims", 4))
    # If configured point_dims doesn't divide the observation length, infer a plausible one.
    if obs_len % pd != 0:
        # Prefer common point dimensions used in this repo.
        for cand in (4, 3, 6, 7, 8, 9, 10, 12):
            if obs_len % cand == 0:
                pd = cand
                break
    o = obs_seg.reshape(vio_idx.numel(), -1, pd)
    if pd == 4:
        ranges = o[..., 0]
        hit_ch = o[..., -1]
        hit_ratio = float((hit_ch > 0.5).float().mean().item())
    else:
        ranges = torch.linalg.norm(o[..., :3], dim=-1)
        hit_ratio = float((ranges < 0.95 * float(dm.dis_threshold)).float().mean().item())

    min_r = ranges.min(dim=1).values
    near_ratio = float((ranges < 0.2).float().mean().item())
    p = torch.quantile(min_r, torch.tensor([0.05, 0.5, 0.95], device=min_r.device))
    print(
        f"[{name}] obs_pattern: hit_ratio={hit_ratio:.4f}, near_ratio(r<0.2)={near_ratio:.4f}, "
        f"min_r(p05/p50/p95)=({float(p[0]):.4f}, {float(p[1]):.4f}, {float(p[2]):.4f})"
    )


def render_violations(ctrl, data_x_in: torch.Tensor, vio_idx: torch.Tensor, save_dir: str, topk: int):
    if vio_idx.numel() == 0 or topk <= 0:
        return
    os.makedirs(save_dir, exist_ok=True)
    take = vio_idx[: min(int(topk), vio_idx.numel())]

    # Infer per-point dimension from the observation segment length; needed for draw_environment.
    try:
        dm = ctrl.dynamics_model
        obs_len = int(data_x_in.shape[1] - int(dm.n_dims) - int(getattr(dm, "aux_dims", 0)))
        pd = int(getattr(dm, "point_dims", 4))
        if obs_len % pd != 0:
            for cand in (4, 3, 6, 7, 8, 9, 10, 12):
                if obs_len % cand == 0:
                    pd = cand
                    break
        _old_pd = int(getattr(dm, "point_dims", pd))
        _old_pdim = int(getattr(dm, "point_dim", pd)) if hasattr(dm, "point_dim") else _old_pd
        dm.point_dims = pd
        if hasattr(dm, "point_dim"):
            dm.point_dim = pd
    except Exception:
        dm = None
        pd = None
        _old_pd = None
        _old_pdim = None

    try:
        for i in take.tolist():
            try:
                draw_environment(ctrl, data_x_in[i].detach().cpu(), int(i), save_dir)
            except Exception as e:
                print(f"[WARN] draw_environment failed at idx={i}: {e}")
    finally:
        try:
            if dm is not None and _old_pd is not None:
                dm.point_dims = _old_pd
            if dm is not None and _old_pdim is not None and hasattr(dm, "point_dim"):
                dm.point_dim = _old_pdim
        except Exception:
            pass


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_a", required=True, help="checkpoint A path")
    ap.add_argument("--ckpt_b", required=True, help="checkpoint B path")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--vis_topk", type=int, default=20)
    ap.add_argument("--save_dir", type=str, default="compare_safe_h_outputs")

    # 必要 args（保持和你训练/评估一致）
    ap.add_argument("--robot_name", default="panda")
    ap.add_argument("--dataset_name", default="data")
    ap.add_argument("--dis_threshold", type=float, default=1.0)
    ap.add_argument("--simulation_dt", type=float, default=1 / 120)
    ap.add_argument("--controller_period", type=float, default=1 / 30)
    ap.add_argument("--n_observation", type=int, default=5)
    ap.add_argument("--point_dim", type=int, default=4)
    ap.add_argument("--n_observation_dataset", type=int, default=5)
    ap.add_argument("--observation_type", default="uniform_lidar")
    ap.add_argument("--ab_mode_a", default=None, choices=["A_no_normal", "B_with_normal"])
    ap.add_argument("--ab_mode_b", default=None, choices=["A_no_normal", "B_with_normal"])
    ap.add_argument("--baseline_a", dest="baseline_a", action="store_true", default=None)
    ap.add_argument("--no_baseline_a", dest="baseline_a", action="store_false")
    ap.add_argument("--baseline_b", dest="baseline_b", action="store_true", default=None)
    ap.add_argument("--no_baseline_b", dest="baseline_b", action="store_false")
    ap.add_argument("--obs_backend_a", default=None, choices=["gphi", "raw"])
    ap.add_argument("--obs_backend_b", default=None, choices=["gphi", "raw"])
    ap.add_argument("--gphi_ckpt_a", default=None)
    ap.add_argument("--gphi_ckpt_b", default=None)
    ap.add_argument("--train_use_fd_a", dest="train_use_fd_a", action="store_true", default=None)
    ap.add_argument("--no_train_use_fd_a", dest="train_use_fd_a", action="store_false")
    ap.add_argument("--train_use_fd_b", dest="train_use_fd_b", action="store_true", default=None)
    ap.add_argument("--no_train_use_fd_b", dest="train_use_fd_b", action="store_false")

    args = ap.parse_args()
    args_a = args_for_ckpt(args, args.ckpt_a, "a")
    args_b = args_for_ckpt(args, args.ckpt_b, "b")
    validate_pair_compatible(args_a, args_b)

    dm, suite = build_dynamics_and_suite(args_a)
    ctrl_a = load_controller(args.ckpt_a, dm, suite, args_a)
    ctrl_b = load_controller(args.ckpt_b, dm, suite, args_b)

    meta_a = ctrl_a.method_metadata() if hasattr(ctrl_a, "method_metadata") else {}
    meta_b = ctrl_b.method_metadata() if hasattr(ctrl_b, "method_metadata") else {}
    print(f"[A] {json.dumps(meta_a, ensure_ascii=False)}")
    print(f"[B] {json.dumps(meta_b, ensure_ascii=False)}")

    eps = float(ctrl_a.safe_level)  # 你说 safe 要 <= -0.1，这里就会是 0.1
    thr = -eps

    # Build one shared batch for both controllers.
    data_x, goal_mask, safe_mask, unsafe_mask, boundary_mask, sample_idx = build_eval_batch(ctrl_a, dm, int(args.n))
    # 你之前代码里有 data_x = data_x[:, :-1]，保持一致
    if data_x.ndim == 2 and data_x.shape[1] == int(getattr(ctrl_a, "n_dims_extended", data_x.shape[1])) + 1:
        data_x_in = data_x[:, :-1]
    else:
        data_x_in = data_x

    safe_mask = safe_mask.bool()
    n_safe = int(safe_mask.sum().item())
    if n_safe == 0:
        print("[ERR] sampled batch contains no safe points; increase --n or check datamodule masks.")
        return

    # 两个模型在“同一输入 data_x_in”上的 h
    h_a = ctrl_a.h(data_x_in).reshape(-1)
    h_b = ctrl_b.h(data_x_in).reshape(-1)

    # safe 子集统计
    ha_s = h_a[safe_mask]
    hb_s = h_b[safe_mask]

    def summarize(name, hvals):
        q = torch.quantile(hvals, torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0]))
        mean = float(hvals.mean().item())
        sat = float((hvals <= thr).float().mean().item())
        print(f"\n[{name}] safe_count={hvals.numel()}  thr={thr:.3f}")
        print(f"  P(h<=thr | safe) = {sat:.4f}")
        print(f"  mean={mean:.6f}")
        print(
            f"  quantiles: min={float(q[0]):.6f}  p05={float(q[1]):.6f}  p50={float(q[2]):.6f}  p95={float(q[3]):.6f}  max={float(q[4]):.6f}"
        )

    summarize("A", ha_s)
    summarize("B", hb_s)

    safe_violate_a = torch.logical_and(safe_mask, h_a > thr)
    safe_violate_b = torch.logical_and(safe_mask, h_b > thr)
    vio_local_a = torch.where(safe_violate_a)[0]
    vio_local_b = torch.where(safe_violate_b)[0]
    vio_global_a = sample_idx[vio_local_a]
    vio_global_b = sample_idx[vio_local_b]

    print(f"\n[SAFE-VIOLATION] baseline(A) count={vio_local_a.numel()} / safe={n_safe}")
    print(f"[SAFE-VIOLATION] jvp(B)      count={vio_local_b.numel()} / safe={n_safe}")
    summarize_violation_observation("A", dm, data_x_in, vio_local_a)
    summarize_violation_observation("B", dm, data_x_in, vio_local_b)

    os.makedirs(args.save_dir, exist_ok=True)
    out_json = os.path.join(args.save_dir, "safe_violation_indices.json")
    payload = {
        "thr": float(thr),
        "n_total": int(data_x_in.shape[0]),
        "n_safe": int(n_safe),
        "baseline_count": int(vio_local_a.numel()),
        "jvp_count": int(vio_local_b.numel()),
        "method_a": meta_a,
        "method_b": meta_b,
        "baseline_local_idx": [int(i) for i in vio_local_a.tolist()],
        "jvp_local_idx": [int(i) for i in vio_local_b.tolist()],
        "baseline_global_idx": [int(i) for i in vio_global_a.tolist()],
        "jvp_global_idx": [int(i) for i in vio_global_b.tolist()],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] safe-violation indices -> {out_json}")

    render_violations(ctrl_a, data_x_in, vio_local_a, os.path.join(args.save_dir, "vis_baseline"), int(args.vis_topk))
    render_violations(ctrl_b, data_x_in, vio_local_b, os.path.join(args.save_dir, "vis_jvp"), int(args.vis_topk))
    print(f"[SAVE] violation visualizations -> {os.path.abspath(args.save_dir)}")

    # 可选：再看一下 unsafe 子集（验证方向正确）
    n_unsafe = int(unsafe_mask.sum().item())
    if n_unsafe > 0:
        hu_a = h_a[unsafe_mask.bool()]
        hu_b = h_b[unsafe_mask.bool()]
        # unsafe 的阈值通常是 >= unsafe_level（你训练里可能 unsafe_level=+0.1）
        u_thr = float(getattr(ctrl_a, "unsafe_level", 0.1))
        sat_u_a = float((hu_a >= u_thr).float().mean().item())
        sat_u_b = float((hu_b >= u_thr).float().mean().item())
        print(f"\n[UNSAFE] count={n_unsafe}  u_thr={u_thr:.3f}  P(h>=u_thr|unsafe): A={sat_u_a:.4f} B={sat_u_b:.4f}")


if __name__ == "__main__":
    main()
