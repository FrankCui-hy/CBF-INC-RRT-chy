import argparse
import numpy as np
import torch
import torch.nn.functional as F
import contextlib

from neural_cbf.controllers import NeuralLidarCBFController
from environment import ArmEnv
from neural_cbf.systems import ArmLidar
from neural_cbf.experiments import ExperimentSuite, BFContourExperiment, LidarRolloutExperiment


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


def merge_args_from_ckpt(args):
    """Overwrite runtime args with ckpt hyper-parameters when available."""
    try:
        with _torch_load_weights_only_false():
            ck = torch.load(args.ckpt_a, map_location="cpu")
        hp = ck.get("hyper_parameters", {}) if isinstance(ck, dict) else {}
    except Exception:
        hp = {}

    keys = [
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
    ]
    for k in keys:
        if k in hp and hp[k] is not None:
            setattr(args, k, hp[k])
    return args


def load_controller(ckpt_path, dm, suite, args):
    loss_config = {
        "u_coef_in_training": getattr(args, "u_coef_in_training", 5e-1),
        "safe_classification_weight": getattr(args, "safe_classification_weight", 20.0),
        "unsafe_classification_weight": getattr(args, "unsafe_classification_weight", 20.0),
        "descent_violation_weight": getattr(args, "descent_violation_weight", 2.0),
        "hdot_divergence_weight": getattr(args, "hdot_divergence_weight", 2e-2),
    }
    with _torch_load_weights_only_false():
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
        return data_x, goal_mask.bool(), safe_mask.bool(), unsafe_mask.bool(), boundary_mask.bool()

    # Fallback path: sample q uniformly and label with system masks.
    ul, ll = dm.state_limits
    ul = ul.detach().clone().float().reshape(1, -1)
    ll = ll.detach().clone().float().reshape(1, -1)
    hi = torch.maximum(ul, ll)
    lo = torch.minimum(ul, ll)
    q = lo + torch.rand(int(n), dm.n_dims) * (hi - lo)
    data_x = dm.complete_sample_with_observations(q, num_samples=int(n))
    safe_mask = dm.safe_mask(q).bool()
    unsafe_mask = dm.unsafe_mask(q).bool()
    goal_mask = torch.zeros_like(safe_mask)
    boundary_mask = torch.logical_not(torch.logical_or(safe_mask, unsafe_mask))
    return data_x, goal_mask, safe_mask, unsafe_mask, boundary_mask


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_a", required=True, help="baseline ckpt path")
    ap.add_argument("--ckpt_b", required=True, help="jvp ckpt path (or vice versa)")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=256)

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

    args = ap.parse_args()
    args = merge_args_from_ckpt(args)

    dm, suite = build_dynamics_and_suite(args)
    ctrl_a = load_controller(args.ckpt_a, dm, suite, args)
    ctrl_b = load_controller(args.ckpt_b, dm, suite, args)

    eps = float(ctrl_a.safe_level)  # 你说 safe 要 <= -0.1，这里就会是 0.1
    thr = -eps

    # Build one shared batch for both controllers.
    data_x, goal_mask, safe_mask, unsafe_mask, boundary_mask = build_eval_batch(ctrl_a, dm, int(args.n))
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
