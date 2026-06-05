#!/usr/bin/env python3
"""Smoke tests for CBF observation modes backed by legacy, RayLink g_phi, and oracle rays.

This script intentionally uses the real ArmEnv/ArmLidar/NeuralLidarCBFController
path. It is not a training-quality test; it only checks that each observation
mode can build observations, run h forward, and exercise the expected derivative
or guard behavior.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class SmokeSkip(RuntimeError):
    pass


@dataclass
class SmokeContext:
    torch: object
    env: object
    robot: object
    obstacle_robot: object
    dynamics_model: object
    datax: object
    q_ego: object
    q_obs: object
    qdot_obs: object
    args: argparse.Namespace


class DummyDataModule:
    state_label_metadata = None
    state_label_cache = ""

    def prepare_data(self):
        return None

    def setup(self, stage=None):
        return None

    def train_dataloader(self):
        raise RuntimeError("DummyDataModule is not expected to provide train data in this smoke test.")

    val_dataloader = train_dataloader
    test_dataloader = train_dataloader


def import_runtime():
    try:
        import numpy as np
        import torch

        from environment import ArmEnv
        from neural_cbf.controllers import NeuralLidarCBFController
        from neural_cbf.experiments import ExperimentSuite
        from neural_cbf.systems import ArmLidar
    except ModuleNotFoundError as exc:
        raise SmokeSkip(f"Missing runtime dependency: {exc.name}") from exc
    return np, torch, ArmEnv, ArmLidar, NeuralLidarCBFController, ExperimentSuite


def parse_vec3(text: str, name: str):
    vals = [float(x) for x in str(text).split(",")]
    if len(vals) != 3:
        raise ValueError(f"{name} must contain 3 comma-separated floats, got {text!r}.")
    return tuple(vals)


def parse_vec4(text: str, name: str):
    vals = [float(x) for x in str(text).split(",")]
    if len(vals) != 4:
        raise ValueError(f"{name} must contain 4 comma-separated floats, got {text!r}.")
    return tuple(vals)


def set_aux_qobs_qdot(datax, dm, q_obs, qdot_obs):
    aux_start = int(datax.shape[1]) - int(dm.state_aux_dims_in_dataset)
    q_start = aux_start + int(dm.sensor_aux_dims)
    qdot_start = q_start + int(dm.obstacle_q_dim)
    datax[:, q_start : q_start + int(dm.obstacle_q_dim)] = q_obs.to(device=datax.device, dtype=datax.dtype)
    datax[:, qdot_start : qdot_start + int(dm.obstacle_qdot_dim)] = qdot_obs.to(device=datax.device, dtype=datax.dtype)
    return datax


def assert_tensor(name: str, tensor, shape: Optional[tuple] = None):
    import torch

    if tensor is None:
        raise AssertionError(f"{name} is None.")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise AssertionError(f"{name} shape mismatch: got {tuple(tensor.shape)}, expected {tuple(shape)}.")
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains non-finite values.")


def expect_raises(name: str, fn: Callable[[], None], contains: str):
    try:
        fn()
    except Exception as exc:
        msg = str(exc)
        if contains not in msg:
            raise AssertionError(f"{name} raised the wrong error: {type(exc).__name__}: {msg}") from exc
        return
    raise AssertionError(f"{name} did not raise an error containing {contains!r}.")


def build_context(args: argparse.Namespace) -> SmokeContext:
    np, torch, ArmEnv, ArmLidar, _, _ = import_runtime()
    if not args.gphi_ckpt:
        raise ValueError("--gphi_ckpt is required because this smoke covers gphi and RayLink oracle modes.")
    if not os.path.exists(args.gphi_ckpt):
        raise FileNotFoundError(f"--gphi_ckpt does not exist: {args.gphi_ckpt}")

    ego_base_pos = parse_vec3(args.ego_base_pos, "ego_base_pos")
    ego_base_orn = parse_vec4(args.ego_base_orn, "ego_base_orn")
    obs_base_pos = parse_vec3(args.obs_base_pos, "obs_base_pos")
    obs_base_orn = parse_vec4(args.obs_base_orn, "obs_base_orn")

    env = ArmEnv(
        [args.robot_name],
        GUI=bool(args.gui),
        config_file="",
        obstacle_robot_name=args.obstacle_robot_name,
        obstacle_traj_path=args.obstacle_traj_path or None,
        obstacle_robot_base_pos=obs_base_pos,
        obstacle_robot_base_orn=obs_base_orn,
    )
    robot = env.robot_list[0]
    obstacle_robot = env.obstacle_robot
    if obstacle_robot is None:
        raise RuntimeError("Smoke test requires an obstacle Panda robot.")

    env.p.resetBasePositionAndOrientation(int(robot.robotId), ego_base_pos, ego_base_orn)
    env.p.resetBasePositionAndOrientation(int(obstacle_robot.robotId), obs_base_pos, obs_base_orn)

    nominal_params = {}
    dm = ArmLidar(
        nominal_params,
        dt=float(args.simulation_dt),
        controller_dt=float(args.controller_period),
        dis_threshold=float(args.dis_threshold),
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
    dm.set_goal(torch.tensor(robot.q0, dtype=torch.float32))

    B = int(args.batch_size)
    if B < 1:
        raise ValueError("--batch_size must be positive.")
    q0 = torch.tensor(robot.q0, dtype=torch.float32).reshape(1, -1).repeat(B, 1)
    jitter = torch.linspace(-0.015, 0.015, steps=B, dtype=torch.float32).reshape(B, 1)
    direction = torch.linspace(1.0, -1.0, steps=robot.body_dim, dtype=torch.float32).reshape(1, -1)
    q_ego = q0 + jitter * direction

    q_obs0 = torch.tensor(obstacle_robot.q0, dtype=torch.float32).reshape(1, -1).repeat(B, 1)
    q_obs = q_obs0 - 0.5 * jitter * direction
    qdot_obs = torch.linspace(0.01, 0.07, steps=obstacle_robot.body_dim, dtype=torch.float32).reshape(1, -1).repeat(B, 1)

    obstacle_robot.set_joint_position(obstacle_robot.body_joints, q_obs[0].detach().cpu().numpy())
    datax = dm.complete_sample_with_observations(q_ego, num_samples=B).float()
    datax = set_aux_qobs_qdot(datax, dm, q_obs, qdot_obs)

    return SmokeContext(
        torch=torch,
        env=env,
        robot=robot,
        obstacle_robot=obstacle_robot,
        dynamics_model=dm,
        datax=datax,
        q_ego=q_ego,
        q_obs=q_obs,
        qdot_obs=qdot_obs,
        args=args,
    )


def make_controller(ctx: SmokeContext, mode: str, train_use_fd: bool, include_qobs_dynamics: bool = False):
    _, _, _, _, NeuralLidarCBFController, ExperimentSuite = import_runtime()
    baseline = mode == "legacy_oracle"
    datamodule = DummyDataModule()
    if mode == "raylink_cached_oracle":
        datamodule.state_label_cache = "__smoke_cached_raylink_oracle__"
    controller = NeuralLidarCBFController(
        ctx.dynamics_model,
        [{}],
        datamodule,
        ExperimentSuite([]),
        safe_level=0.1,
        unsafe_level=0.1,
        cbf_hidden_layers=1,
        cbf_hidden_size=32,
        cbf_alpha=1.0,
        cbf_relaxation_penalty=5000.0,
        feature_dim=16,
        per_feature_dim=16,
        learn_shape_epochs=0,
        loss_config={
            "u_coef_in_training": 0.5,
            "safe_classification_weight": 20.0,
            "unsafe_classification_weight": 20.0,
            "descent_violation_weight": 2.0,
            "hdot_divergence_weight": 0.0,
            "epsilon": 0.0,
        },
        all_hparams=argparse.Namespace(),
        use_bn=False,
        ab_mode="B_with_normal",
        baseline=baseline,
        obs_backend="gphi" if mode == "gphi" else "raw",
        cbf_obs_mode=mode,
        gphi_ckpt=ctx.args.gphi_ckpt if mode in ("gphi", "raylink_oracle", "raylink_cached_oracle") else "",
        gphi_hit_threshold=float(ctx.args.gphi_hit_threshold),
        gphi_hit_temp=float(ctx.args.gphi_hit_temp),
        gphi_freeze=True,
        gphi_include_qobs_dynamics=bool(include_qobs_dynamics),
        train_use_fd=bool(train_use_fd),
        use_neural_actor=False,
    )
    controller.eval()
    return controller


def test_legacy_oracle(ctx: SmokeContext):
    ctrl = make_controller(ctx, "legacy_oracle", train_use_fd=True)
    q_ego, q_obs, _, aux = ctrl.parse_state_from_datax(ctx.datax)
    obs = ctrl.build_observation(datax=ctx.datax, q_ego=q_ego, q_obs=q_obs, aux=aux)
    assert_tensor("legacy_oracle obs", obs, (ctx.datax.shape[0], ctx.dynamics_model.o_dims_in_dataset))
    h = ctrl.h(ctx.datax)
    assert_tensor("legacy_oracle h", h, (ctx.datax.shape[0], 1))


def test_gphi_static(ctx: SmokeContext):
    torch = ctx.torch
    ctrl = make_controller(ctx, "gphi", train_use_fd=False, include_qobs_dynamics=False)
    q_ego, q_obs, _, aux = ctrl.parse_state_from_datax(ctx.datax)
    obs = ctrl.build_observation(datax=ctx.datax, q_ego=q_ego, q_obs=q_obs, aux=aux)
    expected_obs_dim = int(ctrl.g_phi.num_rays) * 3
    assert_tensor("gphi static obs", obs, (ctx.datax.shape[0], expected_obs_dim))
    h = ctrl.h(ctx.datax)
    assert_tensor("gphi static h", h, (ctx.datax.shape[0], 1))

    qe = q_ego.detach().clone().requires_grad_(True)
    qo = q_obs.detach().clone()
    h_chain = ctrl.h_from_state(qe, qo, datax=ctx.datax.detach())
    dH_dqe = torch.autograd.grad(h_chain.sum(), qe, create_graph=True, retain_graph=True, allow_unused=False)[0]
    assert_tensor("gphi static dH_dqego", dH_dqe, tuple(qe.shape))


def test_gphi_dynamic(ctx: SmokeContext):
    torch = ctx.torch
    ctrl = make_controller(ctx, "gphi", train_use_fd=False, include_qobs_dynamics=True)
    q_ego, q_obs, qdot_obs, _ = ctrl.parse_state_from_datax(ctx.datax)
    qe = q_ego.detach().clone().requires_grad_(True)
    qo = q_obs.detach().clone().requires_grad_(True)
    h_chain = ctrl.h_from_state(qe, qo, datax=ctx.datax.detach())
    dH_dqe, dH_dqo = torch.autograd.grad(
        h_chain.sum(),
        (qe, qo),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )
    assert_tensor("gphi dynamic dH_dqego", dH_dqe, tuple(qe.shape))
    assert_tensor("gphi dynamic dH_dqobs", dH_dqo, tuple(qo.shape))
    lf_direct = torch.sum(dH_dqo * qdot_obs.to(device=qo.device, dtype=qo.dtype), dim=-1, keepdim=True)
    assert_tensor("gphi dynamic direct Lf_h", lf_direct, (ctx.datax.shape[0], 1))

    V, Lf_h, Lg_h, _ = ctrl.V_with_lie_derivatives(ctx.datax, data_jacobian=())
    assert_tensor("gphi dynamic V", V, (ctx.datax.shape[0],))
    assert_tensor("gphi dynamic controller Lf_h", Lf_h, (ctx.datax.shape[0], 1, 1))
    assert_tensor("gphi dynamic controller Lg_h", Lg_h, (ctx.datax.shape[0], 1, ctx.dynamics_model.n_controls))


def test_raylink_oracle_fd(ctx: SmokeContext):
    torch = ctx.torch
    ctrl = make_controller(ctx, "raylink_oracle", train_use_fd=True)
    datax = ctx.datax[:1]
    q_ego, q_obs, _, aux = ctrl.parse_state_from_datax(datax)
    obs = ctrl.build_observation(datax=datax, q_ego=q_ego, q_obs=q_obs, aux=aux)
    expected_obs_dim = int(ctrl.g_phi.num_rays) * 3
    assert_tensor("raylink_oracle obs", obs, (1, expected_obs_dim))
    h0 = ctrl.h(datax)
    assert_tensor("raylink_oracle h", h0, (1, 1))

    dq = torch.full((1, ctx.dynamics_model.n_controls), 1e-3, dtype=datax.dtype, device=datax.device)
    datax_next = ctx.dynamics_model.batch_lookahead(datax, dq, data_jacobian=())
    h1 = ctrl.h(datax_next)
    fd_hdot = (h1 - h0) / float(ctx.dynamics_model.controller_dt)
    assert_tensor("raylink_oracle one-step FD hdot", fd_hdot, (1, 1))

    h_fd, J_fd, _ = ctrl.h_with_jacobian(datax, data_jacobian=())
    assert_tensor("raylink_oracle h_with_jacobian h", h_fd, (1, 1))
    assert_tensor("raylink_oracle h_with_jacobian J", J_fd, (1, 1, ctx.dynamics_model.n_controls))


def test_raylink_oracle_no_train_use_fd_guard(ctx: SmokeContext):
    expect_raises(
        "raylink_oracle no_train_use_fd guard",
        lambda: make_controller(ctx, "raylink_oracle", train_use_fd=False),
        "train_use_fd=True only",
    )


def cached_raylink_datax(ctx: SmokeContext, ctrl, datax):
    torch = ctx.torch
    q_ego, q_obs, _, _ = ctrl.parse_state_from_datax(datax)
    with torch.no_grad():
        geom = ctrl.g_phi.compute_geometry(q_ego, q_obs)
        points = geom["ray_origins_W"] + float(ctrl.g_phi.r_max) * geom["ray_dirs_W"]
        points = points.reshape(points.shape[0], -1, 3)
        target_points = int(ctx.dynamics_model.o_dims_in_dataset // 3)
        if points.shape[1] >= target_points:
            idx = torch.linspace(0, points.shape[1] - 1, steps=target_points, device=points.device).round().long()
            points = torch.index_select(points, dim=1, index=idx)
        else:
            idx = torch.arange(target_points, device=points.device).long() % points.shape[1]
            points = torch.index_select(points, dim=1, index=idx)
        obs_flat = points.reshape(points.shape[0], -1).to(device=datax.device, dtype=datax.dtype)
    cached = datax.detach().clone()
    obs_start = int(ctx.dynamics_model.n_dims)
    obs_end = obs_start + int(ctx.dynamics_model.o_dims_in_dataset)
    cached[:, obs_start:obs_end] = obs_flat
    return cached


def test_raylink_cached_oracle_fd(ctx: SmokeContext):
    torch = ctx.torch
    ctrl = make_controller(ctx, "raylink_cached_oracle", train_use_fd=True)
    datax = cached_raylink_datax(ctx, ctrl, ctx.datax[:1])
    q_ego, q_obs, _, aux = ctrl.parse_state_from_datax(datax)
    obs = ctrl.build_observation(datax=datax, q_ego=q_ego, q_obs=q_obs, aux=aux)
    assert_tensor("raylink_cached_oracle obs", obs, (1, ctx.dynamics_model.o_dims_in_dataset))
    h0 = ctrl.h(datax)
    assert_tensor("raylink_cached_oracle h", h0, (1, 1))

    dq = torch.full((1, ctx.dynamics_model.n_controls), 1e-3, dtype=datax.dtype, device=datax.device)
    datax_next = ctx.dynamics_model.batch_lookahead(datax, dq, data_jacobian=())
    h1 = ctrl.h(datax_next)
    fd_hdot = (h1 - h0) / float(ctx.dynamics_model.controller_dt)
    assert_tensor("raylink_cached_oracle one-step FD hdot", fd_hdot, (1, 1))

    h_fd, J_fd, _ = ctrl.h_with_jacobian(datax, data_jacobian=())
    assert_tensor("raylink_cached_oracle h_with_jacobian h", h_fd, (1, 1))
    assert_tensor("raylink_cached_oracle h_with_jacobian J", J_fd, (1, 1, ctx.dynamics_model.n_controls))


def test_raylink_cached_oracle_no_train_use_fd_guard(ctx: SmokeContext):
    expect_raises(
        "raylink_cached_oracle no_train_use_fd guard",
        lambda: make_controller(ctx, "raylink_cached_oracle", train_use_fd=False),
        "train_use_fd=True only",
    )


def test_missing_qobs_guard(ctx: SmokeContext):
    ctrl = make_controller(ctx, "gphi", train_use_fd=False)
    original = ctrl.dynamics_model.get_obstacle_q_from_datax
    ctrl.dynamics_model.get_obstacle_q_from_datax = lambda datax: None
    try:
        expect_raises(
            "missing q_obs guard",
            lambda: ctrl.h(ctx.datax),
            "requires q_obs",
        )
    finally:
        ctrl.dynamics_model.get_obstacle_q_from_datax = original


def test_missing_qdot_guard(ctx: SmokeContext):
    ctrl = make_controller(ctx, "gphi", train_use_fd=False, include_qobs_dynamics=True)
    original = ctrl.dynamics_model.get_obstacle_meta_from_datax
    ctrl.dynamics_model.get_obstacle_meta_from_datax = lambda datax: (None, None, None)
    try:
        expect_raises(
            "missing qdot_obs guard",
            lambda: ctrl.V_with_lie_derivatives(ctx.datax, data_jacobian=()),
            "requires qdot_obs",
        )
    finally:
        ctrl.dynamics_model.get_obstacle_meta_from_datax = original


def run_test(name: str, fn: Callable[[], None]) -> bool:
    try:
        fn()
    except Exception as exc:
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        return False
    print(f"[PASS] {name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test legacy_oracle/gphi/raylink_oracle/raylink_cached_oracle CBF modes.")
    parser.add_argument("--gphi_ckpt", required=True, help="RayLink g_phi checkpoint used for metadata/FK.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--robot_name", default="panda")
    parser.add_argument("--obstacle_robot_name", default="panda")
    parser.add_argument("--obstacle_traj_path", default="")
    parser.add_argument("--ego_base_pos", default="0.0,-0.25,0.0")
    parser.add_argument("--ego_base_orn", default="0.0,0.0,0.0,1.0")
    parser.add_argument("--obs_base_pos", default="0.0,0.25,0.0")
    parser.add_argument("--obs_base_orn", default="0.0,0.0,1.0,0.0")
    parser.add_argument("--simulation_dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--controller_period", type=float, default=1.0 / 30.0)
    parser.add_argument("--dis_threshold", type=float, default=0.02)
    parser.add_argument("--obstacle_horizon_s", type=float, default=0.2)
    parser.add_argument("--n_observation", type=int, default=32)
    parser.add_argument("--n_observation_dataset", type=int, default=64)
    parser.add_argument("--gphi_hit_threshold", type=float, default=0.5)
    parser.add_argument("--gphi_hit_temp", type=float, default=0.1)
    parser.add_argument("--gui", action="store_true", default=False)
    args = parser.parse_args()

    try:
        ctx = build_context(args)
    except SmokeSkip as exc:
        print(f"[SKIP] {exc}")
        return 77

    tests = [
        ("legacy_oracle build_observation + h", lambda: test_legacy_oracle(ctx)),
        ("gphi static build_observation + h + dH/dq_ego", lambda: test_gphi_static(ctx)),
        ("gphi dynamic dH/dq_obs @ qdot_obs", lambda: test_gphi_dynamic(ctx)),
        ("raylink_oracle build_observation + h + FD one-step", lambda: test_raylink_oracle_fd(ctx)),
        ("raylink_oracle + no_train_use_fd guard", lambda: test_raylink_oracle_no_train_use_fd_guard(ctx)),
        ("raylink_cached_oracle build_observation + h + FD one-step", lambda: test_raylink_cached_oracle_fd(ctx)),
        ("raylink_cached_oracle + no_train_use_fd guard", lambda: test_raylink_cached_oracle_no_train_use_fd_guard(ctx)),
        ("missing q_obs guard", lambda: test_missing_qobs_guard(ctx)),
        ("missing qdot_obs guard", lambda: test_missing_qdot_guard(ctx)),
    ]

    ok = True
    for name, fn in tests:
        ok = run_test(name, fn) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
