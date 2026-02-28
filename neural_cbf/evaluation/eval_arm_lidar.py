import os
import time
import argparse
import yaml

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import contextlib

import matplotlib.pyplot as plt
import json

import pybullet as p

from environment import ArmEnv

from neural_cbf.controllers import NeuralLidarCBFController
from neural_cbf.datamodules.episodic_datamodule import (
	EpisodicDataModule,
)
from neural_cbf.systems import ArmLidar
from neural_cbf.experiments import (
	ExperimentSuite,
	BFContourExperiment,
	LidarRolloutExperiment,
)
from neural_cbf.training.utils import current_git_hash
from neural_cbf.systems.utils import grav, Scenario, cartesian_to_spherical, spherical_to_cartesian

from PIL import Image
import cv2

# batch_size = 1


def init_val(path, args):
	# initialize models and parameters for loaded controllers
	nominal_params = {}
	scenarios = [
		nominal_params,
	]
	# Define environment and agent
	config_file = ''
	# config_file = '../../models/env_file/panda_100_8_v1_refined.npz'
	gui_flag = getattr(args, 'gui', 1)
	environment = ArmEnv(
		[args.robot_name],
		GUI=gui_flag,
		config_file=config_file,
		obstacle_robot_name=getattr(args, "obstacle_robot_name", "panda"),
		obstacle_traj_path=getattr(args, "obstacle_traj_path", "data/obstacle_trajs/panda_trajs.npz"),
	)
	robot = environment.robot_list[0]

	# Define the dynamics model
	dynamics_model = ArmLidar(
		nominal_params,
		dis_threshold=args.dis_threshold,
		dt=args.simulation_dt,
		controller_dt=args.controller_period,
		n_obs=args.n_observation,
		point_dim=args.point_dim,
		add_normal=bool('norm' in args.dataset_name),
		point_in_dataset_pc=args.n_observation_dataset,
		list_sensor=robot.body_joints,
		env=environment,
		robot=robot,
		observation_type=args.observation_type,
		include_point_velocity=getattr(args, "include_point_velocity", False),
		obstacle_horizon_s=getattr(args, "obstacle_horizon_s", 0.2),
	)
	dynamics_model.compute_linearized_controller(None)

	# start_x = torch.tensor(np.load(config_file)['init_configs'][0]).unsqueeze(0)
	# goal_state = torch.tensor(np.load(config_file)['goal_configs'][0])

	# Define goal_state
	# If user provides an end-effector goal (xyz), use IK; otherwise choose a goal away from q0.
	goal_xyz = getattr(args, "goal_xyz", None)
	if goal_xyz is not None:
		goal_xyz = [float(goal_xyz[0]), float(goal_xyz[1]), float(goal_xyz[2])]
		try:
			ik = p.calculateInverseKinematics(robot.robotId, robot.body_joints[-1], goal_xyz)
			goal_state = torch.tensor(ik[:dynamics_model.n_dims]).float()
		except Exception:
			goal_state = torch.tensor(robot.q0).float()
	else:
		# A fixed "reachable" IK target that is typically not equal to q0
		try:
			ik = p.calculateInverseKinematics(robot.robotId, robot.body_joints[-1], [0.55, 0.0, 0.45])
			goal_state = torch.tensor(ik[:dynamics_model.n_dims]).float()
		except Exception:
			goal_state = torch.tensor(robot.q0).float()
	# Set and report goal
	dynamics_model.set_goal(goal_state)
	print(f"[GOAL] goal_q={goal_state.tolist()}")

	# Initialize the DataModule
	initial_conditions = [tuple(robot.body_range[i]) for i in range(robot.body_dim)]
	data_module = None #EpisodicDataModule(
	# 	dynamics_model,
	# 	initial_conditions,
	# 	total_point=args.n_observation_dataset,
	# 	max_episode=args.max_episode,
	# 	trajectories_per_episode=args.trajectories_per_episode,
	# 	trajectory_length=args.trajectory_length,
	# 	fixed_samples=args.fixed_samples,
	# 	val_split=args.val_split,
	# 	batch_size=args.batch_size,
	# 	noise_level=args.noise_level,
	# 	quotas={"safe": args.safe_portion, "goal": args.goal_portion, "unsafe": args.unsafe_portion},
	# 	name=args.dataset_name,
	# 	shuffle=False,
	# )

	# start_x = torch.tensor([
	# 	[0.00887519, 0.50546576, -0.69052917, -2.2909179, 2.95208592, 2.29793418, 2.93001438] # + [0 for _ in range(8)],
	# # 	[0.00887519, -0.50546576, -0.69052917, -2.2909179, 2.95208592, 2.29793418, 2.93001438] + [0 for _ in range(
	# # 		# 8)],
	# # 		[-2.60887519, -1.30546576, -1.69052917, -2.2909179, 2.95208592, 3.59793418, 2.93001438]
	# 			# dynamics_model.o_dims + dynamics_model.state_aux_dims)],
	# # 		[0.00887519, -0.50546576, -0.69052917, -2.2909179, 2.95208592, 3.59793418, 2.93001438] + [0 for _ in range(
	# # 		dynamics_model.o_dims + dynamics_model.state_aux_dims)],
	# ])
	# # start_x = dynamics_model.sample_safe(1)
	# # start_x = dynamics_model.sample_boundary(1, data_collection=True)
	ul, ll = dynamics_model.state_limits
	goal_q = dynamics_model.goal_state[:dynamics_model.n_dims].detach().clone().float()
	# Try a few random starts and pick one far from the goal
	best_q = None
	best_dist = -1.0
	for _ in range(50):
		q_try = torch.lerp(ll, ul, torch.rand_like(ll)).reshape(1, -1).float()
		d = torch.norm(q_try.squeeze(0) - goal_q).item()
		if d > best_dist:
			best_dist = d
			best_q = q_try
	start_x = best_q
	start_x = dynamics_model.complete_sample_with_observations(start_x, num_samples=start_x.shape[0])
	print(f"[START] start_q={start_x[0, :dynamics_model.n_dims].tolist()}  dist_to_goal={best_dist:.3f}")

	x_idx = 0
	y_idx = 2
	rollout_experiment = LidarRolloutExperiment(
		"Rollout",
		start_x,
		x_idx,
		f"$\\theta_{x_idx}$",
		y_idx,
		f"$\\theta_{y_idx}$",
		scenarios=scenarios,
		n_sims_per_start=1,
		t_sim=20,
		compare_nominal=False,
	)

	default_state = start_x
	# default_state = dynamics_model.sample_boundary(1).squeeze()
	# # default_state = dynamics_model.complete_sample_with_observations(dynamics_model.goal_state.reshape(1, -1),
	# # 																 num_samples=1).squeeze()

	# Define the experiment suite
	h_contour_experiment = BFContourExperiment(
		"h_Contour",
		domain=[tuple(robot.body_range[x_idx]), tuple(robot.body_range[y_idx])],
		n_grid=40,
		x_axis_index=x_idx,
		y_axis_index=y_idx,
		x_axis_label=f"$\\theta_{x_idx}$",
		y_axis_label=f"$\\theta_{y_idx}$",
		default_state=default_state,
		plot_unsafe_region=True,
	)

	experiment_suite = ExperimentSuite([rollout_experiment, h_contour_experiment])

	loss_config = {
		"u_coef_in_training": getattr(args, "u_coef_in_training", 5e-1),
		"safe_classification_weight": getattr(args, "safe_classification_weight", 20.0),
		"unsafe_classification_weight": getattr(args, "unsafe_classification_weight", 20.0),
		"descent_violation_weight": getattr(args, "descent_violation_weight", 2.0),
		"hdot_divergence_weight": getattr(args, "hdot_divergence_weight", 2e-2),
	}
	# PyTorch >= 2.6 changed torch.load default to weights_only=True, which can break
	# older Lightning .ckpt files that contain pickled training state. This checkpoint
	# is locally produced, so we force weights_only=False during load.
	@contextlib.contextmanager
	def _torch_load_weights_only_false():
		_orig_torch_load = torch.load

		def _patched_torch_load(*args, **kwargs):
			kwargs.setdefault("weights_only", False)
			return _orig_torch_load(*args, **kwargs)

		torch.load = _patched_torch_load
		try:
			yield
		finally:
			torch.load = _orig_torch_load

	with _torch_load_weights_only_false():
		return NeuralLidarCBFController.load_from_checkpoint(path, dynamics_model=dynamics_model, scenarios=scenarios,
														 datamodule=data_module, experiment_suite=experiment_suite,
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
														 map_location='cpu')


def vis_traj_rollout(controller: NeuralLidarCBFController):
	"""
	Visualize trajectories from two-link-arm RolloutStateSpaceExperiments.
	"""
	# Tweak experiment params
	controller.experiment_suite.experiments[0].t_sim = 9.

	# Run the experiments and save the results
	controller.experiment_suite.experiments[0].run_and_plot(
		controller, display_plots=True
	)
	print('finished')


def vis_CBF_contour(controller: NeuralLidarCBFController):
	# Run the experiments and save the results
	controller.experiment_suite.experiments[1].run_and_plot(
		controller_under_test=controller, display_plots=True
	)
	print('finished CBF contour')


@torch.no_grad()
def check_evaluation(controller: NeuralLidarCBFController):
	controller.datamodule.prepare_data()
	# # just check below z=0
	# below_z = 0
	# training_unsafe_mask = torch.nonzero(controller.datamodule.x_training_mask['unsafe']).squeeze()
	# for i in range(training_unsafe_mask.shape[0]):
	# 	q = controller.datamodule.x_training[training_unsafe_mask[i], :7]
	# 	if controller.dynamics_model.robot.forward_kinematics([-2], q)[0][0][2] < 0.05:
	# 		below_z += 1
	# 	if i %500 == 0:
	# 		print(f"below z: {below_z} / {i}")

	batch_size = 50
	for i in range(30):
		init_idx = i * batch_size + 1000
		end_idx = init_idx + batch_size
		data_x, goal_mask, safe_mask, unsafe_mask, boundary_mask, JP, JR = controller.datamodule.training_data[torch.arange(init_idx, end_idx)]
		data_x = data_x[:, :-1]

		eps = controller.safe_level
		h_value = controller.h(data_x)

		#   1.) h < 0 in the safe region
		safe_violation = F.relu(eps + h_value[safe_mask]).squeeze()
		safe_h_term = 20 * safe_violation.mean()
		safe_h_acc = (safe_violation <= eps).sum() / safe_violation.nelement()

		#   2.) h > 0 in the unsafe region
		unsafe_violation = F.relu(eps - h_value[unsafe_mask]).squeeze()
		unsafe_h_term = 20 * unsafe_violation.mean()
		unsafe_h_acc = (unsafe_violation <= eps).sum() / unsafe_violation.nelement()
		# print(f"safe_h_acc: {safe_h_acc}, unsafe_h_acc: {unsafe_h_acc}, safe_h_term: {safe_h_term}, unsafe_h_term: {unsafe_h_term}")

		#   3.) hdot + alpha * h < 0 in all regions
		_, Lf_V, Lg_V, _ = controller.V_with_lie_derivatives(data_x, (JP, JR))

		Lg_V_no_grad = Lg_V.detach().clone().squeeze(1)  # bs * n_control

		qp_sol = controller.u(data_x)[0]
		x_next = controller.dynamics_model.batch_lookahead(data_x, qp_sol * controller.dynamics_model.dt, data_jacobian=(JP, JR))
		hdot_simulated = (controller.h(x_next) - h_value) / controller.dynamics_model.dt

		hdot = hdot_simulated
		alpha = controller.clf_lambda # torch.where(h < 0, 2 * self.clf_lambda, self.clf_lambda).type_as(x)
		qp_relaxation = F.relu(hdot + torch.multiply(alpha, h_value))
		print(f"qp_relaxation: {qp_relaxation.mean():.4f}, qp_relaxation: {qp_relaxation.max():.4f}, "
			  f"safe: {(qp_relaxation[safe_mask] <= 0).sum() /  qp_relaxation[safe_mask].nelement():.4f}, "
			  f"unsafe: {(qp_relaxation[unsafe_mask] <= 0).sum() /  qp_relaxation[unsafe_mask].nelement():.4f}, "
			  f"boundary: {(qp_relaxation[boundary_mask] <= 0).sum() /  qp_relaxation[boundary_mask].nelement():.4f}")
		# print(f"relaxation_safe: {qp_relaxation[safe_mask].mean()}, relaxation_unsafe: {qp_relaxation[unsafe_mask].mean()}, "
		# 	  f"relaxation_boundary: {qp_relaxation[boundary_mask].mean()}")

@torch.no_grad()
def vis_misclassification(controller: NeuralLidarCBFController, log_path: str):
	controller.datamodule.prepare_data()
	init_idx = 0
	end_idx = 20
	data_x, goal_mask, safe_mask, unsafe_mask, boundary_mask, JP, JR, io_label = controller.datamodule.training_data[torch.arange(init_idx, end_idx)]
	x = controller.dynamics_model.datax_to_x(data_x, io_label)
	# x = controller.dynamics_model.datax_lookahead_prepare(data_x, data_lookahead)[0, :, :]

	eps = controller.safe_level
	h_value = controller.h(x)

	#   1.) h < 0 in the safe region
	safe_violation = F.relu(eps + h_value).squeeze()
	# safe_h_term = (1 / eps) * safe_violation[safe_mask].mean()
	# safe_h_acc = (safe_violation[safe_mask] <= eps).sum() / safe_violation[safe_mask].nelement()

	#   2.) h > 0 in the unsafe region
	unsafe_violation = F.relu(eps - h_value).squeeze()
	# unsafe_h_term = (1 / eps) * unsafe_violation[unsafe_mask].mean()
	# unsafe_h_acc = (unsafe_violation[unsafe_mask] <= eps).sum() / unsafe_violation[unsafe_mask].nelement()

	log_fig_path = log_path + '/data_classification/'
	if not os.path.exists(log_fig_path):
		os.makedirs(log_fig_path)
		os.makedirs(log_fig_path + 'gt_safe/')
		os.makedirs(log_fig_path + 'gt_unsafe/')
		os.makedirs(log_fig_path + 'safe/')
		os.makedirs(log_fig_path + 'unsafe/')

	for idx in range(10):
	# 	if safe_violation[idx] < eps and safe_mask[idx]:
	# 		draw_environment(controller, x[idx], idx + init_idx, log_fig_path + 'safe/')
		if unsafe_violation[idx] < eps and unsafe_mask[idx]:
			draw_environment(controller, x[idx], idx + init_idx, log_fig_path + 'unsafe/')
	# exit()

	# safe misclassification
	for idx in range(x.shape[0]):
		# if safe_violation[idx] > eps and safe_mask[idx]:
		# 	draw_environment(controller, x[idx], idx + init_idx, log_fig_path + 'gt_safe/')
		if unsafe_violation[idx] > eps and unsafe_mask[idx]:
			draw_environment(controller, x[idx], idx + init_idx, log_fig_path + 'gt_unsafe/')
			# break


	# print(safe_mask)
	# print(unsafe_mask)
	# print(safe_violation.squeeze())
	# print(unsafe_violation.squeeze())
	pass

@torch.no_grad()
def statistics_safe_level(controller: NeuralLidarCBFController):
	controller.datamodule.prepare_data()
	init_idx = 0
	end_idx = 20
	data_x, goal_mask, safe_mask, unsafe_mask, data_lookahead = controller.datamodule.training_data[
		torch.arange(init_idx, end_idx)]
	x = controller.dynamics_model.datax_to_x(data_x)
	# safe_h_acc = (safe_violation[safe_mask] <= eps).sum() / safe_violation[safe_mask].nelement()

def draw_environment(controller: NeuralLidarCBFController, x: torch.Tensor, idx: int, fig_path):
	controller.dynamics_model.env.reset_env(np.array([]), tidy_env=True)

	robot = controller.dynamics_model.robot
	q = x[:controller.dynamics_model.n_dims]
	robot.set_joint_position(robot.body_joints, q)

	p_p = [torch.Tensor(p.getLinkState(robot.robotId, sensor_idx)[4]) for sensor_idx in controller.dynamics_model.list_sensor]
	p_r = [torch.Tensor(p.getMatrixFromQuaternion(p.getLinkState(robot.robotId, sensor_idx)[5])).reshape(3, 3) for
		   sensor_idx in controller.dynamics_model.list_sensor]
	O = x[controller.dynamics_model.n_dims:].reshape(-1, controller.dynamics_model.ray_per_sensor, controller.dynamics_model.point_dims)
	if controller.dynamics_model.point_dims == 4:
		G = [p_p[i] + spherical_to_cartesian(O[i, :, :3]) @ p_r[i].T for i in range(len(controller.dynamics_model.list_sensor))]
	else:
		G = [p_p[i] + O[i, :, :3] @ p_r[i].T for i in range(len(controller.dynamics_model.list_sensor))]
	G = torch.vstack(G).tolist()

	for pt in G:
		vid = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 1, 1])
		p.createMultiBody(baseVisualShapeIndex=vid, basePosition=pt)

	width = 1280
	height = 720
	total_frame = 30
	video = []
	for i_frame in range(total_frame):
		projectionMatrix = p.computeProjectionMatrixFOV(
			fov=20,
			aspect=width / height,
			nearVal=0.1,
			farVal=50
		)
		viewMatrix = p.computeViewMatrix(
			cameraEyePosition=[3 * np.cos(i_frame/total_frame * 2 * np.pi), 3 * np.sin((i_frame/total_frame * 2 * np.pi)), 1.5],
			cameraTargetPosition=[0, 0, 0.5],
			cameraUpVector=[0, 0, 1]
		)
		width, height, rgbImg, depthImg, segImg = p.getCameraImage(
			width=width,
			height=height,
			viewMatrix=viewMatrix,
			projectionMatrix=projectionMatrix,
			renderer=p.ER_BULLET_HARDWARE_OPENGL
		)
		video.append(rgbImg)
		im = Image.fromarray(rgbImg)

		if not os.path.exists(f"{fig_path}/{idx}/"):
			os.makedirs(f"{fig_path}/{idx}/")
		im.save(f"{fig_path}/{idx}/{i_frame}.png")

	name = idx
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(f'{fig_path}/{name}.mp4', fourcc, 24, (width, height))
	for i_img, img in enumerate(video):
		img_new = cv2.imread(f"{fig_path}/{idx}/{i_img}.png")
		out.write(img_new)
	out.release()


def statistics_robustness_observation(controller: NeuralLidarCBFController):
# 	controller.datamodule.prepare_data()
# 	init_idx = 1000
# 	x, goal_mask, safe_mask, unsafe_mask, lookahead = controller.datamodule.training_data[
# 		torch.arange(init_idx, init_idx + args.batch_size)]
	batch_size = 256
	N_test = 20
	q = torch.Tensor(np.random.uniform(low=controller.dynamics_model.state_limits[1],
									   high=controller.dynamics_model.state_limits[0],
									   size=(batch_size, controller.dynamics_model.n_dims)))
	dq = torch.Tensor(N_test, controller.dynamics_model.n_dims).uniform_(1e-3, 2e-3)

	results = []
	for i in range(N_test):
		x = controller.dynamics_model.complete_sample_with_observations(q + dq[i, :], batch_size)
		results.append(controller.h(x))

	results = torch.cat(results, dim=1).detach().numpy()
	# print(np.mean(results, axis=1))
	# print(np.std(results, axis=1))

	plt.figure(figsize=(9, 3))
	plt.subplot(121)
	plt.hist(np.std(results, axis=1), 10)
	plt.yscale("log")
	# plt.xlim(0., 0.025)
	plt.title("std distribution")
	plt.grid(True)

	plt.subplot(122)
	plt.hist(results.max(axis=1) - results.min(axis=1), 10)
	plt.yscale("log")
	plt.xlim(0., 0.07)
	plt.title("(max-min) distribution")
	plt.grid(True)

	plt.show()


# ---- Moving obstacle rollout helpers ----
import numpy as np
import torch


def _make_obstacle_traj_from_current(
    env: ArmEnv,
    obstacle_ids,
    seed: int = 0,
    amp_range=(0.03, 0.12),
    omega_range=(0.3, 1.2),
):
    """Create simple sinusoidal trajectories for each obstacle based on current positions."""
    rng = np.random.default_rng(seed)
    p_ = env.p
    obstacle_ids = list(obstacle_ids or [])
    if len(obstacle_ids) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    base = np.array([p_.getBasePositionAndOrientation(oid)[0] for oid in obstacle_ids], dtype=np.float32)
    direction = rng.normal(size=base.shape).astype(np.float32)
    norm = np.linalg.norm(direction, axis=1, keepdims=True) + 1e-8
    direction = direction / norm

    omega = rng.uniform(low=omega_range[0], high=omega_range[1], size=(base.shape[0],)).astype(np.float32)
    amp = rng.uniform(low=amp_range[0], high=amp_range[1], size=(base.shape[0],)).astype(np.float32)
    return base, direction, omega, amp


def _update_obstacles(env: ArmEnv, obstacle_ids, t: float, base, direction, omega, amp):
    """Move obstacles in-place each step using resetBasePositionAndOrientation."""
    p_ = env.p
    obstacle_ids = list(obstacle_ids or [])
    for i, oid in enumerate(obstacle_ids):
        pos0 = base[i]
        pos = pos0 + direction[i] * amp[i] * np.sin(omega[i] * t)
        _, orn = p_.getBasePositionAndOrientation(oid)
        p_.resetBasePositionAndOrientation(oid, pos.tolist(), orn)


def _get_eval_obstacle_ids(env: ArmEnv, robot_id: int = None, exclude_ids=None):
    """Return obstacle ids suitable for evaluation.

    Priority:
      1) Use env.obstacle_ids if available.
      2) If empty, fall back to scanning all bodies in the pybullet world.

    Filtering:
      - Exclude the robot itself.
      - Exclude plane/floor bodies when we can detect them.

    Note: Some env implementations accidentally include plane/floor (or even the robot) in
    `env.obstacle_ids`, which would cause an immediate "collision" at step 0.
    """
    p_ = env.p

    # 1) Prefer env.obstacle_ids
    raw_ids = list(getattr(env, "obstacle_ids", []) or [])

    # 2) Fallback: scan all bodies
    if not raw_ids:
        try:
            n = p_.getNumBodies()
            raw_ids = [p_.getBodyUniqueId(i) for i in range(n)]
        except Exception:
            raw_ids = []

    # Exclude specific ids (e.g., scene blocks) if requested
    exclude_set = set(int(x) for x in (exclude_ids or []) if x is not None)

    if not raw_ids:
        return []

    def _looks_like_plane_or_floor(body_id: int) -> bool:
        # 0 is often the ground plane in many pybullet scenes
        try:
            if int(body_id) == 0:
                # Confirm by geometry type if possible
                cs = p_.getCollisionShapeData(body_id, -1) or []
                for c in cs:
                    geom_type = c[2]
                    if geom_type == p_.GEOM_PLANE:
                        return True
        except Exception:
            pass

        # Try body name
        try:
            bi = p_.getBodyInfo(body_id)
            name = bi[1]
            if isinstance(name, (bytes, bytearray)):
                name = name.decode("utf-8", "ignore")
            name = str(name).lower()
            if "plane" in name or "floor" in name or "ground" in name:
                return True
        except Exception:
            pass

        # Collision geometry type is the most reliable
        try:
            cs = p_.getCollisionShapeData(body_id, -1) or []
            for c in cs:
                geom_type = c[2]
                fname = c[4]
                if isinstance(fname, (bytes, bytearray)):
                    fname = fname.decode("utf-8", "ignore")
                fname = str(fname).lower()
                if geom_type == p_.GEOM_PLANE:
                    return True
                if "plane" in fname or "floor" in fname or "ground" in fname:
                    return True
        except Exception:
            pass

        # Visual geometry type / mesh filenames as fallback
        try:
            vs = p_.getVisualShapeData(body_id) or []
            for v in vs:
                geom_type = v[2]
                fname = v[4]
                if isinstance(fname, (bytes, bytearray)):
                    fname = fname.decode("utf-8", "ignore")
                fname = str(fname).lower()
                if geom_type == p_.GEOM_PLANE:
                    return True
                if "plane" in fname or "floor" in fname or "ground" in fname:
                    return True
        except Exception:
            pass

        return False

    kept = []
    for oid in raw_ids:
        if exclude_set and int(oid) in exclude_set:
            continue
        if robot_id is not None and oid == robot_id:
            continue
        if _looks_like_plane_or_floor(oid):
            continue
        kept.append(oid)

    # If filtering removed everything, do NOT fall back to raw_ids.
    # Falling back can accidentally re-introduce the floor/plane body, which makes the robot
    # look "in collision" at step 0 and causes clean-start search to always fail.
    if not kept:
        return []

    return kept


def _min_distance_and_collision(env: ArmEnv, robot_id: int, obstacle_ids, distance: float = 2.0):
    """Return (min_distance, collided) between robot and obstacles.

    `obstacle_ids` should be a list that does NOT include the floor/plane.
    """
    p_ = env.p
    min_d = float("inf")
    for oid in obstacle_ids:
        pts = p_.getClosestPoints(bodyA=robot_id, bodyB=oid, distance=distance)
        # contactDistance is index 8 in pybullet getClosestPoints tuple
        for pp in pts:
            d = pp[8]
            if d < min_d:
                min_d = d
            if d < 0:
                return float(min_d), True

    if min_d == float("inf"):
        # No points returned: treat as far away
        min_d = distance
    return float(min_d), False


# ---- Remove all obstacles helper ----
def _remove_all_obstacles(env: ArmEnv, robot_id: int = None, exclude_ids=None):
    """Remove all non-floor obstacles from the pybullet world.

    Supports excluding specific body ids from removal.
    """
    p_ = env.p
    obstacle_ids = _get_eval_obstacle_ids(env, robot_id, exclude_ids=exclude_ids)
    removed = []
    for oid in obstacle_ids:
        try:
            p_.removeBody(int(oid))
            removed.append(int(oid))
        except Exception:
            pass

    # Also update env.obstacle_ids if present
    try:
        if hasattr(env, "obstacle_ids") and env.obstacle_ids is not None:
            env.obstacle_ids = [oid for oid in env.obstacle_ids if int(oid) not in set(removed)]
    except Exception:
        pass

    return removed



# ---- EE marker/utility helpers ----
def _find_ee_link_index(p_client, body_id: int) -> int:
	"""Best-effort EE link index for Panda-like arms."""
	try:
		preferred_link = ["panda_grasptarget", "panda_hand", "panda_link8", "ee", "gripper"]
		preferred_joint = ["panda_hand_joint", "hand", "ee"]
		nj = int(p_client.getNumJoints(int(body_id)))
		if nj <= 0:
			return -1
		cands = []
		for j in range(nj):
			ji = p_client.getJointInfo(int(body_id), int(j))
			jname = ji[1].decode("utf-8", "ignore") if isinstance(ji[1], (bytes, bytearray)) else str(ji[1])
			lname = ji[12].decode("utf-8", "ignore") if isinstance(ji[12], (bytes, bytearray)) else str(ji[12])
			cands.append((j, jname.lower(), lname.lower()))
		for key in preferred_link:
			for j, _, lname in cands:
				if key in lname:
					return int(j)
		for key in preferred_joint:
			for j, jname, _ in cands:
				if key in jname:
					return int(j)
		return int(nj - 1)
	except Exception:
		return -1


def _get_ee_pos(robot) -> np.ndarray:
	"""End-effector position for quick visualization/printing."""
	try:
		ee_link = _find_ee_link_index(p, int(robot.robotId))
		ls = p.getLinkState(robot.robotId, ee_link)
		return np.array(ls[4], dtype=np.float32)
	except Exception:
		return np.zeros((3,), dtype=np.float32)



def _spawn_marker(pos, rgba=(1, 0, 0, 0.8), radius=0.03) -> int:
	"""Spawn a visual marker sphere in GUI; returns body id."""
	try:
		vid = p.createVisualShape(p.GEOM_SPHERE, radius=float(radius), rgbaColor=list(rgba))
		bid = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vid, basePosition=list(pos))
		return int(bid)
	except Exception:
		return -1


# ---- Visual grasp helper for cross-pick ----
def _update_visual_grasp_block(p_client, arm_id: int, ee_link_index: int, block_id: int, grasp_state: dict,
                              dist_thresh: float = 0.05, ee_z_offset: float = -0.035,
                              z_align_thresh: float = 0.08, xy_align_thresh: float = 1.0,
                              grasp_z_max_offset: float = 1.0):
    """Visual-only grasp: when EE is close, disable block collisions and make block follow EE.

    grasp_state: dict that will store keys {"grabbed": bool}.
    """
    if block_id is None or int(block_id) < 0:
        return
    if arm_id is None or int(arm_id) < 0:
        return
    try:
        ls = p_client.getLinkState(int(arm_id), int(ee_link_index))
        ee_pos = np.array(ls[4], dtype=np.float32)
        ee_orn = ls[5]
    except Exception:
        return

    try:
        bpos, born = p_client.getBasePositionAndOrientation(int(block_id))
        bpos = np.array(bpos, dtype=np.float32)
    except Exception:
        return

    grabbed = bool(grasp_state.get("grabbed", False))

    ee_block_dist = float(np.linalg.norm(ee_pos - bpos))
    grasp_state["ee_block_dist"] = ee_block_dist
    ee_xy_dist = float(np.linalg.norm(ee_pos[:2] - bpos[:2]))
    ee_dz = float(abs(ee_pos[2] - bpos[2]))
    grasp_state["ee_xy_dist"] = ee_xy_dist
    grasp_state["ee_dz"] = ee_dz
    grasp_state["ee_z"] = float(ee_pos[2])
    grasp_state["block_z"] = float(bpos[2])

    # Trigger grasp only when close in 3D and reasonably aligned in Z
    if (
        (not grabbed)
        and (ee_block_dist <= float(dist_thresh))
        and (ee_dz <= float(z_align_thresh))
        and (ee_xy_dist <= float(xy_align_thresh))
        and (float(ee_pos[2]) <= float(bpos[2]) + float(grasp_z_max_offset))
    ):
        grasp_state["grabbed"] = True
        # Disable collisions for the block so it won't affect control/contacts
        try:
            p_client.setCollisionFilterGroupMask(int(block_id), -1, 0, 0)
        except Exception:
            pass

    # If grabbed, kinematically attach the block to EE
    if bool(grasp_state.get("grabbed", False)):
        tgt_pos = (ee_pos + np.array([0.0, 0.0, float(ee_z_offset)], dtype=np.float32)).tolist()
        try:
            p_client.resetBasePositionAndOrientation(int(block_id), tgt_pos, ee_orn)
        except Exception:
            pass

def _spawn_block(env: ArmEnv, pos_xyz, half=0.02, rgba=(0.2, 0.2, 0.9, 1.0), mass=0.0):
    p_ = env.p
    cshape = p_.createCollisionShape(p_.GEOM_BOX, halfExtents=[half, half, half])
    vshape = p_.createVisualShape(p_.GEOM_BOX, halfExtents=[half, half, half], rgbaColor=list(rgba))
    bid = p_.createMultiBody(
        baseMass=float(mass),
        baseCollisionShapeIndex=cshape,
        baseVisualShapeIndex=vshape,
        basePosition=[float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])],
        baseOrientation=[0, 0, 0, 1],
    )
    return int(bid)

def _smoothstep(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return s * s * (3.0 - 2.0 * s)

def _obstacle_ee_target_cross_pick(
    t: float,
    T: float,
    start_xyz,
    left_block_xyz,
    cross_jitter_amp: float = 0.018,
    cross_jitter_hz: float = 6.0,
    cross_window_ratio: float = 0.35,
):
    # start -> pregrasp -> descend -> hold
    t = float(np.clip(t, 0.0, T))
    s = t / max(T, 1e-6)

    pre = np.array([left_block_xyz[0], left_block_xyz[1], left_block_xyz[2] + 0.12], dtype=np.float32)
    grasp = np.array([left_block_xyz[0], left_block_xyz[1], left_block_xyz[2] + 0.03], dtype=np.float32)

    if s <= 0.55:
        w = _smoothstep(s / 0.55)
        xyz = (1 - w) * np.array(start_xyz, dtype=np.float32) + w * pre
    elif s <= 0.75:
        w = _smoothstep((s - 0.55) / 0.20)
        xyz = (1 - w) * pre + w * grasp
    else:
        xyz = grasp.copy()

    # non-smooth jitter window centered at s=0.5 (attacks FD stability)
    hw = 0.5 * float(np.clip(cross_window_ratio, 0.0, 1.0))
    if abs(s - 0.5) <= hw:
        sig = 1.0 if np.sin(2 * np.pi * float(cross_jitter_hz) * t) >= 0 else -1.0
        xyz = xyz + np.array([0.0, sig * float(cross_jitter_amp), 0.0], dtype=np.float32)

    return xyz


def _obstacle_ee_target_cross_pick_nominal(
    t: float,
    T: float,
    start_xyz,
    left_block_xyz,
    cross_jitter_amp: float = 0.010,
    cross_jitter_hz: float = 8.0,
    cross_window_ratio: float = 0.12,
):
    """Obstacle-arm task: pick on obstacle side, brief near-mid feint, then retreat.

    Designed to create one brief avoidance event (helps separate JVP vs FD)
    without long-duration entanglement.
    """
    t = float(np.clip(t, 0.0, T))
    s = t / max(T, 1e-6)

    start = np.array(start_xyz, dtype=np.float32)
    pre = np.array([left_block_xyz[0], left_block_xyz[1], left_block_xyz[2] + 0.12], dtype=np.float32)
    grasp = np.array([left_block_xyz[0], left_block_xyz[1], left_block_xyz[2] + 0.04], dtype=np.float32)
    lift = np.array([left_block_xyz[0], left_block_xyz[1], left_block_xyz[2] + 0.24], dtype=np.float32)
    # Brief "feint" near midline, high-z, then return to its own start position.
    feint = np.array([left_block_xyz[0] - 0.12, -0.05, left_block_xyz[2] + 0.26], dtype=np.float32)
    retreat = start.copy()

    if s <= 0.30:
        w = _smoothstep(s / 0.30)
        xyz = (1.0 - w) * start + w * pre
    elif s <= 0.46:
        w = _smoothstep((s - 0.30) / 0.16)
        xyz = (1.0 - w) * pre + w * grasp
    elif s <= 0.54:
        xyz = grasp.copy()
    elif s <= 0.66:
        w = _smoothstep((s - 0.54) / 0.12)
        xyz = (1.0 - w) * grasp + w * lift
    elif s <= 0.78:
        w = _smoothstep((s - 0.66) / 0.12)
        xyz = (1.0 - w) * lift + w * feint
    else:
        w = _smoothstep((s - 0.78) / 0.22)
        xyz = (1.0 - w) * feint + w * retreat

    # Non-smooth jitter only around the brief feint window.
    hw = 0.5 * float(np.clip(cross_window_ratio, 0.0, 1.0))
    if abs(s - 0.74) <= hw:
        sig = 1.0 if np.sin(2 * np.pi * float(cross_jitter_hz) * t) >= 0 else -1.0
        xyz = xyz + np.array([0.0, sig * float(cross_jitter_amp), 0.0], dtype=np.float32)

    return xyz

def _ik_close_to_q(p_client, robot_id: int, ee_link: int, target_pos, q_ref, target_orn=None):
    """Compute IK biased toward q_ref using pybullet restPoses + joint limits.

    Returns a torch.FloatTensor of size (n_dof,) on success, otherwise None.
    """
    try:
        # Collect revolute joints in the same order as robot.body_joints uses (for Panda this is 7 DOF)
        joints = []
        lower = []
        upper = []
        for j in range(p_client.getNumJoints(robot_id)):
            ji = p_client.getJointInfo(robot_id, j)
            if ji[2] == p_client.JOINT_REVOLUTE:
                joints.append(j)
                lower.append(float(ji[8]))
                upper.append(float(ji[9]))
        if len(joints) == 0:
            return None
        n = len(joints)

        q_ref = torch.as_tensor(q_ref, dtype=torch.float32).reshape(-1)
        if q_ref.numel() < n:
            return None
        q_ref = q_ref[:n].detach().cpu().numpy().astype(np.float32).tolist()

        lower = np.array(lower, dtype=np.float32)
        upper = np.array(upper, dtype=np.float32)
        jr = (upper - lower).astype(np.float32)
        jr = np.where(jr <= 1e-6, 2.0, jr)

        kwargs = dict(
            lowerLimits=lower.tolist(),
            upperLimits=upper.tolist(),
            jointRanges=jr.tolist(),
            restPoses=q_ref,
            maxNumIterations=120,
            residualThreshold=1e-4,
        )

        if target_orn is None:
            sol = p_client.calculateInverseKinematics(robot_id, int(ee_link), list(map(float, target_pos)), **kwargs)
        else:
            sol = p_client.calculateInverseKinematics(robot_id, int(ee_link), list(map(float, target_pos)), targetOrientation=target_orn, **kwargs)

        sol = np.array(sol[:n], dtype=np.float32)
        return torch.tensor(sol, dtype=torch.float32)
    except Exception:
        return None

def _update_obstacle_arm_ik(env: ArmEnv, arm_id: int, ee_target_xyz, ee_link_index: int = None, strength: float = 260.0):
    p_ = env.p
    if ee_link_index is None:
        ee_link_index = _find_ee_link_index(p_, int(arm_id))
    ik = p_.calculateInverseKinematics(
        arm_id,
        ee_link_index,
        targetPosition=[float(ee_target_xyz[0]), float(ee_target_xyz[1]), float(ee_target_xyz[2])],
        maxNumIterations=80,
        residualThreshold=1e-4,
    )

    joints = []
    q_des = []
    for j in range(p_.getNumJoints(arm_id)):
        ji = p_.getJointInfo(arm_id, j)
        if ji[2] == p_.JOINT_REVOLUTE:
            joints.append(j)
            q_des.append(float(ik[len(q_des)]))

    p_.setJointMotorControlArray(
        bodyUniqueId=arm_id,
        jointIndices=joints,
        controlMode=p_.POSITION_CONTROL,
        targetPositions=q_des,
        forces=[float(strength)] * len(joints),
        positionGains=[0.12] * len(joints),
        velocityGains=[0.9] * len(joints),
    )

# ---- Moving obstacle = second robot arm (kinematic obstacle) ----
import math


def _maybe_get_attr(obj, names, default=None):
    for n in names:
        if hasattr(obj, n):
            try:
                v = getattr(obj, n)
                if v is not None:
                    return v
            except Exception:
                pass
    return default


def _find_panda_urdf_path(robot=None):
    """Best-effort: locate a Franka Panda URDF path.

    Tries to reuse paths from the main robot object first, then common repo-relative fallbacks.
    """
    # 1) try to reuse from existing robot wrapper
    cand = _maybe_get_attr(robot, [
        "urdf_path", "urdf", "urdf_file", "urdf_filename", "robot_urdf", "robot_urdf_path"
    ])
    if isinstance(cand, str) and os.path.exists(cand):
        return cand

    # 2) common repo-relative paths
    candidates = [
        "utils/robot/franka_panda/panda.urdf",
        "utils/robot/franka_panda/panda_arm_hand.urdf",
        "utils/robot/franka_panda/urdf/panda.urdf",
        "utils/robot/franka_panda/urdf/panda_arm_hand.urdf",
        "assets/franka_panda/panda.urdf",
        "assets/franka_panda/panda_arm_hand.urdf",
    ]
    for rel in candidates:
        if os.path.exists(rel):
            return rel

    # 3) last resort: try pybullet_data (may not include panda)
    try:
        import pybullet_data
        base = pybullet_data.getDataPath()
        for rel in ["franka_panda/panda.urdf", "panda/panda.urdf"]:
            path = os.path.join(base, rel)
            if os.path.exists(path):
                return path
    except Exception:
        pass

    raise FileNotFoundError(
        "Could not locate Panda URDF. Please set args.obstacle_arm_urdf to a valid URDF path."
    )


def _spawn_obstacle_arm(
    env: ArmEnv,
    main_robot,
    base_xyz=(0.35, 0.25, 0.0),
    base_rpy=(0.0, 0.0, 0.0),
    seed: int = 0,
    urdf_path: str = None,
    use_fixed_base: bool = True,
    amp_scale: float = 1.0,
    omega_scale: float = 1.0,
):
    """Spawn a second Panda arm to act as a moving obstacle.

    - No task / no CBF for this arm.
    - We drive it with simple sinusoidal joint trajectories.
    - The arm is loaded as a separate body in pybullet.
    """
    p_ = env.p

    if urdf_path is None:
        # allow user override via args
        urdf_path = getattr(env, "obstacle_arm_urdf", None)
    if urdf_path is None:
        urdf_path = _find_panda_urdf_path(main_robot)

    # base orientation
    orn = p_.getQuaternionFromEuler(list(base_rpy))

    # load
    arm_id = p_.loadURDF(
        urdf_path,
        basePosition=list(base_xyz),
        baseOrientation=orn,
        useFixedBase=bool(use_fixed_base),
        flags=p_.URDF_USE_INERTIA_FROM_FILE,
    )

    # choose controllable joints (revolute)
    joint_indices = []
    lower = []
    upper = []
    for j in range(p_.getNumJoints(arm_id)):
        ji = p_.getJointInfo(arm_id, j)
        jtype = ji[2]
        if jtype == p_.JOINT_REVOLUTE:
            joint_indices.append(j)
            lower.append(float(ji[8]))
            upper.append(float(ji[9]))

    joint_indices = list(joint_indices)
    lower = np.array(lower, dtype=np.float32)
    upper = np.array(upper, dtype=np.float32)

    # initialize around a "neutral" pose (try to reuse q0 from main robot if sizes match)
    q0_main = None
    try:
        q0_main = np.array(getattr(main_robot, "q0"), dtype=np.float32)
    except Exception:
        q0_main = None

    if q0_main is not None and q0_main.shape[0] >= len(joint_indices):
        q_center = q0_main[: len(joint_indices)].copy()
    else:
        q_center = 0.5 * (lower + upper)

    # randomize center a bit so it is not exactly the same every run
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(low=-0.15, high=0.15, size=q_center.shape).astype(np.float32)
    q_center = np.clip(q_center + jitter, lower + 0.05, upper - 0.05)

    # trajectory parameters: amplitude + frequency per joint
    amp = rng.uniform(low=0.10, high=0.45, size=q_center.shape).astype(np.float32)
    # keep within joint limits
    amp = np.minimum(amp, np.minimum(q_center - (lower + 0.02), (upper - 0.02) - q_center))
    # enlarge/shrink the disturbance range (still respecting joint limits)
    amp = amp * float(amp_scale)
    amp = np.minimum(amp, np.minimum(q_center - (lower + 0.02), (upper - 0.02) - q_center))
    amp = np.clip(amp, 0.03, 0.80)

    omega = rng.uniform(low=0.6, high=2.2, size=q_center.shape).astype(np.float32)
    omega = omega * float(omega_scale)
    phase = rng.uniform(low=0.0, high=2 * np.pi, size=q_center.shape).astype(np.float32)

    # apply initial pose
    for idx, j in enumerate(joint_indices):
        p_.resetJointState(arm_id, j, targetValue=float(q_center[idx]))

    # IMPORTANT: make sure lidar collision checks treat it as an obstacle
    if hasattr(env, "obstacle_ids") and isinstance(env.obstacle_ids, (list, tuple)):
        if arm_id not in env.obstacle_ids:
            try:
                env.obstacle_ids.append(arm_id)
            except Exception:
                pass

    return {
        "arm_id": int(arm_id),
        "joint_indices": joint_indices,
        "q_center": q_center,
        "amp": amp,
        "omega": omega,
        "phase": phase,
        "base_xyz": np.array(base_xyz, dtype=np.float32),
    }


def _update_obstacle_arm(env: ArmEnv, arm_spec: dict, t: float, strength: float = 200.0):
    """Advance obstacle arm motion at time t (seconds)."""
    p_ = env.p
    arm_id = arm_spec["arm_id"]
    joints = arm_spec["joint_indices"]
    q_center = arm_spec["q_center"]
    amp = arm_spec["amp"]
    omega = arm_spec["omega"]
    phase = arm_spec["phase"]

    q_des = q_center + amp * np.sin(omega * float(t) + phase)

    # Drive kinematically via POSITION_CONTROL
    p_.setJointMotorControlArray(
        bodyUniqueId=arm_id,
        jointIndices=joints,
        controlMode=p_.POSITION_CONTROL,
        targetPositions=[float(v) for v in q_des.tolist()],
        forces=[float(strength)] * len(joints),
        positionGains=[0.12] * len(joints),
        velocityGains=[0.9] * len(joints),
    )


def _get_arm_ee_pos(body_id: int, ee_link_index: int = None, p_client=None) -> np.ndarray:
    """Return end-effector position for a pybullet body.

    IMPORTANT: Use the same pybullet client as the environment (`env.p`) when available.
    """
    pc = p_client if p_client is not None else p
    try:
        if ee_link_index is None:
            ee_link_index = _find_ee_link_index(pc, int(body_id))
        ls = pc.getLinkState(body_id, ee_link_index)
        return np.array(ls[4], dtype=np.float32)
    except Exception:
        return np.zeros((3,), dtype=np.float32)


# ---- Ensure non-colliding rollout start ----
def _ensure_noncolliding_start(controller: NeuralLidarCBFController,
                               x: torch.Tensor,
                               min_clearance: float = 0.01,
                               max_tries: int = 30,
                               abort_on_fail: bool = False,
                               exclude_obstacle_ids=None):
    """Ensure the rollout starts collision-free.

    If the provided `x` is already in collision with any obstacle, resample a safe
    configuration (using dm.sample_safe if available) until the robot is at least
    `min_clearance` away from obstacles.

    Returns a (1, D) datax tensor.
    """
    dm = controller.dynamics_model
    env = dm.env
    robot = dm.robot

    # Make sure the observation/aux in x matches the current env
    def _refresh_datax_from_q(q: torch.Tensor) -> torch.Tensor:
        q = q.reshape(1, -1).to(dtype=torch.float32)
        return dm.complete_sample_with_observations(q, num_samples=1)

    # First check the provided state
    q0 = x[0, :dm.n_dims].detach().clone()
    x0 = _refresh_datax_from_q(q0)
    robot.set_joint_position(robot.body_joints, x0[0, :dm.n_dims])
    env.p.stepSimulation()
    obstacle_ids = _get_eval_obstacle_ids(env, robot.robotId, exclude_ids=exclude_obstacle_ids)
    min_d, hit = _min_distance_and_collision(env, robot.robotId, obstacle_ids, distance=2.0)
    if (not hit) and (min_d >= min_clearance):
        return x0

    print(f"[ROLL] start state in collision/min_d={min_d:.4f}. Resampling up to {max_tries} tries...")

    for _ in range(max_tries):
        # Prefer the repo's safe sampler if present
        x_try = None
        try:
            x_try = dm.sample_safe(1)
        except Exception:
            x_try = None

        if x_try is None:
            # Fallback: random in joint limits
            ul, ll = dm.state_limits
            q = torch.lerp(ll, ul, torch.rand_like(ll)).reshape(1, -1).float()
            x_try = _refresh_datax_from_q(q)
        else:
            # dm.sample_safe may already return datax, but refresh to match env
            q = x_try[0, :dm.n_dims].detach().clone()
            x_try = _refresh_datax_from_q(q)

        robot.set_joint_position(robot.body_joints, x_try[0, :dm.n_dims])
        env.p.stepSimulation()
        min_d, hit = _min_distance_and_collision(env, robot.robotId, obstacle_ids, distance=2.0)
        if (not hit) and (min_d >= min_clearance):
            print(f"[ROLL] found collision-free start. min_d={min_d:.4f}")
            return x_try

    if abort_on_fail:
        print("[ROLL] WARNING: could not find a collision-free start state. Aborting rollout.")
        return None

    # Backward-compatible fallback.
    print("[ROLL] WARNING: could not find a collision-free start state. Proceeding anyway.")
    return x0



@torch.no_grad()
def run_moving_obstacle_rollout(
	controller: NeuralLidarCBFController,
	t_sim: float = 20.0,
	move_obstacles: bool = True,
	seed: int = 0,
	realtime: bool = False,
	realtime_scale: float = 1.0,
	speed_scale: float = 1.0,
	obstacle_speed_scale: float = None,
	obstacle_arm_speed_scale: float = None,
	stop_on_goal: bool = True,
	goal_tol: float = 0.10,
	print_every: int = 60,
	amp_range=(0.03, 0.12),
	omega_range=(0.3, 1.2),
	obstacle_mode: str = "arm",
	obstacle_arm_base_xyz=(0.35, 0.25, 0.0),
	obstacle_arm_base_rpy=(0.0, 0.0, 0.0),
	obstacle_arm_strength: float = 200.0,
	obstacle_arm_seed: int = 0,
	obstacle_arm_urdf: str = None,
    pause_on_goal: bool = True,
    goal_pause_tol: float = 1e-4,
	obstacle_arm_amp_scale: float = 1.4,
	obstacle_arm_omega_scale: float = 1.0,
	pause_on_collision: bool = True,
	require_clean_start: bool = True,
	max_start_resample: int = 30,
	start_min_clearance: float = 0.01,
	start_q_override: np.ndarray = None,
	use_motor_control: bool = True,
	max_dq_per_step: float = 0.03,
	pause_on_floor_penetration: bool = True,
	floor_z_tol: float = -0.005,
    scene: str = "plain",
    block_x: float = 0.50,
    block_y_off: float = 0.10,
    block_z: float = 0.03,
    main_base_y: float = -0.20,
    obst_base_y: float = +0.20,
    cross_jitter_amp: float = 0.018,
    cross_jitter_hz: float = 6.0,
    cross_window_ratio: float = 0.35,
    pure_cbf_eval: bool = False,
    obst_freeze_on_close: bool = False,
    continue_after_collision: bool = False,
):
	"""Run a single closed-loop rollout. If move_obstacles=True, obstacles move sinusoidally or as a second arm.

	Prints collision status and returns a dict with trajectory statistics.
	"""
	controller.eval()
	dm = controller.dynamics_model
	env = dm.env
	robot = dm.robot
	p_ = env.p
	main_ee_link_idx = _find_ee_link_index(p_, int(robot.robotId))
	try:
		ji = p_.getJointInfo(int(robot.robotId), int(main_ee_link_idx))
		jn = ji[1].decode("utf-8", "ignore") if isinstance(ji[1], (bytes, bytearray)) else str(ji[1])
		ln = ji[12].decode("utf-8", "ignore") if isinstance(ji[12], (bytes, bytearray)) else str(ji[12])
		print(f"[EE] main_ee_link_index={int(main_ee_link_idx)} joint={jn} link={ln}")
	except Exception:
		print(f"[EE] main_ee_link_index={int(main_ee_link_idx)}")

	# --- Sync speeds: by default, keep both arms and obstacle motion on the same scale ---
	if obstacle_speed_scale is None:
		obstacle_speed_scale = float(speed_scale)
	if obstacle_arm_speed_scale is None:
		obstacle_arm_speed_scale = float(speed_scale)

	# Use the same start state used by the rollout experiment if available
	start_x = None
	try:
		start_x = controller.experiment_suite.experiments[0].start_x
	except Exception:
		start_x = None

	# Reset the environment first so obstacles are in a known state
	try:
		env.reset_env(np.array([]), tidy_env=True)
	except Exception:
		# Some env implementations may not support reset; ignore
		pass

	# Decide obstacle behavior
	# obstacle_mode:
	#   - "none": remove all obstacles and skip collision/distance checks
	#   - "rigid": use existing rigid obstacles and optionally move them
	#   - "arm": spawn a second arm as a moving obstacle (and remove rigid boxes)
	mode = (obstacle_mode or "none").lower()

	if mode == "none":
		removed = _remove_all_obstacles(env, robot.robotId)
		obstacle_ids = []
		# Explicitly clear stale obstacle handles so observation/Jacobian code
		# won't read invalid obstacle robot state in no-obstacle rollouts.
		try:
			if hasattr(env, "obstacle_ids"):
				env.obstacle_ids = []
		except Exception:
			pass
		try:
			if hasattr(env, "obstacle_robot"):
				env.obstacle_robot = None
		except Exception:
			pass
		print(f"[ROLL] obstacle_mode=none -> removed {len(removed)} obstacles: {removed}")

	elif mode in ("arm", "arm_task"):
		# Keep ArmEnv's built-in obstacle robot (if present) so explicit/JVP observation pipeline stays consistent
		_keep = []
		try:
			_obst = getattr(env, "obstacle_robot", None)
			_oid = int(getattr(_obst, "robotId", -1)) if _obst is not None else -1
			if _oid >= 0:
				_keep.append(_oid)
		except Exception:
			_keep = []
		removed = _remove_all_obstacles(env, robot.robotId, exclude_ids=_keep)
		obstacle_ids = []
		print(f"[ROLL] obstacle_mode={mode} -> removed {len(removed)} rigid obstacles: {removed}")
		if mode == "arm_task":
			# IMPORTANT: disable ArmEnv's built-in obstacle trajectory playback.
			# Otherwise dm.closed_loop_dynamics() calls env.step_obstacle() every step
			# and overrides our nominal pick task IK commands.
			try:
				env.obstacle_traj = None
				env.obstacle_traj_dt = None
				env.obstacle_qdot = None
				print("[OBST_ARM_TASK] disabled env obstacle trajectory playback")
			except Exception as e:
				print(f"[OBST_ARM_TASK] WARN: failed to disable env obstacle traj: {e}")

	else:
		# mode == "rigid": keep existing rigid obstacles (boxes/meshes)
		obstacle_ids = _get_eval_obstacle_ids(env, robot.robotId)
		# Debug print once per rollout so you can verify what bodies are considered obstacles
		try:
			names = []
			for oid in obstacle_ids:
				bi = p_.getBodyInfo(oid)
				nm = bi[1]
				if isinstance(nm, (bytes, bytearray)):
					nm = nm.decode("utf-8", "ignore")
				names.append(str(nm))
			print(f"[ROLL] obstacle_ids ({len(obstacle_ids)}): {list(zip(obstacle_ids, names))}")
		except Exception:
			print(f"[ROLL] obstacle_ids ({len(obstacle_ids)}): {obstacle_ids}")

	if len(obstacle_ids) == 0:
		print("[ROLL] WARNING: obstacle_ids is empty after filtering; obstacles will not move and collision checks will be skipped.")

	# --- Cross-pick scene: spawn blocks AFTER obstacle cleanup so they won't be removed ---
	scene_block_ids = []
	table_top_z = None
	left_block_id = None
	right_block_id = None
	obst_target_block_xyz = None
	obst_grasp_state = {"grabbed": False}
	main_grasp_state = {"grabbed": False}
	# For cross_pick: track return-to-home after grasp
	main_return_state = {"returning": False, "home_q": None, "enable_return": True}

	def _find_table_top_z(p_):
		# best-effort: find a body whose name contains "table" and return its AABB top z
		try:
			n = p_.getNumBodies()
			for i in range(n):
				bid = p_.getBodyUniqueId(i)
				bi = p_.getBodyInfo(bid)
				nm = bi[1]
				if isinstance(nm, (bytes, bytearray)):
					nm = nm.decode("utf-8", "ignore")
				nm = str(nm).lower()
				if "table" in nm:
					aabb = p_.getAABB(bid, -1)
					return float(aabb[1][2])
		except Exception:
			pass
		return None

	if str(scene).lower() == "cross_pick":
		# move main robot base to left in Y for visual separation
		try:
			bpos, born = p_.getBasePositionAndOrientation(robot.robotId)
			p_.resetBasePositionAndOrientation(
				robot.robotId,
				[float(bpos[0]), float(main_base_y), float(bpos[2])],
				born,
			)
			print(f"[SCENE] main_base_y={float(main_base_y):.3f}")
		except Exception as e:
			print(f"[SCENE] WARN: cannot reset main base y: {e}")

		table_top_z = _find_table_top_z(p_)
		if table_top_z is None:
			# fallback: assume z=0 is support surface
			table_top_z = 0.0

		# Print kept obstacle robot id for debugging
		try:
			_obst = getattr(env, "obstacle_robot", None)
			_oid = int(getattr(_obst, "robotId", -1)) if _obst is not None else -1
			if _oid >= 0:
				print(f"[SCENE] env.obstacle_robot id={_oid} (kept)")
		except Exception:
			pass

		# Replanned asymmetric layout (stable grasp-first behavior):
		# - green block (obstacle arm task) on obstacle side (+y)
		# - blue block (main arm task) on main-arm side (-y)
		left_block = (float(block_x) + 0.04, +0.75 * float(block_y_off), float(table_top_z) + float(block_z))
		right_block = (float(block_x) - 0.04, -0.55 * float(block_y_off), float(table_top_z) + float(block_z))
		lb_id = _spawn_block(env, left_block, rgba=(0.2, 0.6, 0.2, 1.0))
		rb_id = _spawn_block(env, right_block, rgba=(0.2, 0.2, 0.9, 1.0))
		scene_block_ids = [int(lb_id), int(rb_id)]
		print(f"[SCENE] blocks: left(id={lb_id})={left_block}, right(id={rb_id})={right_block}")
		# Track blocks explicitly for visual grasp
		left_block_id = int(lb_id)
		right_block_id = int(rb_id)
		obst_target_block_xyz = np.array(left_block, dtype=np.float32)
		obst_grasp_state = {"grabbed": False}
		main_grasp_state = {
			"grabbed": False,
			"ee_block_dist": float("inf"),
			"approach_lock": False,
			"descent_goal_set": False,
			"descent_mode": False,
		}

		# main arm goal xyz: lower pregrasp so EE can descend to the block.
		goal_xyz = [right_block[0], right_block[1], right_block[2] + 0.025]
		print(f"[GOAL][cross_pick] blue_block_grasp_xyz={goal_xyz} (will solve IK after start_q)")

	if start_q_override is not None:
		q0 = torch.tensor(start_q_override, dtype=torch.float32).reshape(1, -1)
		x = dm.complete_sample_with_observations(q0, num_samples=1)
	elif start_x is None:
		# Fallback: start near mid of limits
		ul, ll = dm.state_limits
		q0 = torch.lerp(ll, ul, 0.4 * torch.ones(ll.shape[-1]).double()).reshape(1, -1).float()
		x = dm.complete_sample_with_observations(q0, num_samples=1)
	else:
		# Only take q from the saved start state; recompute obs/aux for the current env
		q0 = start_x[0, :dm.n_dims].detach().clone()
		x = dm.complete_sample_with_observations(q0.reshape(1, -1), num_samples=1)

	# If we are using a moving obstacle arm, spawn it BEFORE clean-start checks so
	# collision/clearance is measured against the actual obstacle arm.
	obstacle_arm = None
	if mode in ("arm", "arm_task"):
		# Prefer reusing ArmEnv's built-in obstacle robot (keeps observation pipeline consistent)
		try:
			_env_obst = getattr(env, "obstacle_robot", None)
			oid = int(getattr(_env_obst, "robotId", -1)) if _env_obst is not None else -1
			# Validate the obstacle robot id is still a robot arm (has joints)
			try:
				if oid >= 0 and p_.getNumJoints(oid) <= 0:
					oid = -1
			except Exception:
				oid = -1
			if oid >= 0:
				# Reposition obstacle robot base to the right
				try:
					bpos, born = p_.getBasePositionAndOrientation(oid)
					p_.resetBasePositionAndOrientation(oid, [float(bpos[0]), float(obst_base_y), float(bpos[2])], born)
				except Exception:
					pass

				# Build an obstacle_arm spec that matches our helper expectations
				joints = []
				lower = []
				upper = []
				for j in range(p_.getNumJoints(oid)):
					ji = p_.getJointInfo(oid, j)
					if ji[2] == p_.JOINT_REVOLUTE:
						joints.append(j)
						lower.append(float(ji[8]))
						upper.append(float(ji[9]))
				joints = list(joints)
				lower = np.array(lower, dtype=np.float32)
				upper = np.array(upper, dtype=np.float32)

				# Initialize around mid-limits (deterministic with obstacle_arm_seed)
				rng = np.random.default_rng(int(obstacle_arm_seed))
				q_center = 0.5 * (lower + upper)
				jitter = rng.uniform(low=-0.15, high=0.15, size=q_center.shape).astype(np.float32)
				q_center = np.clip(q_center + jitter, lower + 0.05, upper - 0.05)
				amp = rng.uniform(low=0.10, high=0.45, size=q_center.shape).astype(np.float32)
				amp = np.minimum(amp, np.minimum(q_center - (lower + 0.02), (upper - 0.02) - q_center))
				amp = amp * float(obstacle_arm_amp_scale)
				amp = np.minimum(amp, np.minimum(q_center - (lower + 0.02), (upper - 0.02) - q_center))
				amp = np.clip(amp, 0.03, 0.80)
				omega = rng.uniform(low=0.6, high=2.2, size=q_center.shape).astype(np.float32)
				omega = omega * float(obstacle_arm_omega_scale)
				phase = rng.uniform(low=0.0, high=2 * np.pi, size=q_center.shape).astype(np.float32)

				# Reset joints to q_center
				for idx, j in enumerate(joints):
					p_.resetJointState(oid, int(j), targetValue=float(q_center[idx]))

				obstacle_arm = {
					"arm_id": oid,
					"joint_indices": joints,
					"q_center": q_center,
					"amp": amp,
					"omega": omega,
					"phase": phase,
					"base_xyz": np.array([0.0, float(obst_base_y), 0.0], dtype=np.float32),
				}

				# Ensure evaluation collision checks include the obstacle arm
				if oid not in obstacle_ids:
					obstacle_ids = [oid] + list(obstacle_ids)

				# Do not treat blocks as obstacles for collision/distance
				if scene_block_ids:
					obstacle_ids = [x for x in obstacle_ids if int(x) not in set(scene_block_ids)]

				# Keep observation focused on the obstacle arm
				try:
					env.obstacle_ids = [int(oid)]
				except Exception:
					pass

				# Record ee0
				try:
					ee_link_idx = _find_ee_link_index(p_, int(oid))
					obstacle_arm["ee_link_index"] = int(ee_link_idx)
					ee0 = _get_arm_ee_pos(oid, ee_link_index=ee_link_idx, p_client=p_)
					obstacle_arm["ee0"] = ee0.copy()
					print(f"[OBST_ARM] (env.obstacle_robot) ee0={ee0.tolist()}")
				except Exception:
					pass

				print(f"[OBST_ARM] using env.obstacle_robot id={oid} y={float(obst_base_y):.3f}")
				# If we successfully reused env.obstacle_robot, skip spawning a new URDF arm
				raise StopIteration
		except StopIteration:
			pass
		except Exception:
			# Fall back to spawning a separate URDF arm below
			pass
		if obstacle_arm is None:
			try:
				if obstacle_arm_urdf is not None:
					setattr(env, "obstacle_arm_urdf", obstacle_arm_urdf)
				_b = list(obstacle_arm_base_xyz)
				_b[1] = float(obst_base_y)
				obstacle_arm = _spawn_obstacle_arm(
					env,
					main_robot=robot,
					base_xyz=tuple(_b),
					base_rpy=tuple(obstacle_arm_base_rpy),
					seed=int(obstacle_arm_seed),
					urdf_path=obstacle_arm_urdf,
					use_fixed_base=True,
					amp_scale=float(obstacle_arm_amp_scale),
					omega_scale=float(obstacle_arm_omega_scale),
				)
				# Include obstacle arm in collision checks
				if obstacle_arm["arm_id"] not in obstacle_ids:
					obstacle_ids = [obstacle_arm["arm_id"]] + list(obstacle_ids)
				# Don't treat the scene blocks as obstacles for collision/distance checks
				if scene_block_ids:
					obstacle_ids = [oid for oid in obstacle_ids if int(oid) not in set(scene_block_ids)]
				# IMPORTANT: make observation only "see" the obstacle arm by restricting env.obstacle_ids
				try:
					env.obstacle_ids = [int(obstacle_arm["arm_id"])]
				except Exception:
					pass
				try:
					ee_link_idx = _find_ee_link_index(p_, int(obstacle_arm["arm_id"]))
					obstacle_arm["ee_link_index"] = int(ee_link_idx)
					ee0 = _get_arm_ee_pos(obstacle_arm["arm_id"], ee_link_index=ee_link_idx, p_client=p_)
					obstacle_arm["ee0"] = ee0.copy()
					print(f"[OBST_ARM] ee0={ee0.tolist()}")
				except Exception:
					pass
				print(f"[OBST_ARM] spawned (pre-start) id={obstacle_arm['arm_id']} base={obstacle_arm['base_xyz'].tolist()}")
			except Exception as e:
				print(f"[OBST_ARM] ERROR spawning obstacle arm (pre-start): {e}")
				obstacle_arm = None

	# Make sure we don't start in collision (unless start_q is explicitly fixed).
	if start_q_override is not None:
		print("[ROLL] lock_start_q=True -> skip start resampling")
	else:
		x = _ensure_noncolliding_start(
			controller,
			x,
			min_clearance=float(start_min_clearance),
			max_tries=int(max_start_resample),
			abort_on_fail=bool(require_clean_start),
			exclude_obstacle_ids=scene_block_ids,
		)
		if x is None:
			result = {
				"collided": None,
				"skipped": True,
				"skip_reason": "no_clean_start",
				"seed": int(seed),
				"start_q": start_q_override.tolist() if isinstance(start_q_override, np.ndarray) else None,
				"clean_start": False,
				"steps_ran": 0,
				"min_dist_min": None,
				"min_dist_mean": None,
				"qp_infeasible_count": 0,
				"u_jitter_mean": None,
				"diagnostic_samples": 0,
			}
			print("[ROLL] skipped=True reason=no_clean_start")
			try:
				p_.disconnect()
			except Exception:
				try:
					p.disconnect()
				except Exception:
					pass
			return result

	# Ensure robot is in sync with x at t=0
	q = x[0, :dm.n_dims]
	print(f"[ROLL] start_q_used={q.detach().cpu().tolist()}")
	robot.set_joint_position(robot.body_joints, q)
	# In cross-pick scene, explicitly set goal to the blue block grasp pose
	# and prefer IK solutions close to current start_q.
	if str(scene).lower() == "cross_pick":
		try:
			ee_link = int(main_ee_link_idx)

			def _eval_goal_err(q_goal_t: torch.Tensor) -> float:
				q_save_local = q.detach().clone()
				try:
					robot.set_joint_position(robot.body_joints, q_goal_t)
					p_.stepSimulation()
					ee_now = _get_arm_ee_pos(int(robot.robotId), ee_link_index=ee_link, p_client=p_)
					return float(np.linalg.norm(ee_now - np.array(goal_xyz, dtype=np.float32)))
				finally:
					robot.set_joint_position(robot.body_joints, q_save_local)
					p_.stepSimulation()

			cands = []
			q_goal_near = _ik_close_to_q(
				p_,
				int(robot.robotId),
				ee_link,
				goal_xyz,
				q_ref=q.detach().clone().float(),
			)
			if q_goal_near is not None:
				cands.append(("near_start", q_goal_near))

			try:
				ik = p_.calculateInverseKinematics(
					robot.robotId,
					ee_link,
					goal_xyz,
					maxNumIterations=200,
					residualThreshold=1e-5,
				)
				cands.append(("plain_ik", torch.tensor(ik[:dm.n_dims]).float()))
			except Exception:
				pass

			best_name = None
			best_q = None
			best_err = float("inf")
			for name, q_cand in cands:
				err = _eval_goal_err(q_cand)
				if err < best_err:
					best_err = err
					best_q = q_cand
					best_name = name

			if best_q is not None:
				dm.set_goal(best_q)
				print(
					f"[GOAL][cross_pick] set_goal_to_blue_block=True source={best_name} "
					f"goal_xyz={goal_xyz} ee_err={best_err:.4f}"
				)
			else:
				print("[GOAL][cross_pick] WARN: no valid IK candidate for blue block")
		except Exception as e:
			print(f"[GOAL][cross_pick] WARN: failed to set blue-block goal from start_q: {e}")
	# Save home configuration only when return-home behavior is enabled
	if bool(main_return_state.get("enable_return", False)):
		try:
			main_return_state["home_q"] = q.detach().clone().float()
		except Exception:
			main_return_state["home_q"] = None
	p_.stepSimulation()
	# Visualize and print start/goal (end-effector markers)
	start_ee = _get_arm_ee_pos(int(robot.robotId), ee_link_index=int(main_ee_link_idx), p_client=p_)
	# Put the robot at goal once to get goal EE, then restore
	q_goal = dm.goal_state[:dm.n_dims].detach().clone().float()
	q_save = q.detach().clone()
	robot.set_joint_position(robot.body_joints, q_goal)
	p_.stepSimulation()
	goal_ee = _get_arm_ee_pos(int(robot.robotId), ee_link_index=int(main_ee_link_idx), p_client=p_)
	# restore start
	robot.set_joint_position(robot.body_joints, q_save)
	p_.stepSimulation()
	print(f"[START/GOAL] start_ee={start_ee.tolist()}  goal_ee={goal_ee.tolist()}")
	_spawn_marker(start_ee, rgba=(0, 1, 0, 0.8), radius=0.035)
	# In cross_pick, blue block itself is the visual goal; hide red goal marker.
	if str(scene).lower() != "cross_pick":
		_spawn_marker(goal_ee, rgba=(1, 0, 0, 0.8), radius=0.035)
	# Make sim stepping consistent with the dynamics dt
	try:
		p_.setRealTimeSimulation(0)
		p_.setTimeStep(dm.dt)
	except Exception:
		pass

	# Physics stabilization parameters (best-effort)
	try:
		p_.setPhysicsEngineParameter(numSolverIterations=200)
		p_.setPhysicsEngineParameter(numSubSteps=2)
	except Exception:
		pass

	# --- moving obstacle source ---
	base = direction = omega = amp = None
	q_sidestep = None
	if str(scene).lower() == "cross_pick":
		# Side-step waypoint (joint-space) to make avoidance visibly detour around obstacle arm.
		try:
			away_sign = -1.0 if float(obst_base_y) >= 0.0 else 1.0
			side_goal_xyz = [float(goal_xyz[0]), float(goal_xyz[1]) + away_sign * 0.12, float(goal_xyz[2]) + 0.03]
			q_side = _ik_close_to_q(
				p_,
				int(robot.robotId),
				int(main_ee_link_idx),
				side_goal_xyz,
				q_ref=q.detach().clone().float(),
			)
			if q_side is not None:
				q_sidestep = q_side.detach().clone().float()
				print(f"[DODGE] sidestep_goal_xyz={side_goal_xyz}")
		except Exception:
			q_sidestep = None

	if mode == "arm":
		# Obstacle arm was spawned before the clean-start check above.
		# Nothing to do here.
		pass

	elif mode == "rigid":
		# Move existing rigid-body obstacles (sinusoidal base motion)
		base, direction, omega, amp = _make_obstacle_traj_from_current(
			env,
			obstacle_ids,
			seed=seed,
			amp_range=amp_range,
			omega_range=omega_range,
		)

	else:
		# mode == "none": obstacles already removed above
		pass

	steps = int(t_sim / dm.dt)
	min_dist_hist = []
	collided = False
	collide_step = None
	qp_infeasible_count = 0
	u_prev = None
	u_jitter_hist = []
	diag_hist = []
	diag_bucket = {
		"near": [],
		"far": [],
		"hit_low": [],
		"hit_high": [],
	}
	obst_task_freeze_steps = 0
	obst_task_freeze_hold = max(1, int(0.4 / float(dm.dt)))  # ~0.4s

	for k in range(steps):
		# Base time used for obstacle motion
		t_base = (k * dm.dt) * float(obstacle_speed_scale)
		# Optionally speed up ONLY the obstacle arm (separate from rigid obstacles)
		t_arm = t_base * float(obstacle_arm_speed_scale)
		# Proximity check before moving obstacles (used by arm_task freeze gate)
		pre_min_d_gate = None
		if mode != "none" and len(obstacle_ids) > 0:
			try:
				pre_min_d_gate, _ = _min_distance_and_collision(env, robot.robotId, obstacle_ids, distance=2.0)
			except Exception:
				pre_min_d_gate = None
		if bool(obst_freeze_on_close) and mode == "arm_task" and (pre_min_d_gate is not None) and (pre_min_d_gate < 0.08):
			obst_task_freeze_steps = int(obst_task_freeze_hold)

		# 1) Move the obstacle(s) first (so observation sees the new positions)
		if move_obstacles:
			if mode == "arm" and obstacle_arm is not None:
				_update_obstacle_arm(env, obstacle_arm, t_arm, strength=float(obstacle_arm_strength))
				# optional tiny debug prints
				if k < 3:
					ee = _get_arm_ee_pos(
						obstacle_arm["arm_id"],
						ee_link_index=int(obstacle_arm.get("ee_link_index", _find_ee_link_index(p_, int(obstacle_arm["arm_id"])))),
						p_client=p_,
					)
					print(f"[OBST_ARM] t={t_arm:.3f} ee={ee.tolist()}")
				# Visual-only grasp for obstacle arm in sinusoidal mode as well
				if str(scene).lower() == "cross_pick":
					ee_link = int(obstacle_arm.get("ee_link_index", _find_ee_link_index(p_, int(obstacle_arm["arm_id"]))))
					_update_visual_grasp_block(p_, int(obstacle_arm["arm_id"]), ee_link, left_block_id, obst_grasp_state,
									  dist_thresh=0.05, ee_z_offset=-0.035)
			elif mode == "arm_task" and obstacle_arm is not None:
				if str(scene).lower() == "cross_pick":
					if bool(obst_freeze_on_close) and obst_task_freeze_steps > 0:
						obst_task_freeze_steps -= 1
						if (k % max(int(print_every), 1)) == 0:
							print(f"[OBST_ARM_TASK] freeze_hold active steps_left={obst_task_freeze_steps}")
					else:
						# Match the Z used when spawning blocks: table_top_z + block_z
						_tz = float(table_top_z) if table_top_z is not None else 0.0
						left_block_xyz = (
							obst_target_block_xyz.copy()
							if obst_target_block_xyz is not None
							else np.array([float(block_x), -float(block_y_off), _tz + float(block_z)], dtype=np.float32)
						)
						ee0 = obstacle_arm.get("ee0", _get_arm_ee_pos(obstacle_arm["arm_id"], p_client=p_))
						ee_tgt = _obstacle_ee_target_cross_pick_nominal(
							t=float(k * dm.dt),
							T=float(t_sim),
							start_xyz=ee0,
							left_block_xyz=left_block_xyz,
							cross_jitter_amp=float(cross_jitter_amp),
							cross_jitter_hz=float(cross_jitter_hz),
							cross_window_ratio=float(cross_window_ratio),
						)
						_update_obstacle_arm_ik(env, int(obstacle_arm["arm_id"]), ee_tgt, strength=float(obstacle_arm_strength))
						if k in (0, int(0.35 * steps), int(0.55 * steps), int(0.82 * steps)):
							print(f"[OBST_ARM_TASK] nominal_pick ee_tgt={ee_tgt.tolist()}")
					# Visual-only grasp: obstacle arm attaches left block when close
					ee_link = int(obstacle_arm.get("ee_link_index", _find_ee_link_index(p_, int(obstacle_arm["arm_id"]))))
					_update_visual_grasp_block(p_, int(obstacle_arm["arm_id"]), ee_link, left_block_id, obst_grasp_state,
									  dist_thresh=0.05, ee_z_offset=-0.035)
				else:
					_update_obstacle_arm(env, obstacle_arm, t_arm, strength=float(obstacle_arm_strength))
			elif mode == "rigid":
				_update_obstacles(env, obstacle_ids, t_base, base, direction, omega, amp)
			# else: mode == "none" -> do nothing

		# Pre-control proximity check (for adaptive safety scaling).
		pre_min_d = None
		if mode != "none" and len(obstacle_ids) > 0:
			try:
				pre_min_d, _ = _min_distance_and_collision(env, robot.robotId, obstacle_ids, distance=2.0)
			except Exception:
				pre_min_d = None

		# 2) Compute control using current datax (q + obs + aux)
		u = controller.u(x)[0]
		# Visual detour assist: near moving obstacle, blend toward a side-step waypoint.
		if (
			(not bool(pure_cbf_eval))
			and str(scene).lower() == "cross_pick"
			and (mode != "none")
			and (pre_min_d is not None)
			and (q_sidestep is not None)
			and (not bool(main_return_state.get("returning", False)))
			and (not bool(main_grasp_state.get("grabbed", False)))
		):
			ee_to_blue = float(main_grasp_state.get("ee_block_dist", 1e9))
			approach_lock = bool(main_grasp_state.get("approach_lock", False))
			descent_mode = bool(main_grasp_state.get("descent_mode", False))
			if pre_min_d < 0.35 and (ee_to_blue > 0.35) and (not approach_lock) and (not descent_mode):
				try:
					q_now = x[0, :dm.n_dims]
					u_side = 2.8 * (q_sidestep.to(x.device) - q_now)
					gamma = 0.75 * float(np.clip((0.35 - pre_min_d) / 0.22, 0.0, 1.0))
					u = (1.0 - gamma) * u + gamma * u_side
					if (k % max(int(print_every), 1)) == 0:
						print(f"[DODGE] pre_min_d={pre_min_d:.4f} sidestep_blend={gamma:.3f}")
				except Exception:
					pass
		# Near-goal stabilization for cross_pick:
		# explicit/JVP can stall near GOAL/HOME; progressively blend in a
		# reference/track term to force final convergence.
		if (not bool(pure_cbf_eval)) and str(scene).lower() == "cross_pick":
			try:
				q_now_pre = x[0, :dm.n_dims]
				q_goal_dev = q_goal.to(x.device)
				d_goal_pre = float(torch.norm(q_now_pre - q_goal_dev).item())
			except Exception:
				d_goal_pre = None
			if d_goal_pre is not None:
				try:
					u_ref = controller.u_reference(x)[0]
					is_return = bool(main_return_state.get("returning", False))
					if is_return:
						# During return, keep obstacle avoidance active.
						# If obstacles exist, do NOT blend in nominal/reference terms.
						if mode == "none":
							alpha = float(np.clip((0.90 - d_goal_pre) / 0.70, 0.0, 1.0))
							u = (1.0 - alpha) * u + alpha * u_ref
							if d_goal_pre < 0.80:
								k_track = 3.2
								u_track = k_track * (q_goal_dev - q_now_pre)
								beta = float(np.clip((0.80 - d_goal_pre) / 0.65, 0.0, 1.0))
								u = (1.0 - beta) * u + beta * u_track
						else:
							alpha = 0.0
					else:
						# GOAL phase (grasp): keep pure CBF when obstacles exist.
						if mode == "none":
							alpha = float(np.clip((0.60 - d_goal_pre) / 0.45, 0.0, 1.0))
							u = (1.0 - alpha) * u + alpha * u_ref
							if d_goal_pre < 0.45:
								k_track = 2.4
								u_track = k_track * (q_goal_dev - q_now_pre)
								beta = float(np.clip((0.45 - d_goal_pre) / 0.35, 0.0, 1.0))
								u = (1.0 - beta) * u + beta * u_track
						else:
							# With obstacles: keep CBF dominant, but add stronger approach
							# assistance near the blue block to avoid stalling.
							ee_to_blue = float(main_grasp_state.get("ee_block_dist", 1e9))
							descent_mode = bool(main_grasp_state.get("descent_mode", False))
							if descent_mode:
								# Force a stronger down-reaching motion once descent starts.
								alpha = 0.85 if ((pre_min_d is None) or (pre_min_d > 0.03)) else 0.55
								u = (1.0 - alpha) * u + alpha * u_ref
							elif (pre_min_d is not None) and (pre_min_d > 0.05) and (ee_to_blue < 0.35):
								alpha = float(np.clip((0.35 - ee_to_blue) / 0.25, 0.30, 0.75))
								u = (1.0 - alpha) * u + alpha * u_ref
							else:
								alpha = 0.0
					if (k % max(int(print_every), 1)) == 0:
						phase = "HOME" if bool(main_return_state.get("returning", False)) else "GOAL"
						print(f"[CTRL] near_goal_blend phase={phase} alpha={alpha:.3f} d_goal={d_goal_pre:.3f}")
				except Exception:
					pass
		if torch.isnan(u).any() or torch.isinf(u).any():
			qp_infeasible_count += 1
			u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
		# Make the arm move faster/slower (visual + actual) while keeping it bounded
		u = u * float(speed_scale)
		# If we get too close to moving obstacles, reduce commanded speed to
		# give CBF/QP more room to react (visible avoidance instead of late collision).
		if (not bool(pure_cbf_eval)) and (mode != "none") and (pre_min_d is not None):
			ee_to_blue = float(main_grasp_state.get("ee_block_dist", 1e9))
			approach_lock = bool(main_grasp_state.get("approach_lock", False))
			descent_mode = bool(main_grasp_state.get("descent_mode", False))
			if pre_min_d < 0.35 and (ee_to_blue > 0.35) and (not approach_lock) and (not descent_mode):
				slow_floor = 0.15 if ee_to_blue > 0.18 else 0.35
				slow = float(np.clip((pre_min_d - 0.05) / 0.30, slow_floor, 1.0))
				u = u * slow
				if (k % max(int(print_every), 1)) == 0:
					print(f"[SAFE] pre_min_d={pre_min_d:.4f} speed_scale_near_obs={slow:.3f}")

		# Enforce a per-step max joint increment to avoid tunneling
		dq_max = float(max_dq_per_step)
		if dq_max > 0:
			u = torch.clamp(u, -dq_max / float(dm.dt), dq_max / float(dm.dt))

		# Conservative default clamp if the dynamics doesn't expose limits
		try:
			u_hi, u_lo = getattr(dm, "control_limits")
			u = torch.max(torch.min(u, u_hi), u_lo)
		except Exception:
			u = torch.clamp(u, -2.5, 2.5)

		if u_prev is not None:
			u_jitter_hist.append(float(torch.norm(u - u_prev).item()))
		u_prev = u.detach().clone()

		diag = None
		if hasattr(controller, "derivative_diagnostics"):
			try:
				diag = controller.derivative_diagnostics(x, u)
			except Exception:
				diag = None
		if diag is not None:
			diag_hist.append(diag)
			# Bucketing by near/far and predicted hit count
			hc = int(diag.get("hit_count_pred", 0))
			if hc >= 128:
				diag_bucket["hit_high"].append(diag)
			else:
				diag_bucket["hit_low"].append(diag)

		# 3) Step dynamics with observation update
		x = dm.closed_loop_dynamics(x, u, collect_dataset=False, use_motor_control=bool(use_motor_control), update_observation=True)

		# 4) Advance physics (if dm.closed_loop_dynamics didn't already step physics)
		p_.stepSimulation()
		# Visual-only grasp for MAIN arm: attach the RIGHT block when close
		if str(scene).lower() == "cross_pick":
			main_ee_link = int(main_ee_link_idx)
			_update_visual_grasp_block(
				p_,
				int(robot.robotId),
				main_ee_link,
				right_block_id,
				main_grasp_state,
				dist_thresh=0.12,
				ee_z_offset=-0.035,
				z_align_thresh=0.05,
				xy_align_thresh=0.06,
				grasp_z_max_offset=0.04,
			)
			try:
				if float(main_grasp_state.get("ee_block_dist", 1e9)) < 0.35:
					main_grasp_state["approach_lock"] = True
			except Exception:
				pass
			# One-shot descent stage: once near the blue block, lower the goal to an explicit
			# down-reaching pose so the gripper actually moves down instead of hovering.
			try:
				ee_to_blue_now = float(main_grasp_state.get("ee_block_dist", 1e9))
				if (not bool(main_grasp_state.get("grabbed", False))) and (not bool(main_grasp_state.get("descent_goal_set", False))) and (ee_to_blue_now < 0.40):
					descend_xyz = [float(right_block[0]), float(right_block[1]), float(right_block[2]) + 0.015]
					q_goal_down = _ik_close_to_q(
						p_,
						int(robot.robotId),
						int(main_ee_link_idx),
						descend_xyz,
						q_ref=x[0, :dm.n_dims].detach().clone().float(),
					)
					if q_goal_down is not None:
						dm.set_goal(q_goal_down)
						q_goal = dm.goal_state[:dm.n_dims].detach().clone().float()
						main_grasp_state["descent_goal_set"] = True
						main_grasp_state["descent_mode"] = True
						print(f"[TASK] set_descent_goal_xyz={descend_xyz}")
			except Exception:
				pass
			if (k % max(int(print_every), 1)) == 0:
				try:
					print(f"[TASK] main_ee_to_blue_block={float(main_grasp_state.get('ee_block_dist', float('nan'))):.4f} grabbed={bool(main_grasp_state.get('grabbed', False))}")
				except Exception:
					pass
			# Hard trigger: if EE-to-block distance enters threshold, mark grasp.
			try:
				if (
					(not bool(main_grasp_state.get("grabbed", False)))
					and float(main_grasp_state.get("ee_block_dist", 1e9)) <= 0.12
					and float(main_grasp_state.get("ee_xy_dist", 1e9)) <= 0.06
					and float(main_grasp_state.get("ee_dz", 1e9)) <= 0.05
					and float(main_grasp_state.get("ee_z", 1e9)) <= float(main_grasp_state.get("block_z", -1e9)) + 0.04
				):
					main_grasp_state["grabbed"] = True
			except Exception:
				pass
			# Optional behavior: after grasp, switch goal to return home
			if str(scene).lower() == "cross_pick" and bool(main_return_state.get("enable_return", False)) and (not main_return_state.get("returning", False)):
				if bool(main_grasp_state.get("grabbed", False)):
					hq = main_return_state.get("home_q", None)
					if hq is not None:
						try:
							dm.set_goal(hq)
							q_goal = dm.goal_state[:dm.n_dims].detach().clone().float()
							main_return_state["returning"] = True
							print("[TASK] main grasped right block -> switching goal to HOME")
						except Exception as e:
							print(f"[TASK] WARN: failed to switch goal to HOME: {e}")

		# Detect obvious floor penetration (debug)
		if pause_on_floor_penetration:
			try:
				min_z = float("inf")
				for j in robot.body_joints:
					ls = p_.getLinkState(robot.robotId, int(j))
					z = float(ls[4][2])
					if z < min_z:
						min_z = z
				# If cross-pick scene, forbid going below the tabletop ("under the table")
				_limit_z = float(floor_z_tol)
				if str(scene).lower() == "cross_pick" and table_top_z is not None:
					_limit_z = float(table_top_z) - 0.005

				if min_z <= _limit_z:
					print(f"[FLOOR] WARNING: link below floor: min_link_z={min_z:.6f} <= {_limit_z:.6f} at step {k}")
					if pause_on_collision:
						try:
							p_.setRealTimeSimulation(0)
						except Exception:
							pass
						while True:
							time.sleep(0.1)
			except Exception:
				pass

		# Goal progress (in joint space)
		q_now = x[0, :dm.n_dims]
		d_goal = torch.norm(q_now - q_goal.to(q_now.device)).item()
		if (k % max(int(print_every), 1)) == 0:
			if mode != "none" and len(min_dist_hist) > 0:
				md = min_dist_hist[-1]
			else:
				md = float("nan")
			print(f"[ROLL] step={k:5d}/{steps}  t={k*dm.dt:6.3f}s  ||q-goal||={d_goal:.3f}  min_d={md:.4f}")
			if str(scene).lower() == "cross_pick" and bool(main_return_state.get("returning", False)):
				print(f"[TASK] return_q_dist={d_goal:.4f}")
		# Pause when the robot is extremely close to goal (default tol=1e-4)
		if d_goal <= float(goal_pause_tol):
			print(f"[ROLL] GOAL reached (pause): ||q-goal||={d_goal:.6f} <= {float(goal_pause_tol):.6f} at step {k}")
			if pause_on_goal:
				try:
					p_.setRealTimeSimulation(0)
				except Exception:
					pass
				while True:
					time.sleep(0.1)
			# If you prefer to stop instead of pause, keep stop_on_goal=True
		if stop_on_goal and (d_goal <= float(goal_tol)):
			# In cross_pick, do not stop at pre-grasp goal before actual grasp trigger.
			if str(scene).lower() == "cross_pick" and (not bool(main_return_state.get("returning", False))) and (not bool(main_grasp_state.get("grabbed", False))):
				pass
			else:
				phase = "HOME" if bool(main_return_state.get("returning", False)) else "GOAL"
				print(f"[ROLL] reached {phase}: ||q-goal||={d_goal:.6f} <= {float(goal_tol):.6f} at step {k}")
				break

		# 5) Measure collision / distance (skip if obstacle_mode==none)
		if mode != "none" and len(obstacle_ids) > 0:
			min_d, hit = _min_distance_and_collision(env, robot.robotId, obstacle_ids, distance=2.0)
			min_dist_hist.append(min_d)
			if diag is not None:
				(diag_bucket["near"] if min_d < 0.2 else diag_bucket["far"]).append(diag)
			if hit:
				collided = True
				if collide_step is None:
					collide_step = k
				print(f"[ROLL] COLLISION detected at step {k}, sim_time={k*dm.dt:.3f}s, min_d={min_d:.6f}")
				if pause_on_collision:
					# Keep the GUI open and pause here. Press Ctrl+C in the terminal to exit.
					try:
						p_.setRealTimeSimulation(0)
					except Exception:
						pass
					while True:
						time.sleep(0.1)
				if not bool(continue_after_collision):
					break

		if realtime:
			# realtime_scale > 1 slows down the visualization (e.g., 2.0 means 2x slower than real time)
			sleep_dt = max(dm.dt * float(realtime_scale), 1.0 / 60.0)
			time.sleep(sleep_dt)

	result = {
		"move_obstacles": move_obstacles,
		"seed": seed,
		"t_sim": t_sim,
		"start_q": q.detach().cpu().tolist(),
		"clean_start": True,
		"steps_ran": ((collide_step + 1) if (collided and (not bool(continue_after_collision)) and (collide_step is not None)) else (k + 1)),
		"collided": collided,
		"min_dist_min": float(np.min(min_dist_hist)) if len(min_dist_hist) else None,
		"min_dist_mean": float(np.mean(min_dist_hist)) if len(min_dist_hist) else None,
		"qp_infeasible_count": int(qp_infeasible_count),
		"u_jitter_mean": float(np.mean(u_jitter_hist)) if len(u_jitter_hist) else None,
		"diagnostic_samples": int(len(diag_hist)),
		"odot_err_p": None,
		"odot_err_n": None,
		"odot_err_m": None,
		"odot_jvp_p_meanabs": None,
		"odot_jvp_n_meanabs": None,
		"odot_jvp_m_meanabs": None,
		"odot_fd_p_meanabs": None,
		"odot_fd_n_meanabs": None,
		"odot_fd_m_meanabs": None,
		"hdot_auto_mean": None,
		"hdot_fd_mean": None,
		"hdot_err": None,
	}
	if len(diag_hist) > 0:
		for k in [
			"odot_err_p", "odot_err_n", "odot_err_m",
			"odot_jvp_p_meanabs", "odot_jvp_n_meanabs", "odot_jvp_m_meanabs",
			"odot_fd_p_meanabs", "odot_fd_n_meanabs", "odot_fd_m_meanabs",
			"hdot_auto_mean", "hdot_fd_mean", "hdot_err"
		]:
			vals = [d[k] for d in diag_hist if k in d]
			if len(vals) > 0:
				result[k] = float(np.mean(vals))
		for bk, blist in diag_bucket.items():
			if len(blist) == 0:
				continue
			for k in [
				"hdot_auto_mean", "hdot_fd_mean", "hdot_err",
				"odot_err_p", "odot_err_n", "odot_err_m",
				"odot_jvp_p_meanabs", "odot_jvp_n_meanabs", "odot_jvp_m_meanabs",
				"odot_fd_p_meanabs", "odot_fd_n_meanabs", "odot_fd_m_meanabs",
			]:
				vals = [d[k] for d in blist if k in d]
				if len(vals) > 0:
					result[f"{k}_{bk}"] = float(np.mean(vals))
	print("[ROLL] move_obstacles=", move_obstacles,
			" seed=", seed,
			" collided=", collided,
			" steps=", result["steps_ran"],
			" min_dist_min=", result["min_dist_min"],
			" qp_infeasible=", result.get("qp_infeasible_count"),
			" u_jitter_mean=", result.get("u_jitter_mean"))
	# Best-effort cleanup to avoid BulletClient __del__ warnings on interpreter shutdown
	try:
		p_.disconnect()
	except Exception:
		try:
			p.disconnect()
		except Exception:
			pass
	return result



# ---- Offline metrics evaluation helper ----
@torch.no_grad()
def eval_metrics_offline(
    controller: NeuralLidarCBFController,
    num_samples: int = 2048,
    batch_size: int = 256,
    seed: int = 0,
    alpha: float = None,
    u_clamp: float = 2.5,
    near_ratio: float = 0.0,
    fd_eps_list: str = "1e-2,5e-3,1e-3",
    near_mode: str = "boundary_or_unsafe",
    fd_obs_source: str = "model",
):
    """Offline (no rollout) evaluation for A/B.

    This evaluates:
      - h statistics
      - QP infeasible / NaN rate (from controller.u)
      - finite-difference hdot using one-step lookahead: (h(x_next)-h(x))/dt
      - relaxation term: relu(hdot + alpha*h)
      - split terms: relu(hdot), relu(alpha*h)
      - near/far bucket stats (near := boundary | unsafe)

    It avoids datamodule dependencies and does NOT run long rollouts.
    """
    controller.eval()
    dm = controller.dynamics_model

    rng = np.random.default_rng(int(seed))
    # sample q uniformly in joint limits
    ul, ll = dm.state_limits  # note: in this repo ul/ll ordering may be (high, low)
    # ensure shapes are torch tensors
    ul_t = ul.detach().clone().float().reshape(1, -1)
    ll_t = ll.detach().clone().float().reshape(1, -1)

    # choose alpha
    if alpha is None:
        alpha = float(getattr(controller, "cbf_alpha", getattr(controller, "clf_lambda", 1.0)))
    dt_eval = float(getattr(dm, "controller_dt", dm.dt))
    near_ratio = float(max(0.0, min(1.0, near_ratio)))

    n = int(num_samples)
    bs = int(batch_size)

    h_all = []
    hdot_all = []
    relax_all = []
    relax_auto_all = []
    relax_fd_all = []
    relu_hdot_all = []
    relu_ah_all = []
    near_mask_all = []
    hdot_auto_all = []
    fd_eps_values = []
    for token in str(fd_eps_list).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            val = float(token)
            if val > 0:
                fd_eps_values.append(val)
        except Exception:
            continue
    fd_eps_values = sorted(set(fd_eps_values), reverse=True)
    hdot_fd_scan = {eps: [] for eps in fd_eps_values}
    hdot_auto_err_scan = {eps: [] for eps in fd_eps_values}
    diag_all = {
        "odot_err_p": [],
        "odot_err_n": [],
        "odot_err_m": [],
        "hdot_err": [],
    }
    infeasible = 0
    total = 0

    ratio_warned = False

    def _random_state(batch_n: int) -> torch.Tensor:
        u01 = torch.from_numpy(rng.random((batch_n, ul_t.shape[1])).astype(np.float32))
        q = ll_t + (ul_t - ll_t) * u01
        return dm.complete_sample_with_observations(q, num_samples=batch_n)

    def _near_mask(x_in: torch.Tensor) -> torch.Tensor:
        mode = str(near_mode).lower()
        if mode == "unsafe_only":
            return dm.unsafe_mask(x_in)
        if mode == "boundary_only":
            return dm.boundary_mask(x_in)
        return dm.boundary_mask(x_in) | dm.unsafe_mask(x_in)

    def _h_eval(x_in: torch.Tensor) -> torch.Tensor:
        src = str(fd_obs_source).lower()
        if src != "raw":
            return controller.h(x_in).reshape(-1)
        if not hasattr(controller, "use_gphi_chain"):
            return controller.h(x_in).reshape(-1)
        prev = bool(getattr(controller, "use_gphi_chain", False))
        try:
            # Evaluate h on raw observation (no gphi replacement).
            controller.use_gphi_chain = False
            return controller.h(x_in).reshape(-1)
        finally:
            controller.use_gphi_chain = prev

    def _collect_x(target: int, want_near: bool) -> torch.Tensor:
        if target <= 0:
            return torch.zeros((0, dm.n_dims + dm.o_dims_in_dataset + dm.state_aux_dims_in_dataset), dtype=torch.float32)
        chunks = []
        collected = 0
        tries = 0
        max_tries = max(200, target * 60)
        while collected < target and tries < max_tries:
            tries += 1
            cbs = min(64, target - collected)
            # Use task-aware samplers first to avoid near/far collapse.
            x_try = None
            try:
                if want_near:
                    if tries % 2 == 0:
                        x_try = dm.sample_unsafe(cbs, max_tries=30)
                    else:
                        x_try = dm.sample_boundary(cbs, max_tries=30)
                else:
                    x_try = dm.sample_safe(cbs, max_tries=40)
            except RuntimeWarning:
                # Some samplers raise RuntimeWarning as exception when they can't
                # collect enough points; fallback to random rejection sampling.
                x_try = None
            except Exception:
                x_try = None
            if x_try is None:
                x_try = _random_state(cbs)
            near_try = _near_mask(x_try)
            keep_mask = near_try if want_near else (~near_try)
            if keep_mask.any():
                keep = x_try[keep_mask]
                take = min(target - collected, keep.shape[0])
                chunks.append(keep[:take])
                collected += take

        # Last-resort random rejection sampling.
        while collected < target and tries < (max_tries + 300):
            tries += 1
            cbs = min(128, target - collected)
            x_try = _random_state(cbs)
            near_try = _near_mask(x_try)
            keep_mask = near_try if want_near else (~near_try)
            if keep_mask.any():
                keep = x_try[keep_mask]
                take = min(target - collected, keep.shape[0])
                chunks.append(keep[:take])
                collected += take

        if len(chunks) == 0:
            return torch.zeros((0, dm.n_dims + dm.o_dims_in_dataset + dm.state_aux_dims_in_dataset), dtype=torch.float32)

        x_cat = torch.cat(chunks, dim=0)
        if x_cat.shape[0] < target:
            pad_idx = torch.randint(low=0, high=x_cat.shape[0], size=(target - x_cat.shape[0],))
            x_cat = torch.cat([x_cat, x_cat[pad_idx]], dim=0)
        return x_cat[:target]

    for start in range(0, n, bs):
        cur_bs = min(bs, n - start)
        near_target = int(round(cur_bs * near_ratio))
        far_target = cur_bs - near_target

        x_near = _collect_x(near_target, True)
        x_far = _collect_x(far_target, False)

        # If one bucket is empty, fallback to random states to keep eval running.
        if x_near.shape[0] == 0 and near_target > 0:
            x_near = _random_state(near_target)
        if x_far.shape[0] == 0 and far_target > 0:
            x_far = _random_state(far_target)

        x = torch.cat([x_near, x_far], dim=0)
        if x.shape[0] != cur_bs:
            # final guard: trim/pad with random states
            if x.shape[0] > cur_bs:
                x = x[:cur_bs]
            else:
                x = torch.cat([x, _random_state(cur_bs - x.shape[0])], dim=0)

        # shuffle batch so controller doesn't see ordered near/far chunks
        perm = torch.randperm(x.shape[0])
        x = x[perm]
        near_mask = _near_mask(x)
        if (near_ratio > 0.0 and near_ratio < 1.0) and (not ratio_warned):
            realized = float(near_mask.float().mean().item())
            if abs(realized - near_ratio) > 0.15:
                print(
                    f"[metrics][WARN] near_ratio target={near_ratio:.3f}, realized_batch={realized:.3f}. "
                    f"Try smaller batch_size or adjust near definition."
                )
                ratio_warned = True

        # compute control
        u = controller.u(x)[0]
        bad = torch.isnan(u).any(dim=1) | torch.isinf(u).any(dim=1)
        infeasible += int(bad.sum().item())
        u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        u = torch.clamp(u, -float(u_clamp), float(u_clamp))

        # h and one-step FD hdot
        h = _h_eval(x)
        x_next = dm.batch_lookahead(x, u * dt_eval, data_jacobian=())
        h_next = _h_eval(x_next)
        hdot = (h_next - h) / dt_eval
        hdot_auto = None
        if hasattr(controller, "_compute_hdot_auto"):
            try:
                hdot_auto = controller._compute_hdot_auto(x, u)
            except Exception:
                hdot_auto = None
        if hdot_auto is not None:
            hdot_auto = hdot_auto.reshape(-1)
            hdot_auto_all.append(hdot_auto.detach().cpu())

        for eps in fd_eps_values:
            x_eps = dm.batch_lookahead(x, u * float(eps), data_jacobian=())
            h_eps = controller.h(x_eps).reshape(-1)
            hdot_fd_eps = (h_eps - h) / float(eps)
            hdot_fd_scan[eps].append(hdot_fd_eps.detach().cpu())
            if hdot_auto is not None:
                hdot_auto_err_scan[eps].append((hdot_auto - hdot_fd_eps).abs().detach().cpu())

        relu_hdot = F.relu(hdot)
        relu_ah = F.relu(float(alpha) * h)
        relax = F.relu(hdot + float(alpha) * h)
        if hdot_auto is not None:
            relax_auto = F.relu(hdot_auto + float(alpha) * h)
            relax_auto_all.append(relax_auto.detach().cpu())
            relax_fd_all.append(relax.detach().cpu())

        if hasattr(controller, "derivative_diagnostics"):
            try:
                d = controller.derivative_diagnostics(x, u)
            except Exception:
                d = None
            if d is not None:
                for k in diag_all.keys():
                    if k in d and d[k] is not None:
                        diag_all[k].append(float(d[k]))

        h_all.append(h.detach().cpu())
        hdot_all.append(hdot.detach().cpu())
        relu_hdot_all.append(relu_hdot.detach().cpu())
        relu_ah_all.append(relu_ah.detach().cpu())
        relax_all.append(relax.detach().cpu())
        near_mask_all.append(near_mask.detach().cpu())
        total += int(cur_bs)

    h_all = torch.cat(h_all)
    hdot_all = torch.cat(hdot_all)
    relu_hdot_all = torch.cat(relu_hdot_all)
    relu_ah_all = torch.cat(relu_ah_all)
    relax_all = torch.cat(relax_all)
    near_mask_all = torch.cat(near_mask_all).bool()
    far_mask_all = ~near_mask_all
    if len(hdot_auto_all) > 0:
        hdot_auto_all = torch.cat(hdot_auto_all)
    if len(relax_auto_all) > 0:
        relax_auto_all = torch.cat(relax_auto_all)
        relax_fd_all = torch.cat(relax_fd_all)

    def _bucket_stats(mask: torch.Tensor):
        if mask.sum().item() == 0:
            return {"count": 0, "relax_mean": None, "relax_p95": None, "relax_zero_rate": None}
        vals = relax_all[mask]
        return {
            "count": int(mask.sum().item()),
            "relax_mean": float(vals.mean().item()),
            "relax_p95": float(torch.quantile(vals, 0.95).item()),
            "relax_zero_rate": float((vals <= 0).float().mean().item()),
        }

    out = {
        "num_samples": int(total),
        "batch_size": int(bs),
        "seed": int(seed),
        "alpha": float(alpha),
        "dt": float(dt_eval),
        "near_ratio_target": float(near_ratio),
        "near_ratio_realized": float(near_mask_all.float().mean().item()),
        "near_mode": str(near_mode),
        "fd_obs_source": str(fd_obs_source),
        "infeasible_count": int(infeasible),
        "infeasible_rate": float(infeasible) / float(max(total, 1)),
        "h_mean": float(h_all.mean().item()),
        "h_std": float(h_all.std(unbiased=False).item()),
        "h_p05": float(torch.quantile(h_all, 0.05).item()),
        "h_p50": float(torch.quantile(h_all, 0.50).item()),
        "h_p95": float(torch.quantile(h_all, 0.95).item()),
        "hdot_mean": float(hdot_all.mean().item()),
        "hdot_std": float(hdot_all.std(unbiased=False).item()),
        "relu_hdot_mean": float(relu_hdot_all.mean().item()),
        "relu_hdot_p95": float(torch.quantile(relu_hdot_all, 0.95).item()),
        "relu_alpha_h_mean": float(relu_ah_all.mean().item()),
        "relu_alpha_h_p95": float(torch.quantile(relu_ah_all, 0.95).item()),
        "relax_mean": float(relax_all.mean().item()),
        "relax_max": float(relax_all.max().item()),
        "relax_p95": float(torch.quantile(relax_all, 0.95).item()),
        "relax_zero_rate": float((relax_all <= 0).float().mean().item()),
        "near": _bucket_stats(near_mask_all),
        "far": _bucket_stats(far_mask_all),
    }
    if isinstance(hdot_auto_all, torch.Tensor):
        out["hdot_auto_mean"] = float(hdot_auto_all.mean().item())
        out["hdot_auto_std"] = float(hdot_auto_all.std(unbiased=False).item())
        out["hdot_auto_fd_dt_mae"] = float((hdot_auto_all - hdot_all).abs().mean().item())
        out["relax_auto_mean"] = float(relax_auto_all.mean().item())
        out["relax_auto_p95"] = float(torch.quantile(relax_auto_all, 0.95).item())
        out["relax_auto_zero_rate"] = float((relax_auto_all <= 0).float().mean().item())
        out["relax_fd_mean_same_u"] = float(relax_fd_all.mean().item())
        out["relax_fd_p95_same_u"] = float(torch.quantile(relax_fd_all, 0.95).item())
        out["relax_fd_zero_rate_same_u"] = float((relax_fd_all <= 0).float().mean().item())
        out["relax_auto_lt_fd_rate"] = float((relax_auto_all < relax_fd_all).float().mean().item())
    else:
        out["hdot_auto_mean"] = None
        out["hdot_auto_std"] = None
        out["hdot_auto_fd_dt_mae"] = None
        out["relax_auto_mean"] = None
        out["relax_auto_p95"] = None
        out["relax_auto_zero_rate"] = None
        out["relax_fd_mean_same_u"] = None
        out["relax_fd_p95_same_u"] = None
        out["relax_fd_zero_rate_same_u"] = None
        out["relax_auto_lt_fd_rate"] = None

    fd_scan_out = {}
    for eps in fd_eps_values:
        vals = hdot_fd_scan.get(eps, [])
        if len(vals) == 0:
            continue
        v = torch.cat(vals)
        item = {
            "hdot_fd_mean": float(v.mean().item()),
            "hdot_fd_std": float(v.std(unbiased=False).item()),
        }
        errs = hdot_auto_err_scan.get(eps, [])
        if len(errs) > 0:
            e = torch.cat(errs)
            item["mae_vs_hdot_auto"] = float(e.mean().item())
            item["p95_vs_hdot_auto"] = float(torch.quantile(e, 0.95).item())
        else:
            item["mae_vs_hdot_auto"] = None
            item["p95_vs_hdot_auto"] = None
        fd_scan_out[f"{eps:g}"] = item
    out["fd_eps_scan"] = fd_scan_out

    for k, vals in diag_all.items():
        if len(vals) > 0:
            arr = torch.tensor(vals, dtype=torch.float32)
            out[f"{k}_mean"] = float(arr.mean().item())
            out[f"{k}_p95"] = float(torch.quantile(arr, 0.95).item())
        else:
            out[f"{k}_mean"] = None
            out[f"{k}_p95"] = None
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--ckpt", type=str, required=True, help="Path to Lightning .ckpt")

    # Optional overrides (will start from hparams.yaml next to ckpt unless provided)
    parser.add_argument("--hparams", type=str, default=None, help="Path to hparams.yaml (optional)")
    parser.add_argument("--robot_name", type=str, default="panda")
    parser.add_argument("--gui", action="store_true", help="Enable pybullet GUI for visualization.")

    # Modes
    parser.add_argument(
        "--mode",
        type=str,
        default="metrics",
        choices=["metrics", "rollout", "contour"],
        help="metrics=offline stats only (no rollout); rollout=run moving obstacle rollout; contour=plot BF contour",
    )

    # Metrics options
    parser.add_argument("--num_samples", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--u_clamp", type=float, default=2.5)
    parser.add_argument("--near_ratio", type=float, default=0.0, help="Target ratio of near samples in metrics mode.")
    parser.add_argument(
        "--near_mode",
        type=str,
        default="boundary_or_unsafe",
        choices=["boundary_or_unsafe", "unsafe_only", "boundary_only"],
        help="Definition of near set for metrics sampling/bucketing.",
    )
    parser.add_argument("--fd_eps_list", type=str, default="1e-2,5e-3,1e-3",
                        help="Comma-separated eps values for FD hdot scan, e.g. '1e-2,5e-3,1e-3'.")
    parser.add_argument(
        "--fd_obs_source",
        type=str,
        default="model",
        choices=["model", "raw"],
        help="Observation source used when computing FD hdot. model=controller.h as-is; raw=temporarily disable gphi replacement for h eval.",
    )
    parser.add_argument(
        "--obstacle_horizon_s",
        type=float,
        default=None,
        help="Override obstacle horizon (seconds) for safe/unsafe masks. Use 0 for current-step-only masks.",
    )
    parser.add_argument("--out", type=str, default=None, help="Write metrics JSON to this path")

    # Rollout options (only used when --mode rollout)
    parser.add_argument("--t_sim", type=float, default=6.0)
    parser.add_argument("--speed_scale", type=float, default=1.8)
    parser.add_argument("--obstacle_mode", type=str, default="arm_task", choices=["none", "rigid", "arm", "arm_task"])
    parser.add_argument("--scene", type=str, default="cross_pick", choices=["plain", "cross_pick"])
    parser.add_argument("--block_x", type=float, default=0.50)
    parser.add_argument("--block_y_off", type=float, default=0.10)
    parser.add_argument("--block_z", type=float, default=0.03)
    parser.add_argument("--main_base_y", type=float, default=-0.20)
    parser.add_argument("--obst_base_y", type=float, default=+0.20)
    parser.add_argument("--cross_jitter_amp", type=float, default=0.018)
    parser.add_argument("--cross_jitter_hz", type=float, default=6.0)
    parser.add_argument("--cross_window_ratio", type=float, default=0.35)
    parser.add_argument("--pure_cbf_eval", action="store_true", help="Disable all rollout helper policies; use pure controller.u(x).")
    parser.add_argument("--obst_freeze_on_close", action="store_true", help="Freeze obstacle arm briefly when too close (debug safety helper).")
    parser.add_argument("--continue_after_collision", action="store_true", help="Do not stop rollout when collision is detected; keep running to task end.")
    parser.add_argument("--pause_on_collision", action="store_true")
    parser.add_argument(
        "--start_q",
        type=str,
        default=None,
        help="Optional fixed start q as comma-separated values, e.g. '0.1,0.2,...' (length = q_dims).",
    )
    parser.add_argument(
        "--allow_dirty_start",
        action="store_true",
        help="Allow rollout to proceed even if a collision-free start cannot be found.",
    )
    parser.add_argument(
        "--max_start_resample",
        type=int,
        default=30,
        help="Maximum attempts to resample a collision-free start.",
    )
    parser.add_argument(
        "--start_min_clearance",
        type=float,
        default=0.01,
        help="Minimum required start clearance from obstacles.",
    )
    parser.add_argument("--use_motor_control", action="store_true", help="Use pybullet motor control for the main robot (reduces floor tunneling).")
    parser.add_argument("--max_dq_per_step", type=float, default=0.03, help="Max joint increment per step (rad) to avoid tunneling.")
    parser.add_argument("--no_floor_pause", action="store_true", help="Disable pausing when a link penetrates below the floor tolerance.")

    args_cli = parser.parse_args()

    # Resolve hparams.yaml
    ckpt_path = args_cli.ckpt
    if args_cli.hparams is not None:
        hparams_path = args_cli.hparams
    else:
        # default: sibling hparams.yaml in the same lightning version dir
        hparams_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), "hparams.yaml")
        if not os.path.exists(hparams_path):
            # fallback: same directory
            hparams_path = os.path.join(os.path.dirname(ckpt_path), "hparams.yaml")

    if not os.path.exists(hparams_path):
        raise FileNotFoundError(f"hparams.yaml not found: {hparams_path}")

    with open(hparams_path, "r") as f:
        base_args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))

    # Apply evaluation overrides
    base_args.accelerator = "cpu"  # controller loads to cpu; your training uses GPU elsewhere
    base_args.gui = 1 if args_cli.gui else 0
    base_args.robot_name = args_cli.robot_name
    if args_cli.obstacle_horizon_s is not None:
        base_args.obstacle_horizon_s = float(args_cli.obstacle_horizon_s)

    # Load controller
    neural_controller = init_val(ckpt_path, base_args)

    # Ensure modules are in eval
    try:
        neural_controller.h_nn.eval()
        neural_controller.encoder.eval()
        neural_controller.pc_head.eval()
    except Exception:
        pass

    mode = args_cli.mode

    if mode == "metrics":
        metrics = eval_metrics_offline(
            neural_controller,
            num_samples=args_cli.num_samples,
            batch_size=args_cli.batch_size,
            seed=args_cli.seed,
            alpha=args_cli.alpha,
            u_clamp=args_cli.u_clamp,
            near_ratio=args_cli.near_ratio,
            near_mode=args_cli.near_mode,
            fd_eps_list=args_cli.fd_eps_list,
            fd_obs_source=args_cli.fd_obs_source,
        )
        print(json.dumps(metrics, indent=2))
        if args_cli.out is not None:
            os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
            with open(args_cli.out, "w") as f:
                json.dump(metrics, f, indent=2)

    elif mode == "contour":
        vis_CBF_contour(neural_controller)

    elif mode == "rollout":
        start_q_override = None
        if args_cli.start_q is not None:
            toks = [t.strip() for t in str(args_cli.start_q).split(",") if t.strip() != ""]
            try:
                vals = np.array([float(t) for t in toks], dtype=np.float32)
            except Exception as e:
                raise ValueError(f"Invalid --start_q: {args_cli.start_q}") from e
            q_dims = int(neural_controller.dynamics_model.n_dims)
            if vals.shape[0] != q_dims:
                raise ValueError(f"--start_q length must be {q_dims}, got {vals.shape[0]}")
            start_q_override = vals

        rollout_result = run_moving_obstacle_rollout(
            neural_controller,
            t_sim=float(args_cli.t_sim),
            move_obstacles=True,
            seed=int(args_cli.seed),
            realtime=False,
            realtime_scale=1.0,
            speed_scale=float(args_cli.speed_scale),
            stop_on_goal=False,
            goal_tol=0.10,
            print_every=120,
            amp_range=(0.08, 0.20),
            omega_range=(1.2, 3.0),
            obstacle_mode=str(args_cli.obstacle_mode),
            obstacle_arm_seed=int(args_cli.seed),
            obstacle_arm_base_xyz=(0.35, 0.25, 0.0),
            obstacle_arm_base_rpy=(0.0, 0.0, 0.0),
            obstacle_arm_strength=260.0,
            pause_on_goal=False,
            goal_pause_tol=1e-4,
            obstacle_arm_amp_scale=1.4,
            obstacle_arm_omega_scale=1.0,
            pause_on_collision=bool(args_cli.pause_on_collision),
            require_clean_start=not bool(args_cli.allow_dirty_start),
            max_start_resample=int(args_cli.max_start_resample),
            start_min_clearance=float(args_cli.start_min_clearance),
            start_q_override=start_q_override,
            use_motor_control=bool(args_cli.use_motor_control),
            max_dq_per_step=float(args_cli.max_dq_per_step),
            pause_on_floor_penetration=not bool(args_cli.no_floor_pause),
            scene=str(args_cli.scene),
            block_x=float(args_cli.block_x),
            block_y_off=float(args_cli.block_y_off),
            block_z=float(args_cli.block_z),
            main_base_y=float(args_cli.main_base_y),
            obst_base_y=float(args_cli.obst_base_y),
            cross_jitter_amp=float(args_cli.cross_jitter_amp),
            cross_jitter_hz=float(args_cli.cross_jitter_hz),
            cross_window_ratio=float(args_cli.cross_window_ratio),
            pure_cbf_eval=bool(args_cli.pure_cbf_eval),
            obst_freeze_on_close=bool(args_cli.obst_freeze_on_close),
            continue_after_collision=bool(args_cli.continue_after_collision),
        )
        if args_cli.out is not None:
            os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
            with open(args_cli.out, "w") as f:
                json.dump(rollout_result, f, indent=2)
