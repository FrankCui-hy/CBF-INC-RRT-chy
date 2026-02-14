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
import seaborn

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
	obstacle_robot_name = getattr(args, "obstacle_robot_name", None)
	obstacle_traj_path = getattr(args, "obstacle_traj_path", None)
	obstacle_robot_base_pos = getattr(args, "obstacle_robot_base_pos", (0.3, 0.0, 0.0))
	obstacle_robot_base_orn = getattr(args, "obstacle_robot_base_orn", (0.0, 0.0, 0.0, 1.0))
	environment = ArmEnv(
		[args.robot_name],
		GUI=gui_flag,
		config_file=config_file,
		obstacle_robot_name=obstacle_robot_name,
		obstacle_traj_path=obstacle_traj_path,
		obstacle_robot_base_pos=obstacle_robot_base_pos,
		obstacle_robot_base_orn=obstacle_robot_base_orn,
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
		"u_coef_in_training": args.u_coef_in_training,
		"safe_classification_weight": args.safe_classification_weight,
		"unsafe_classification_weight": args.unsafe_classification_weight,
		"descent_violation_weight": args.descent_violation_weight,
		"hdot_divergence_weight": args.hdot_divergence_weight,
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
														 use_bn=args.use_bn,
														 cbf_hidden_layers=args.cbf_hidden_layers,
														 cbf_hidden_size=args.cbf_hidden_size,
														 cbf_alpha=args.cbf_alpha,
														 cbf_relaxation_penalty=args.cbf_relaxation_penalty,
														 feature_dim=args.feature_dim,
														 per_feature_dim=args.per_feature_dim,
														 loss_config=loss_config,
														 controller_period=args.controller_period,
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


def _get_eval_obstacle_ids(env: ArmEnv, robot_id: int = None):
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
        if robot_id is not None and oid == robot_id:
            continue
        if _looks_like_plane_or_floor(oid):
            continue
        kept.append(oid)

    # If filtering removed everything, fall back to raw_ids excluding robot (better than nothing)
    if not kept:
        kept = [oid for oid in raw_ids if (robot_id is None or oid != robot_id)]

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
def _remove_all_obstacles(env: ArmEnv, robot_id: int = None):
    """Remove all non-floor obstacles from the pybullet world.

    This is used when you want a clean, obstacle-free evaluation scene.
    """
    p_ = env.p
    obstacle_ids = _get_eval_obstacle_ids(env, robot_id)
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
def _get_ee_pos(robot) -> np.ndarray:
	"""End-effector position for quick visualization/printing."""
	try:
		ls = p.getLinkState(robot.robotId, robot.body_joints[-1])
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


def _get_arm_ee_pos(body_id: int, ee_link_index: int = None) -> np.ndarray:
    """Return end-effector position for a pybullet body."""
    try:
        if ee_link_index is None:
            ee_link_index = p.getNumJoints(body_id) - 1
        ls = p.getLinkState(body_id, ee_link_index)
        return np.array(ls[4], dtype=np.float32)
    except Exception:
        return np.zeros((3,), dtype=np.float32)


# ---- Ensure non-colliding rollout start ----
def _ensure_noncolliding_start(controller: NeuralLidarCBFController,
                               x: torch.Tensor,
                               min_clearance: float = 0.01,
                               max_tries: int = 30):
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
    obstacle_ids = _get_eval_obstacle_ids(env, robot.robotId)
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

    # If we couldn't find one, return the refreshed original and let the caller see the failure
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
):
	"""Run a single closed-loop rollout. If move_obstacles=True, obstacles move sinusoidally or as a second arm.

	Prints collision status and returns a dict with trajectory statistics.
	"""
	controller.eval()
	dm = controller.dynamics_model
	env = dm.env
	robot = dm.robot
	p_ = env.p

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
		print(f"[ROLL] obstacle_mode=none -> removed {len(removed)} obstacles: {removed}")

	elif mode == "arm":
		# User request: delete the original rigid/box obstacles, keep only the obstacle arm.
		removed = _remove_all_obstacles(env, robot.robotId)
		obstacle_ids = []
		print(f"[ROLL] obstacle_mode=arm -> removed {len(removed)} rigid obstacles: {removed}")

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

	if start_x is None:
		# Fallback: start near mid of limits
		ul, ll = dm.state_limits
		q0 = torch.lerp(ll, ul, 0.4 * torch.ones(ll.shape[-1]).double()).reshape(1, -1).float()
		x = dm.complete_sample_with_observations(q0, num_samples=1)
	else:
		# Only take q from the saved start state; recompute obs/aux for the current env
		q0 = start_x[0, :dm.n_dims].detach().clone()
		x = dm.complete_sample_with_observations(q0.reshape(1, -1), num_samples=1)

	# Make sure we don't start in collision
	x = _ensure_noncolliding_start(controller, x, min_clearance=0.01, max_tries=30)

	# Ensure robot is in sync with x at t=0
	q = x[0, :dm.n_dims]
	robot.set_joint_position(robot.body_joints, q)
	p_.stepSimulation()
	# Visualize and print start/goal (end-effector markers)
	start_ee = _get_ee_pos(robot)
	# Put the robot at goal once to get goal EE, then restore
	q_goal = dm.goal_state[:dm.n_dims].detach().clone().float()
	q_save = q.detach().clone()
	robot.set_joint_position(robot.body_joints, q_goal)
	p_.stepSimulation()
	goal_ee = _get_ee_pos(robot)
	# restore start
	robot.set_joint_position(robot.body_joints, q_save)
	p_.stepSimulation()
	print(f"[START/GOAL] start_ee={start_ee.tolist()}  goal_ee={goal_ee.tolist()}")
	_spawn_marker(start_ee, rgba=(0, 1, 0, 0.8), radius=0.035)
	_spawn_marker(goal_ee, rgba=(1, 0, 0, 0.8), radius=0.035)
	# Make sim stepping consistent with the dynamics dt
	try:
		p_.setRealTimeSimulation(0)
		p_.setTimeStep(dm.dt)
	except Exception:
		pass

	# --- moving obstacle source ---
	obstacle_arm = None
	base = direction = omega = amp = None

	if mode == "arm":
		# Spawn a second arm as the moving obstacle
		try:
			# allow passing an explicit URDF path
			if obstacle_arm_urdf is not None:
				setattr(env, "obstacle_arm_urdf", obstacle_arm_urdf)
			obstacle_arm = _spawn_obstacle_arm(
				env,
				main_robot=robot,
				base_xyz=tuple(obstacle_arm_base_xyz),
				base_rpy=tuple(obstacle_arm_base_rpy),
				seed=int(obstacle_arm_seed),
				urdf_path=obstacle_arm_urdf,
				use_fixed_base=True,
				amp_scale=float(obstacle_arm_amp_scale),
				omega_scale=float(obstacle_arm_omega_scale),
			)
			# Make sure collision/distance checks include the obstacle arm
			if obstacle_arm["arm_id"] not in obstacle_ids:
				obstacle_ids = [obstacle_arm["arm_id"]] + list(obstacle_ids)
			print(f"[OBST_ARM] spawned id={obstacle_arm['arm_id']} base={obstacle_arm['base_xyz'].tolist()}")
		except Exception as e:
			print(f"[OBST_ARM] ERROR spawning obstacle arm: {e}")
			obstacle_arm = None

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

	for k in range(steps):
		# Base time used for obstacle motion
		t_base = (k * dm.dt) * float(obstacle_speed_scale)
		# Optionally speed up ONLY the obstacle arm (separate from rigid obstacles)
		t_arm = t_base * float(obstacle_arm_speed_scale)

		# 1) Move the obstacle(s) first (so observation sees the new positions)
		if move_obstacles:
			if mode == "arm" and obstacle_arm is not None:
				_update_obstacle_arm(env, obstacle_arm, t_arm, strength=float(obstacle_arm_strength))
				# optional tiny debug prints
				if k < 3:
					ee = _get_arm_ee_pos(obstacle_arm["arm_id"], ee_link_index=(p_.getNumJoints(obstacle_arm["arm_id"]) - 1))
					print(f"[OBST_ARM] t={t_arm:.3f} ee={ee.tolist()}")
			elif mode == "rigid":
				_update_obstacles(env, obstacle_ids, t_base, base, direction, omega, amp)
			# else: mode == "none" -> do nothing

		# 2) Compute control using current datax (q + obs + aux)
		u = controller.u(x)[0]
		# Make the arm move faster/slower (visual + actual) while keeping it bounded
		u = u * float(speed_scale)
		# Conservative default clamp if the dynamics doesn't expose limits
		try:
			u_hi, u_lo = getattr(dm, "control_limits")
			u = torch.max(torch.min(u, u_hi), u_lo)
		except Exception:
			u = torch.clamp(u, -2.5, 2.5)

		# 3) Step dynamics with observation update
		x = dm.closed_loop_dynamics(x, u, collect_dataset=False, use_motor_control=False, update_observation=True)

		# 4) Advance physics (if dm.closed_loop_dynamics didn't already step physics)
		p_.stepSimulation()

		# Goal progress (in joint space)
		q_now = x[0, :dm.n_dims]
		d_goal = torch.norm(q_now - q_goal.to(q_now.device)).item()
		if (k % max(int(print_every), 1)) == 0:
			if mode != "none" and len(min_dist_hist) > 0:
				md = min_dist_hist[-1]
			else:
				md = float("nan")
			print(f"[ROLL] step={k:5d}/{steps}  t={k*dm.dt:6.3f}s  ||q-goal||={d_goal:.3f}  min_d={md:.4f}")
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
			print(f"[ROLL] reached goal: ||q-goal||={d_goal:.6f} <= {float(goal_tol):.6f} at step {k}")
			break

		# 5) Measure collision / distance (skip if obstacle_mode==none)
		if mode != "none" and len(obstacle_ids) > 0:
			min_d, hit = _min_distance_and_collision(env, robot.robotId, obstacle_ids, distance=2.0)
			min_dist_hist.append(min_d)
			if hit:
				collided = True
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
				break

		if realtime:
			# realtime_scale > 1 slows down the visualization (e.g., 2.0 means 2x slower than real time)
			sleep_dt = max(dm.dt * float(realtime_scale), 1.0 / 60.0)
			time.sleep(sleep_dt)

	result = {
		"move_obstacles": move_obstacles,
		"seed": seed,
		"t_sim": t_sim,
		"steps_ran": (collide_step if collided else (k + 1)),
		"collided": collided,
		"min_dist_min": float(np.min(min_dist_hist)) if len(min_dist_hist) else None,
		"min_dist_mean": float(np.mean(min_dist_hist)) if len(min_dist_hist) else None,
	}
	print("[ROLL] move_obstacles=", move_obstacles,
			" seed=", seed,
			" collided=", collided,
			" steps=", result["steps_ran"],
			" min_dist_min=", result["min_dist_min"])
	# Best-effort cleanup to avoid BulletClient __del__ warnings on interpreter shutdown
	try:
		p_.disconnect()
	except Exception:
		try:
			p.disconnect()
		except Exception:
			pass
	return result


if __name__ == "__main__":
	# Load the checkpoint file. This should include the experiment suite used during training.
	robot_name = "panda"
	log_dir = "./models/neural_cbf/"
	git_version = f"Franka_Panda_lidar_Dynamics/multiple_seeds/version_10/"

	log_file = "checkpoints/epoch=119-step=151919.ckpt"  # specify the checkpoint file

	# load arguments from yaml
	with open(log_dir + git_version + 'hparams.yaml', 'r') as f:
		args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))
	args.accelerator = 'cpu'
	args.n_observation = 1024
	args.gui = 1
	# Ensure eval config matches checkpoint expectations
	args.obstacle_robot_name = getattr(args, "obstacle_robot_name", "panda")
	if not hasattr(args, "dataset_name") or args.dataset_name is None:
		args.dataset_name = "ocbf_panda_vel_norm"
	if "norm" not in str(args.dataset_name):
		print(f"[eval] Warning: dataset_name={args.dataset_name} has no 'norm' token. "
			  f"Checkpoint may expect normals.")
	# Evaluation-only overrides
	# Make sure we use the same observation count as the trained checkpoint expects
	# (if you trained with 64, set 64; if 1024, keep 1024).
	# args.n_observation = 64
	args.goal_xyz = [0.55, 0.0, 0.45]
	# args.simulation_dt = 0.01
	# args.controller_period = 0.01
	args.cbf_relaxation_penalty = 50000.
	args.cbf_alpha = 20
	# args.dis_threshold = 0.02
	# args.observation_type = 'uniform_lidar'

	# args.use_bn = 0

	# args.dataset_name = 'prob08_motor'
	# args.max_episode=100
	# args.trajectories_per_episode=40
	# args.trajectory_length=35

	neural_controller = init_val(log_dir + git_version + log_file, args)

	neural_controller.h_nn.eval()
	neural_controller.encoder.eval()
	neural_controller.pc_head.eval()


	# neural_controller.h_alpha=0.3
	# vis_misclassification(neural_controller, log_path = log_dir+ git_version)
	# vis_traj_rollout(neural_controller)

	# 1) Moving obstacle-arm test (remove boxes, keep only obstacle arm)
	run_moving_obstacle_rollout(
		neural_controller,
		t_sim=20.0,
		move_obstacles=True,
		seed=0,
		realtime=True,
		realtime_scale=1.0,
		speed_scale=1.8,
		stop_on_goal=False,
		goal_tol=0.10,
		print_every=120,
		amp_range=(0.08, 0.20),
		omega_range=(1.2, 3.0),
		obstacle_mode="arm",
		obstacle_arm_seed=0,
		obstacle_arm_base_xyz=(0.35, 0.25, 0.0),
		obstacle_arm_base_rpy=(0.0, 0.0, 0.0),
		obstacle_arm_strength=260.0,
		pause_on_goal=True,
		goal_pause_tol=1e-4,
		obstacle_arm_amp_scale=1.4,
		obstacle_arm_omega_scale=1.0,
		pause_on_collision=True,
	)

	# If you still want the contour plot, uncomment:
	# vis_CBF_contour(neural_controller)
