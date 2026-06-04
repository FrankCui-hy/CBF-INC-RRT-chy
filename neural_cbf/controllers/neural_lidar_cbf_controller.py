import itertools
import time
from functools import partial
from typing import Tuple, List, Optional
from collections import OrderedDict
import random
import tqdm

import pybullet as p

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.func import jvp

import matplotlib.pyplot as plt

from neural_cbf.systems import ArmLidar
from neural_cbf.systems.utils import ScenarioList, cartesian_to_spherical
from neural_cbf.controllers import NeuralObsCBFController
from neural_cbf.controllers.cbf_observation_builder import (
	CBFObservationBuilder,
	compute_metadata_fingerprint,
	raylink_cbf_metadata,
	_translation_from_T,
	_quat_xyzw_from_T,
)
from neural_cbf.controllers.utils import PointNetfeat, PointNetVanillaEncoder
from neural_cbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_cbf.experiments import ExperimentSuite
from loss.models.raylink_g_phi import RayLinkMLPGPhi


class NeuralLidarCBFController(NeuralObsCBFController):
	"""
	h:  (observations -> encoder) + state -> fully-connected layers -> h
	"""

	def __init__(
			self,
			dynamics_model: ArmLidar,
			scenarios: ScenarioList,
			datamodule: EpisodicDataModule,
			experiment_suite: ExperimentSuite,
			**kwargs,
	):
		"""Initialize the controller.

		args:
			dynamics_model: the control-affine dynamics of the underlying system
			scenarios: a list of parameter scenarios to train on
			experiment_suite: defines the experiments to run during training
			cbf_hidden_layers: number of hidden layers to use for the CBF network
			cbf_hidden_size: number of neurons per hidden layer in the CBF network
			cbf_lambda: convergence rate for the CBF
			cbf_relaxation_penalty: the penalty for relaxing CBF conditions.
			controller_period: the timestep to use in simulating forward Vdot
			learn_shape_epochs: number of epochs to spend just learning the shape
			state_only: if True, define the barrier function in terms of robot state
		"""
		super(NeuralLidarCBFController, self).__init__(
			dynamics_model=dynamics_model,
			scenarios=scenarios,
			datamodule=datamodule,
			experiment_suite=experiment_suite,
			**kwargs,
		)

		self.all_encoded_obs_dim = kwargs["feature_dim"]
		self.obstacle_qdot_dim = getattr(self.dynamics_model, "obstacle_qdot_dim", 0)
		self.z_dim = self.all_encoded_obs_dim + self.obstacle_qdot_dim
		self.n_dims_extended = self.dynamics_model.n_dims + self.dynamics_model.o_dims

		# ----------------------------------------------------------------------------
		# Define the encoder network
		# ----------------------------------------------------------------------------
		self.pc_head = PointNetfeat(num_sensor=len(self.dynamics_model.list_sensor),
									ray_per_sensor=self.dynamics_model.ray_per_sensor,
									input_channel=self.dynamics_model.point_dims,
									output_channel=kwargs["per_feature_dim"],
									use_bn=kwargs["use_bn"], )
		self.encoder = PointNetVanillaEncoder(num_sensor=len(self.dynamics_model.list_sensor),
											  ray_per_sensor=self.dynamics_model.ray_per_sensor,
											  input_channel=kwargs["per_feature_dim"],
											  output_dim=self.all_encoded_obs_dim)

		# ----------------------------------------------------------------------------
		# Define the BF network, which we denote h
		# ----------------------------------------------------------------------------
		num_h_inputs = self.dynamics_model.n_dims + self.z_dim

		# CBF head
		self.h_layers: OrderedDict[str, nn.Module] = OrderedDict()
		self.h_layers["input_linear"] = nn.Linear(num_h_inputs, self.h_hidden_size)
		self.h_layers["input_activation"] = nn.LeakyReLU(0.1)
		for i in range(self.h_hidden_layers):
			self.h_layers[f"layer_{i}_linear"] = nn.Linear(
				self.h_hidden_size, self.h_hidden_size
			)
			self.h_layers[f"layer_{i}_activation"] = nn.LeakyReLU(0.1)
		self.h_layers["output_linear"] = nn.Linear(self.h_hidden_size, 1)
		self.h_nn = nn.Sequential(self.h_layers)

		# ----------------------------------------------------------------------------
		# Define the actor network, which we denote u
		# ----------------------------------------------------------------------------
		self.use_neural_actor = kwargs["use_neural_actor"]
		self.ab_mode = kwargs.get("ab_mode", "B_with_normal")
		if self.ab_mode not in ("A_no_normal", "B_with_normal"):
			raise ValueError(f"Unknown ab_mode={self.ab_mode}. Use A_no_normal or B_with_normal.")
		self.baseline = bool(kwargs.get("baseline", False))
		self.obs_backend = kwargs.get("obs_backend", "gphi")
		self.cbf_obs_mode = kwargs.get("cbf_obs_mode", None)
		if self.cbf_obs_mode is None:
			self.cbf_obs_mode = "gphi" if ((not self.baseline) and self.obs_backend == "gphi") else "legacy_oracle"
		self.gphi_ckpt = kwargs.get("gphi_ckpt", "")
		self.gphi_hit_threshold = float(kwargs.get("gphi_hit_threshold", 0.5))
		self.gphi_hit_temp = float(kwargs.get("gphi_hit_temp", 0.1))
		self.gphi_freeze = bool(kwargs.get("gphi_freeze", True))
		self.gphi_include_qobs_dynamics = bool(kwargs.get("gphi_include_qobs_dynamics", False))
		self.gphi_metadata = None
		self.gphi_config = None
		self.gphi_metadata_fingerprint = ""
		self.train_use_fd = bool(kwargs.get("train_use_fd", self.baseline))
		self._validate_method_config()
		self.use_gphi_chain = self.cbf_obs_mode == "gphi"
		self.use_raylink_layout = self.cbf_obs_mode in ("gphi", "raylink_oracle")
		self.g_phi = None
		if self.use_raylink_layout:
			self._init_g_phi()
			self._validate_raylink_metadata()
		self.obs_builder = CBFObservationBuilder(
			mode=self.cbf_obs_mode,
			q_dim=self.dynamics_model.n_dims,
			obs_dim=self.dynamics_model.o_dims_in_dataset,
			aux_dim=self.dynamics_model.state_aux_dims_in_dataset,
			g_phi=self.g_phi,
			gphi_metadata=self.gphi_metadata,
			r_max=float((self.gphi_metadata or {}).get("r_max", 5.0)),
			hit_threshold=self.gphi_hit_threshold,
			hit_temp=self.gphi_hit_temp,
			add_normal=bool(self.dynamics_model.add_normal),
			point_dim=int(self.dynamics_model.point_dims),
			raycast_env=getattr(self.dynamics_model, "env", None),
			raycast_ego_robot=getattr(self.dynamics_model, "robot", None),
			raycast_obstacle_robot=getattr(getattr(self.dynamics_model, "env", None), "obstacle_robot", None),
		)
		self._validate_state_label_cache_metadata()
		if self.use_neural_actor:
			# actor head
			self.actor_layers: OrderedDict[str, nn.Module] = OrderedDict()
			self.actor_layers["input_linear"] = nn.Linear(num_h_inputs + self.dynamics_model.n_controls,
														  self.h_hidden_size)
			self.actor_layers["input_activation"] = nn.ReLU()
			for i in range(self.h_hidden_layers):
				self.actor_layers[f"layer_{i}_linear"] = nn.Linear(
					self.h_hidden_size, self.h_hidden_size
				)
				self.actor_layers[f"layer_{i}_activation"] = nn.ReLU()
			self.actor_layers["output_linear"] = nn.Linear(self.h_hidden_size, self.dynamics_model.n_dims)
			self.actor_layers["output_clamp"] = nn.Sigmoid()
			self.actor_nn = nn.Sequential(self.actor_layers)

	def _validate_method_config(self):
		if self.obs_backend not in ("gphi", "raw"):
			raise ValueError(f"Unknown obs_backend={self.obs_backend}. Use gphi or raw.")
		if self.cbf_obs_mode not in ("legacy_oracle", "gphi", "raylink_oracle"):
			raise ValueError(f"Unknown cbf_obs_mode={self.cbf_obs_mode}.")
		if self.gphi_include_qobs_dynamics:
			if self.cbf_obs_mode != "gphi":
				raise ValueError("gphi_include_qobs_dynamics is only supported for cbf_obs_mode='gphi'.")
			if self.train_use_fd:
				raise ValueError("gphi_include_qobs_dynamics requires analytic chain rule, i.e. train_use_fd=False.")
		if self.baseline:
			if self.cbf_obs_mode != "legacy_oracle":
				raise ValueError("baseline=True requires cbf_obs_mode='legacy_oracle'.")
			if not self.train_use_fd:
				raise ValueError("baseline=True requires train_use_fd=True so the baseline uses the legacy FD/simulated Lie derivative.")
			return
		if self.cbf_obs_mode == "legacy_oracle":
			return
		if self.cbf_obs_mode == "gphi":
			if self.train_use_fd:
				raise ValueError("cbf_obs_mode='gphi' requires train_use_fd=False so analytic chain-rule hdot is used.")
			if self.gphi_ckpt is None or str(self.gphi_ckpt).strip() == "":
				raise ValueError("cbf_obs_mode='gphi' requires a non-empty gphi_ckpt.")
			if bool(self.dynamics_model.add_normal):
				raise ValueError("cbf_obs_mode='gphi' is point-only; use a non-normal CBF dataset/model.")
			if int(self.dynamics_model.point_dims) != 3:
				raise ValueError("cbf_obs_mode='gphi' currently requires point_dim=3 and no extra point channels.")
		if self.cbf_obs_mode == "raylink_oracle":
			if not self.train_use_fd:
				raise ValueError(
					"cbf_obs_mode='raylink_oracle' currently supports train_use_fd=True only. "
					"Oracle raycast is not differentiable, so analytic chain rule is not supported."
				)
			if self.gphi_ckpt is None or str(self.gphi_ckpt).strip() == "":
				raise ValueError("cbf_obs_mode='raylink_oracle' requires a non-empty gphi_ckpt for RayLink metadata.")
			if bool(self.dynamics_model.add_normal):
				raise ValueError("cbf_obs_mode='raylink_oracle' is point-only; use a non-normal CBF dataset/model.")
			if int(self.dynamics_model.point_dims) != 3:
				raise ValueError("cbf_obs_mode='raylink_oracle' currently requires point_dim=3 and no extra point channels.")

	def method_metadata(self) -> dict:
		method = "baseline_fd_raw" if self.baseline else f"cbf_obs_{self.cbf_obs_mode}"
		return {
			"method": method,
			"baseline": bool(self.baseline),
			"obs_backend": str(self.obs_backend),
			"cbf_obs_mode": str(self.cbf_obs_mode),
			"ab_mode": str(self.ab_mode),
			"train_use_fd": bool(self.train_use_fd),
			"use_gphi_chain": bool(self.use_gphi_chain),
			"use_raylink_layout": bool(getattr(self, "use_raylink_layout", False)),
			"gphi_ckpt": str(self.gphi_ckpt or ""),
			"gphi_hit_threshold": float(self.gphi_hit_threshold),
			"gphi_hit_temp": float(self.gphi_hit_temp),
			"gphi_include_qobs_dynamics": bool(self.gphi_include_qobs_dynamics),
			"gphi_metadata_fingerprint": str(self.gphi_metadata_fingerprint),
		}

	def _init_g_phi(self):
		ckpt = torch.load(self.gphi_ckpt, map_location="cpu")
		if "metadata" not in ckpt:
			raise KeyError("RayLink checkpoint must contain metadata.")
		if self.cbf_obs_mode == "gphi" and "model_state_dict" not in ckpt:
			raise KeyError("RayLink g_phi checkpoint must contain model_state_dict for cbf_obs_mode='gphi'.")
		self.gphi_metadata = ckpt["metadata"]
		self.gphi_config = ckpt.get("config", {})
		model_cfg = self.gphi_config.get("model", {}) if isinstance(self.gphi_config, dict) else {}
		self.g_phi = RayLinkMLPGPhi(
			self.gphi_metadata,
			pair_hidden_dim=int(model_cfg.get("pair_hidden_dim", 128)),
			head_hidden_dims=list(model_cfg.get("head_hidden_dims", [128, 64])),
			link_embed_dim=int(model_cfg.get("link_embed_dim", 8)),
			anchor_embed_dim=int(model_cfg.get("anchor_embed_dim", 8)),
			activation=str(model_cfg.get("activation", "silu")),
			finger_open=float(model_cfg.get("finger_open", 0.04)),
		)
		if "model_state_dict" in ckpt:
			self.g_phi.load_state_dict(ckpt["model_state_dict"])
		self.g_phi.eval()
		if self.gphi_freeze or self.cbf_obs_mode == "raylink_oracle":
			for p in self.g_phi.parameters():
				p.requires_grad_(False)
		fp_meta = raylink_cbf_metadata(
			self.gphi_metadata,
			gphi_ckpt=str(self.gphi_ckpt or ""),
			add_normal=bool(self.dynamics_model.add_normal),
			point_dim=int(self.dynamics_model.point_dims),
		)
		self.gphi_metadata_fingerprint = compute_metadata_fingerprint(fp_meta)

	def _validate_raylink_metadata(self):
		meta = self.gphi_metadata or {}
		required_keys = ("T_W_Bego", "T_W_Bobs", "anchor_link_ids", "anchor_T_L_S", "local_ray_dirs", "r_max")
		missing = [key for key in required_keys if key not in meta]
		if missing:
			raise ValueError(f"RayLink metadata is missing required keys: {missing}.")

		local_ray_dirs = torch.as_tensor(meta["local_ray_dirs"])
		if local_ray_dirs.ndim != 3 or int(local_ray_dirs.shape[-1]) != 3:
			raise ValueError(
				"RayLink metadata local_ray_dirs must have shape [num_anchors, rays_per_anchor, 3], "
				f"got {tuple(local_ray_dirs.shape)}."
			)
		num_anchors = int(local_ray_dirs.shape[0])
		rays_per_anchor = int(local_ray_dirs.shape[1])
		num_rays = int(num_anchors * rays_per_anchor)
		if int(meta.get("num_anchors", num_anchors)) != num_anchors:
			raise ValueError("RayLink metadata num_anchors does not match local_ray_dirs.")
		if int(meta.get("num_rays_per_anchor", rays_per_anchor)) != rays_per_anchor:
			raise ValueError("RayLink metadata num_rays_per_anchor does not match local_ray_dirs.")
		if int(meta.get("num_rays_total", num_rays)) != num_rays:
			raise ValueError("RayLink metadata num_rays_total does not match num_anchors * rays_per_anchor.")
		if self.g_phi is not None and int(getattr(self.g_phi, "num_rays", num_rays)) != num_rays:
			raise ValueError("RayLink model num_rays does not match metadata num_rays_total.")
		if len(meta.get("anchor_link_ids", [])) != num_anchors:
			raise ValueError("RayLink metadata anchor_link_ids length does not match num_anchors.")

		ray_order = str(meta.get("ray_ordering_rule", meta.get("ray_order", "anchor_major"))).lower()
		if "anchor_major" not in ray_order and not ("anchor" in ray_order and "local_ray_index" in ray_order):
			raise ValueError(f"raylink_oracle requires anchor-major RayLink ray order, got {ray_order!r}.")

		if self.cbf_obs_mode == "raylink_oracle":
			env = getattr(self.dynamics_model, "env", None)
			robot = getattr(self.dynamics_model, "robot", None)
			obstacle_robot = getattr(env, "obstacle_robot", None) if env is not None else None
			if env is None or not hasattr(env, "p") or robot is None or obstacle_robot is None:
				raise ValueError("cbf_obs_mode='raylink_oracle' requires a live ArmEnv with ego and obstacle Panda bodies.")
			self._validate_raylink_env_pose(env, robot, obstacle_robot)

	def _validate_raylink_env_pose(self, env, robot, obstacle_robot):
		def _close_vec(label, actual, expected, tol=1e-5):
			if expected is None:
				raise ValueError(f"RayLink metadata is missing {label}.")
			if len(actual) != len(expected) or any(abs(float(a) - float(b)) > tol for a, b in zip(actual, expected)):
				raise ValueError(f"{label} mismatch: env has {list(actual)}, RayLink metadata has {list(expected)}.")

		ego_expected = _translation_from_T(self.gphi_metadata.get("T_W_Bego"))
		obs_expected = _translation_from_T(self.gphi_metadata.get("T_W_Bobs"))
		obs_quat_expected = _quat_xyzw_from_T(self.gphi_metadata.get("T_W_Bobs"))
		ego_pos, _ = env.p.getBasePositionAndOrientation(int(robot.robotId))
		obs_pos, obs_quat = env.p.getBasePositionAndOrientation(int(obstacle_robot.robotId))
		_close_vec("ego_base_pos", ego_pos, ego_expected)
		_close_vec("obs_base_pos", obs_pos, obs_expected)
		if obs_quat_expected is None:
			raise ValueError("RayLink metadata is missing obs_base_quat/T_W_Bobs rotation.")
		same = all(abs(float(a) - float(b)) <= 1e-5 for a, b in zip(obs_quat, obs_quat_expected))
		neg_same = all(abs(float(a) + float(b)) <= 1e-5 for a, b in zip(obs_quat, obs_quat_expected))
		if not (same or neg_same):
			raise ValueError(f"obs_base_quat mismatch: env has {list(obs_quat)}, RayLink metadata has {list(obs_quat_expected)}.")

	def _validate_state_label_cache_metadata(self):
		if self.cbf_obs_mode not in ("gphi", "raylink_oracle"):
			return
		state_meta = getattr(self.datamodule, "state_label_metadata", None)
		if not state_meta or not self.gphi_metadata:
			return

		def _translation_from_T(T):
			if T is None:
				return None
			return [float(T[0][3]), float(T[1][3]), float(T[2][3])]

		ego_pos = _translation_from_T(self.gphi_metadata.get("T_W_Bego"))
		obs_pos = _translation_from_T(self.gphi_metadata.get("T_W_Bobs"))
		for key, gphi_value in (("ego_base_pos", ego_pos), ("obs_base_pos", obs_pos)):
			cache_value = state_meta.get(key, None)
			if cache_value is None or gphi_value is None:
				continue
			if any(abs(float(a) - float(b)) > 1e-6 for a, b in zip(cache_value, gphi_value)):
				raise ValueError(
					f"state_label_cache metadata {key}={cache_value} does not match g_phi metadata {key}={gphi_value}."
				)
		obs_quat = _quat_xyzw_from_T(self.gphi_metadata.get("T_W_Bobs"))
		cache_quat = state_meta.get("obs_base_quat", None)
		if cache_quat is not None and obs_quat is not None:
			same = all(abs(float(a) - float(b)) <= 1e-6 for a, b in zip(cache_quat, obs_quat))
			neg_same = all(abs(float(a) + float(b)) <= 1e-6 for a, b in zip(cache_quat, obs_quat))
			if not (same or neg_same):
				raise ValueError(
					f"state_label_cache metadata obs_base_quat={cache_quat} does not match g_phi metadata obs_base_quat={obs_quat}."
				)

	def prepare_data(self):
		out = super().prepare_data()
		self._validate_state_label_cache_metadata()
		return out

	def parse_state_from_datax(self, datax: torch.Tensor):
		q_ego = datax[:, : self.dynamics_model.n_dims]
		q_obs = self.dynamics_model.get_obstacle_q_from_datax(datax)
		qdot_obs, _, _ = self.dynamics_model.get_obstacle_meta_from_datax(datax)
		aux = datax[:, -self.dynamics_model.state_aux_dims_in_dataset :]
		if self.cbf_obs_mode in ("gphi", "raylink_oracle") and q_obs is None:
			raise ValueError(
				f"cbf_obs_mode='{self.cbf_obs_mode}' requires q_obs in datax auxv2. "
				"Use scripts/extract_cbf_state_label_cache.py on an auxv2 cache or regenerate the CBF data."
			)
		if self.gphi_include_qobs_dynamics:
			qdot_obs = self._require_qdot_obs(qdot_obs, datax=datax, q_obs=q_obs)
		return q_ego, q_obs, qdot_obs, aux

	def _require_qdot_obs(
			self,
			qdot_obs: Optional[torch.Tensor],
			datax: Optional[torch.Tensor] = None,
			q_obs: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		if qdot_obs is None:
			raise ValueError(
				"gphi_include_qobs_dynamics=True requires qdot_obs in datax/state-label cache, "
				"but qdot_obs could not be parsed. Please regenerate state-label cache with auxv2."
			)
		if qdot_obs.ndim != 2:
			raise ValueError(
				"gphi_include_qobs_dynamics=True requires rank-2 qdot_obs, "
				f"got shape {tuple(qdot_obs.shape)}."
			)
		expected_dim = int(getattr(self.dynamics_model, "obstacle_qdot_dim", self.dynamics_model.n_dims))
		if int(qdot_obs.shape[1]) != expected_dim:
			raise ValueError(
				"gphi_include_qobs_dynamics=True requires qdot_obs width "
				f"{expected_dim}, got {int(qdot_obs.shape[1])}. Please regenerate state-label cache with auxv2."
			)
		if datax is not None and int(qdot_obs.shape[0]) != int(datax.shape[0]):
			raise ValueError(
				"gphi_include_qobs_dynamics=True qdot_obs batch size does not match datax: "
				f"{int(qdot_obs.shape[0])} vs {int(datax.shape[0])}."
			)
		if q_obs is not None:
			qdot_obs = qdot_obs.to(device=q_obs.device, dtype=q_obs.dtype)
		elif datax is not None:
			qdot_obs = qdot_obs.to(device=datax.device, dtype=datax.dtype)
		return qdot_obs

	def build_observation(
			self,
			datax: Optional[torch.Tensor] = None,
			q_ego: Optional[torch.Tensor] = None,
			q_obs: Optional[torch.Tensor] = None,
			aux: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		return self.obs_builder.build(datax=datax, q_ego=q_ego, q_obs=q_obs)

	def _raylink_points_to_controller_x(self, q_ego: torch.Tensor, obs_flat: torch.Tensor) -> torch.Tensor:
		if self.g_phi is None:
			raise ValueError("RayLink point conversion requires g_phi to be loaded.")
		if bool(self.dynamics_model.add_normal) or int(self.dynamics_model.point_dims) != 3:
			raise ValueError("RayLink CBF path is point-only and requires point_dims=3.")
		bs = int(q_ego.shape[0])
		points_w = obs_flat.reshape(bs, -1, 3)
		link_ids = [int(x) for x in self.dynamics_model.list_sensor]
		T_B_L = self.g_phi.fk(q_ego, link_ids)
		T_W_L = self.g_phi._world_from_base(self.g_phi.T_W_Bego, T_B_L)
		origins = T_W_L[:, :, :3, 3]
		rotations = T_W_L[:, :, :3, :3]

		local_parts = []
		for idx in range(len(link_ids)):
			if points_w.shape[1] >= self.dynamics_model.ray_per_sensor:
				sampled_index = torch.linspace(
					0,
					points_w.shape[1] - 1,
					steps=self.dynamics_model.ray_per_sensor,
					device=q_ego.device,
				).round().long()
			else:
				sampled_index = torch.arange(
					self.dynamics_model.ray_per_sensor,
					device=q_ego.device,
				).long() % points_w.shape[1]
			raw_points = torch.index_select(points_w, dim=1, index=sampled_index)
			origin = origins[:, idx, :]
			rotation = rotations[:, idx, :, :]
			offset_pos = torch.transpose(
				torch.bmm(torch.transpose(rotation, 1, 2), torch.transpose(raw_points - origin.unsqueeze(1), 1, 2)),
				1,
				2,
			)
			if self.dynamics_model.point_dims == 4 and not self.dynamics_model.include_point_velocity:
				offset_pos = cartesian_to_spherical(offset_pos.reshape(-1, 3)).reshape(bs, -1, 4)
			local_parts.append(offset_pos)
		obs = torch.stack(local_parts, dim=1)
		return torch.cat((q_ego, obs.reshape(bs, -1)), dim=1)

	def compose_h_input(
			self,
			datax: Optional[torch.Tensor],
			q_ego: torch.Tensor,
			obs: torch.Tensor,
			aux: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		if self.cbf_obs_mode == "legacy_oracle":
			if datax is None:
				if aux is None:
					raise ValueError("legacy_oracle compose_h_input requires datax or aux.")
				datax = torch.cat((q_ego, obs, aux), dim=1)
			return self.dynamics_model.datax_to_x(datax)
		if self.cbf_obs_mode in ("gphi", "raylink_oracle"):
			return self._raylink_points_to_controller_x(q_ego, obs)
		raise NotImplementedError(f"Unsupported cbf_obs_mode={self.cbf_obs_mode}.")

	def h_from_state(
			self,
			q_ego: torch.Tensor,
			q_obs: Optional[torch.Tensor],
			datax: Optional[torch.Tensor] = None,
			aux: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		obs = self.build_observation(datax=datax, q_ego=q_ego, q_obs=q_obs, aux=aux)
		x = self.compose_h_input(datax, q_ego, obs, aux=aux)
		state = x[:, :self.dynamics_model.n_dims]
		z = self.encode_observation(x, datax)
		return self.h_nn(torch.cat([state, z], dim=-1))

	def _compute_hdot_auto(self, data_x: torch.Tensor, u: torch.Tensor) -> Optional[torch.Tensor]:
		if not self.use_gphi_chain:
			return None
		q_obs = self.dynamics_model.get_obstacle_q_from_datax(data_x)
		if q_obs is None:
			if self.gphi_include_qobs_dynamics:
				raise ValueError(
					"gphi_include_qobs_dynamics=True requires q_obs in datax/state-label cache, "
					"but q_obs could not be parsed. Please regenerate state-label cache with auxv2."
				)
			return None
		qdot_obs = None
		if self.gphi_include_qobs_dynamics:
			qdot_obs, _, _ = self.dynamics_model.get_obstacle_meta_from_datax(data_x)
			qdot_obs = self._require_qdot_obs(qdot_obs, datax=data_x, q_obs=q_obs)

		hdot_vals = []
		for i in range(data_x.shape[0]):
			base = data_x[i : i + 1].detach()
			qe0 = data_x[i, : self.dynamics_model.n_dims]
			qde = u[i]
			if self.gphi_include_qobs_dynamics:
				qo0 = q_obs[i].detach().clone().requires_grad_(True)
				qdo0 = qdot_obs[i].to(device=qo0.device, dtype=qo0.dtype)

				def H(qe_s: torch.Tensor, qo_s: torch.Tensor) -> torch.Tensor:
					return self.h_from_state(qe_s.unsqueeze(0), qo_s.unsqueeze(0), datax=base).squeeze()

				_, hdot_i = jvp(H, (qe0, qo0), (qde, qdo0), strict=False)
			else:
				qo0 = q_obs[i].detach()

				def H(qe_s: torch.Tensor) -> torch.Tensor:
					return self.h_from_state(qe_s.unsqueeze(0), qo0.unsqueeze(0), datax=base).squeeze()

				_, hdot_i = jvp(H, (qe0,), (qde,), strict=False)
			hdot_vals.append(hdot_i)
		return torch.stack(hdot_vals).reshape(-1, 1)

	def V_with_lie_derivatives(
			self,
			x: torch.Tensor,
			data_jacobian: tuple=(),
			scenarios: Optional[ScenarioList] = None,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
		"""Compute CBF Lie derivatives through H(qe, qo)=h(qe, g_phi(qe, qo))."""
		if not self.use_gphi_chain:
			return super().V_with_lie_derivatives(x, data_jacobian=data_jacobian, scenarios=scenarios)
		q_obs = self.dynamics_model.get_obstacle_q_from_datax(x)
		if q_obs is None:
			if self.gphi_include_qobs_dynamics:
				raise ValueError(
					"gphi_include_qobs_dynamics=True requires q_obs in datax/state-label cache, "
					"but q_obs could not be parsed. Please regenerate state-label cache with auxv2."
				)
			return super().V_with_lie_derivatives(x, data_jacobian=data_jacobian, scenarios=scenarios)
		qdot_obs = None
		if self.gphi_include_qobs_dynamics:
			qdot_obs, _, _ = self.dynamics_model.get_obstacle_meta_from_datax(x)
			qdot_obs = self._require_qdot_obs(qdot_obs, datax=x, q_obs=q_obs)

		t0 = time.time()
		if scenarios is None:
			scenarios = self.scenarios
		n_scenarios = len(scenarios)
		bs = x.shape[0]
		V_vals = []
		Lf_vals = []
		Lg_vals = []

		if next(self.g_phi.parameters()).device != x.device:
			self.g_phi = self.g_phi.to(x.device)

		with torch.enable_grad():
			for i in range(bs):
				base = x[i].detach()
				qe0 = x[i, : self.dynamics_model.n_dims].detach().clone().requires_grad_(True)
				if self.gphi_include_qobs_dynamics:
					qo0 = q_obs[i].detach().clone().requires_grad_(True)
					qdo0 = qdot_obs[i].to(device=qo0.device, dtype=qo0.dtype)

					def H(qe_s: torch.Tensor, qo_s: torch.Tensor) -> torch.Tensor:
						return self.h_from_state(qe_s.unsqueeze(0), qo_s.unsqueeze(0), datax=base.unsqueeze(0)).squeeze()

					h_i = H(qe0, qo0)
					grad_qe, grad_qo = torch.autograd.grad(
						h_i,
						(qe0, qo0),
						create_graph=bool(self.training),
						retain_graph=bool(self.training),
						allow_unused=False,
					)
					dynamic_lf = torch.sum(grad_qo * qdo0)
				else:
					qo0 = q_obs[i].detach().clone()

					def H(qe_s: torch.Tensor) -> torch.Tensor:
						return self.h_from_state(qe_s.unsqueeze(0), qo0.unsqueeze(0), datax=base.unsqueeze(0)).squeeze()

					h_i = H(qe0)
					grad_qe = torch.autograd.grad(
						h_i,
						qe0,
						create_graph=bool(self.training),
						retain_graph=bool(self.training),
						allow_unused=False,
					)[0]
					dynamic_lf = torch.zeros((), device=x.device, dtype=x.dtype)
				V_vals.append(h_i)
				Lf_vals.append(dynamic_lf.reshape(1))
				Lg_vals.append(grad_qe)

		V = torch.stack(V_vals, dim=0).reshape(bs)
		Lf_base = torch.stack(Lf_vals, dim=0).reshape(bs, 1, 1)
		Lg_base = torch.stack(Lg_vals, dim=0).reshape(bs, 1, self.dynamics_model.n_controls)
		Lf_V = Lf_base.expand(-1, n_scenarios, -1)
		Lg_V = Lg_base.expand(-1, n_scenarios, -1)

		return V, Lf_V, Lg_V, {"V_w_Jacobian": time.time() - t0, "lie_derivative": 0.0}

	def derivative_diagnostics(self, data_x: torch.Tensor, u: torch.Tensor, fd_eps: float = 1e-3):
		return None

	# @torch.autocast('cuda' if torch.cuda.is_available() else 'cpu')
	def h(self, datax: torch.Tensor):
		"""Return the CBF value for the observations o."""
		q_ego, q_obs, _, aux = self.parse_state_from_datax(datax)
		obs = self.build_observation(datax=datax, q_ego=q_ego, q_obs=q_obs, aux=aux)
		x = self.compose_h_input(datax, q_ego, obs, aux=aux)
		assert x.shape[1] == self.n_dims_extended

		state = x[:, :self.dynamics_model.n_dims]
		z = self.encode_observation(x, datax)
		h = self.h_nn(torch.cat([state, z], dim=-1))

		return h

	def encode_observation(self, x: torch.Tensor, datax: torch.Tensor) -> torch.Tensor:
		"""
		Encode observation into feature vector z.
		z = [encoded_pointcloud, obstacle_qdot] if available.
		x is the state+observation tensor after datax_to_x.
		"""
		observation = x[:, self.dynamics_model.n_dims:]
		if self.ab_mode == "A_no_normal" and getattr(self.dynamics_model, "add_normal", False):
			# Keep network structure unchanged; mask normal channels only.
			bs = observation.shape[0]
			num_sensor = len(self.dynamics_model.list_sensor)
			ray_per_sensor = self.dynamics_model.ray_per_sensor
			point_dims = self.dynamics_model.point_dims
			obs_view = observation.reshape(bs, num_sensor, ray_per_sensor, point_dims).clone()
			# Normal channels are the last 3 dims in ArmLidar when add_normal=True.
			obs_view[..., -3:] = 0.0
			observation = obs_view.reshape(bs, -1)
		feature = self.pc_head(observation)
		encoded_obs = self.encoder(feature)
		if self.obstacle_qdot_dim > 0:
			if datax is None:
				if self.gphi_include_qobs_dynamics:
					raise ValueError(
						"gphi_include_qobs_dynamics=True requires qdot_obs in datax/state-label cache, "
						"but h_from_state/encode_observation was called without datax."
					)
				qdot_obs = torch.zeros(
					(x.shape[0], self.obstacle_qdot_dim),
					device=x.device,
					dtype=x.dtype,
				)
			else:
				qdot_obs, _, _ = self.dynamics_model.get_obstacle_meta_from_datax(datax)
				if self.gphi_include_qobs_dynamics:
					qdot_obs = self._require_qdot_obs(qdot_obs, datax=datax)
			return torch.cat([encoded_obs, qdot_obs], dim=-1)
		return encoded_obs

	def h_with_jacobian(self, datax: torch.Tensor, data_jacobian: tuple) -> Tuple[
		torch.Tensor, torch.Tensor, dict]:
		"""Computes the CLBF value and its Jacobian

		args:
			x: bs x (self.dynamics_model.n_dims + o_dims) the points at which to evaluate the CLBF
		returns:
			V: bs tensor of CLBF values
			JV: bs x 1 x self.dynamics_model.n_dims Jacobian of each row of V wrt x
		"""
		bs = datax.shape[0]
		feature_level = False
		dq_scale = self.dynamics_model.controller_dt/2

		t_dict = {}

		self.pc_head.eval()
		if torch.cuda.is_available():
			torch.cuda.synchronize()
		t0 = time.time()
		# prepare x_prime=x_{t+1}, shape: (bs * q_dims) * x_dim
		with torch.no_grad():
			dq1 = dq_scale * torch.eye(self.dynamics_model.q_dims, device=datax.device).unsqueeze(0).expand(bs, -1, -1)
			dq2 = -dq_scale * torch.eye(self.dynamics_model.q_dims, device=datax.device).unsqueeze(0).expand(bs, -1, -1)
			dqs = torch.cat([dq1, dq2], dim=1)
			assert datax.shape[
					   1] == self.dynamics_model.n_dims + self.dynamics_model.o_dims_in_dataset + self.dynamics_model.state_aux_dims_in_dataset

			if torch.cuda.is_available():
				torch.cuda.synchronize()
			t00 = time.time()
			datax_prime = [self.dynamics_model.batch_lookahead(datax, dqs[:, i, :], data_jacobian) for i in
						   range(dqs.shape[1])]
			datax_prime = torch.cat(datax_prime, dim=1).reshape(-1, datax.shape[1])
		if torch.cuda.is_available():
			torch.cuda.synchronize()
		t_dict['prepare_x_prime_1'] = time.time() - t00
		t_dict['prepare_x_prime'] = time.time() - t0

		if feature_level:  # numerical on feature level and symbolic to cbf value
			raise NotImplementedError('Did not implement feature level for h(data_x)')
			feature = self.pc_head(observation)
			with torch.enable_grad():
				state_for_grad = torch.autograd.Variable(state.data, requires_grad=True)
				feature_for_grad = torch.autograd.Variable(feature.data, requires_grad=True)

				encoded_obs = self.encoder(feature_for_grad)
				h = self.h_nn(torch.cat([state_for_grad, encoded_obs], dim=-1))
				Jh_q = torch.autograd.grad(h.sum(), state_for_grad, create_graph=True, retain_graph=True)[0].unsqueeze(
					1)
				ph_pf = torch.autograd.grad(h.sum(), feature_for_grad, create_graph=True, retain_graph=True)[
					0].unsqueeze(1)
			with torch.no_grad():
				dfdq = torch.zeros((bs, feature.shape[1], self.dynamics_model.q_dims)).type_as(x)
				# x_prime[:, self.dynamics_model.n_dims:] += torch.Tensor(x_prime.shape[0], x_prime.shape[1]-self.dynamics_model.n_dims).uniform_(-1e-5, 1e-5).type_as(x_prime)
				feature_prime = self.pc_head(x_prime[:, self.dynamics_model.n_dims:]).reshape(bs, -1, feature.shape[-1])
				for dim in range(self.dynamics_model.q_dims):
					dfdq[:, :, dim] = (feature_prime[:, dim, :] - feature[:, :]) / dqs[dim][dim]
			J = Jh_q + torch.bmm(ph_pf, dfdq.type_as(x))
		else:  # pure numerical estimation
			with torch.no_grad():
				if torch.cuda.is_available():
					torch.cuda.synchronize()
				t1 = time.time()
				all_h = self.h(torch.cat((datax, datax_prime), dim=0))
				if torch.cuda.is_available():
					torch.cuda.synchronize()
				t2 = time.time()
				h = all_h[:bs]
				h_prime = all_h[bs:].reshape(bs, 1, -1)
				J = (h_prime[:, :, :self.dynamics_model.q_dims] - h.unsqueeze(1)) / (dq_scale * 2)

			if torch.cuda.is_available():
				torch.cuda.synchronize()
			t3 = time.time()
			t_dict['single_prime_forward'] = t2 - t1
			t_dict['jacobian'] = t3 - t2

		if self.h_nn.training:
			self.pc_head.train()

		return h, J, t_dict

	def descent_loss(
			self,
			data_x: torch.Tensor,
			goal_mask: torch.Tensor,
			safe_mask: torch.Tensor,
			unsafe_mask: torch.Tensor,
			boundary_mask: torch.Tensor,
			data_jacobian: Tuple[torch.Tensor, torch.Tensor],
			accuracy: bool = False,
			requires_grad: bool = False,
	) -> List[Tuple[str, torch.Tensor]]:
		"""
		Evaluate the loss on the CBF due to the descent condition

		args:
			x: the points at which to evaluate the loss,
			goal_mask: the points in x marked as part of the goal
			safe_mask: the points in x marked safe
			unsafe_mask: the points in x marked unsafe
			accuracy: if True, return the accuracy (from 0 to 1) as well as the losses
		returns:
			loss: a list of tuples containing ("category_name", loss_value).
		"""
		# Compute loss to encourage satisfaction of the following conditions...
		loss = []

		bs = safe_mask.shape[0]
		ul, ll = self.dynamics_model.control_limits
		upper_limit = ul.unsqueeze(0).expand(bs, -1).type_as(data_x)
		lower_limit = ll.unsqueeze(0).expand(bs, -1).type_as(data_x)

		qp_coef = self.loss_config["descent_violation_weight"]
		epsilon = float(self.loss_config.get("epsilon", self.loss_config.get("eps", 0.0)))
		# qp_coef = min(max(0, (self.current_epoch-self.learn_shape_epochs)/50), 1) * self.loss_config["descent_violation_weight"]

		if self.use_neural_actor:
			u_goal_reaching = torch.lerp(lower_limit, upper_limit,
										 torch.Tensor(*upper_limit.shape).uniform_(0, 1).type_as(data_x))

			h = self.h(data_x)
			u, u_residual = self.u(data_x, u_goal_reaching)

			hdot_auto = self._compute_hdot_auto(data_x, u)
			if (hdot_auto is not None) and (not self.train_use_fd):
				hdot_simulated = hdot_auto.detach()
			else:
				datax_next = self.dynamics_model.batch_lookahead(data_x, u * self.dynamics_model.dt,
																 data_jacobian=data_jacobian)
				hdot_simulated = (self.h(datax_next) - h) / self.dynamics_model.dt
			hdot = hdot_auto if hdot_auto is not None else hdot_simulated
			alpha = self.clf_lambda
			qp_relaxation = F.relu(epsilon + hdot + alpha * h)

			# Minimize the qp relaxation to encourage satisfying the decrease condition
			qp_relaxation_loss = qp_relaxation.mean() * qp_coef
			loss.append(("QP relaxation", qp_relaxation_loss))
			loss.append(("residual", torch.norm(u_residual, p=2, dim=1).mean() * self.loss_config["actor_weight"]))
		else:
			_, Lf_V, Lg_V, _ = self.V_with_lie_derivatives(data_x, data_jacobian)

			Lg_V_no_grad = Lg_V.detach().clone().squeeze(1)  # bs * n_control

			h = self.h(data_x)
			u = torch.where(Lg_V_no_grad >= 0, lower_limit, upper_limit)

			# h_dot = Lf_h + Lg_h @ u; CBF condition: Lf_h + Lg_h @ u + alpha(h) >= 0.
			hdot_expected = (Lf_V.squeeze(1).squeeze(1) + torch.bmm(Lg_V, u.unsqueeze(2)).squeeze(1).squeeze(
				1)).unsqueeze(1)
			if self.train_use_fd:
				datax_next = self.dynamics_model.batch_lookahead(data_x, u * self.dynamics_model.controller_dt,
																 data_jacobian=data_jacobian)
				hdot_simulated = (self.h(datax_next) - h) / self.dynamics_model.controller_dt
				hdot = hdot_simulated
			else:
				hdot = hdot_expected
			alpha = self.clf_lambda  # torch.where(h < 0, 2 * self.clf_lambda, self.clf_lambda).type_as(x)
			qp_relaxation = F.relu(epsilon + hdot + alpha * h)

			# Minimize the qp relaxation to encourage satisfying the decrease condition
			qp_relaxation_loss = qp_relaxation.mean() * qp_coef / alpha
			loss.append(("QP relaxation", qp_relaxation_loss))

			divergence_weight = float(self.loss_config.get("hdot_divergence_weight", 0.0))
			if divergence_weight > 0.0 and self.train_use_fd:
				loss.append(("hdot divergence", divergence_weight * torch.abs(hdot_simulated - hdot_expected).mean()))

		if accuracy:
			def _zero_relaxation_rate(mask):
				vals = qp_relaxation[mask]
				if vals.nelement() == 0:
					return torch.zeros((), dtype=qp_relaxation.dtype, device=qp_relaxation.device)
				return (vals <= 1e-6).sum() / vals.nelement()

			qp_acc_safe = _zero_relaxation_rate(safe_mask)
			qp_acc_unsafe = _zero_relaxation_rate(unsafe_mask)
			qp_acc_boundary = _zero_relaxation_rate(boundary_mask)
			loss.append(("boundary condition accuracy/safe", qp_acc_safe))
			loss.append(("boundary condition accuracy/unsafe", qp_acc_unsafe))
			loss.append(("boundary condition accuracy/boundary", qp_acc_boundary))

		return loss
