import itertools
import time
import sys
from functools import partial
from typing import Tuple, List, Optional
from collections import OrderedDict
import random
import tqdm
from pathlib import Path

import pybullet as p

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.func import jvp

import matplotlib.pyplot as plt

from neural_cbf.systems import ArmLidar
from neural_cbf.systems.utils import ScenarioList
from neural_cbf.controllers import NeuralObsCBFController
from neural_cbf.controllers.utils import PointNetfeat, PointNetVanillaEncoder
from neural_cbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_cbf.experiments import ExperimentSuite

try:
	from loss.models.g_phi import SurrogateObservationNet
except Exception:
	# Fallback when running scripts directly (e.g. python neural_cbf/training/train_arm_lidar.py),
	# where repo root may not be on sys.path.
	repo_root = str(Path(__file__).resolve().parents[2])
	if repo_root not in sys.path:
		sys.path.insert(0, repo_root)
	try:
		from loss.models.g_phi import SurrogateObservationNet
	except Exception:
		SurrogateObservationNet = None


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
		self.gphi_ckpt = kwargs.get("gphi_ckpt", "")
		self.train_use_fd = bool(kwargs.get("train_use_fd", self.baseline))
		self._validate_method_config()
		self.use_gphi_chain = (not self.baseline) and self.obs_backend == "gphi"
		self.g_phi = None
		if self.use_gphi_chain:
			self._init_g_phi()
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
		if self.baseline:
			if self.obs_backend != "raw":
				raise ValueError("baseline=True requires obs_backend='raw'. Use baseline=False for Safe_Dual gphi-chain.")
			if not self.train_use_fd:
				raise ValueError("baseline=True requires train_use_fd=True so the baseline uses the legacy FD/simulated Lie derivative.")
			return
		if self.obs_backend != "gphi":
			raise ValueError("Safe_Dual method requires baseline=False and obs_backend='gphi'. Use baseline=True for raw/FD baseline.")
		if self.train_use_fd:
			raise ValueError("Safe_Dual method requires train_use_fd=False so the analytic g_phi chain derivative is used.")
		if self.gphi_ckpt is None or str(self.gphi_ckpt).strip() == "":
			raise ValueError("Safe_Dual method requires a non-empty gphi_ckpt.")

	def method_metadata(self) -> dict:
		method = "baseline_fd_raw" if self.baseline else "safe_dual_gphi_chain"
		return {
			"method": method,
			"baseline": bool(self.baseline),
			"obs_backend": str(self.obs_backend),
			"ab_mode": str(self.ab_mode),
			"train_use_fd": bool(self.train_use_fd),
			"use_gphi_chain": bool(self.use_gphi_chain),
			"gphi_ckpt": str(self.gphi_ckpt or ""),
		}

	def _init_g_phi(self):
		if SurrogateObservationNet is None:
			raise RuntimeError("loss.models.g_phi is not available. Cannot use obs_backend=gphi.")
		if self.gphi_ckpt is None or self.gphi_ckpt == "":
			raise ValueError("obs_backend=gphi requires --gphi_ckpt.")
		ckpt = torch.load(self.gphi_ckpt, map_location="cpu")
		model_cfg = ckpt.get("model_cfg", {})
		sys_cfg = ckpt.get("system_cfg", {})
		self.g_phi = SurrogateObservationNet(
			n_ego=int(sys_cfg.get("n_ego", self.dynamics_model.n_dims)),
			n_obs=int(sys_cfg.get("n_obs", self.dynamics_model.n_dims)),
			rays=int(sys_cfg.get("rays", self.dynamics_model.point_in_dataset_pc)),
			hidden_dims=list(model_cfg.get("hidden_dims", [256, 256, 256])),
			activation=str(model_cfg.get("activation", "silu")),
			predict_hit_prob=bool(model_cfg.get("predict_hit_prob", True)),
			norm_eps=float(model_cfg.get("norm_eps", 1e-6)),
		)
		self.g_phi.load_state_dict(ckpt["model"])
		self.g_phi.eval()
		for p in self.g_phi.parameters():
			p.requires_grad = False

	def _reshape_gphi_to_dataset_obs(self, p_hat: torch.Tensor, n_hat: torch.Tensor) -> torch.Tensor:
		"""
		Map g_phi outputs to ArmLidar dataset observation layout:
		(B, point_in_dataset_pc, point_dim_dataset), where point_dim_dataset is 3 or 6.
		"""
		target = int(self.dynamics_model.point_in_dataset_pc)
		B, R, _ = p_hat.shape
		idx = torch.linspace(0, max(R - 1, 0), steps=target, device=p_hat.device).round().long()
		p_sel = p_hat[:, idx, :]
		if self.dynamics_model.add_normal:
			n_sel = n_hat[:, idx, :]
			if self.ab_mode == "A_no_normal":
				n_sel = torch.zeros_like(n_sel)
			return torch.cat([p_sel, n_sel], dim=-1)
		return p_sel

	def _replace_datax_obs_with_gphi(self, datax: torch.Tensor) -> torch.Tensor:
		if not self.use_gphi_chain:
			return datax
		qe = datax[:, : self.dynamics_model.n_dims]
		qo = self.dynamics_model.get_obstacle_q_from_datax(datax)
		if qo is None:
			return datax
		device = datax.device
		if next(self.g_phi.parameters()).device != device:
			self.g_phi = self.g_phi.to(device)
		pred = self.g_phi(qe, qo)
		obs_dataset = self._reshape_gphi_to_dataset_obs(pred["p_hat"], pred["n_hat"]).reshape(datax.shape[0], -1)
		datax_new = datax.clone()
		datax_new[:, self.dynamics_model.n_dims : -self.dynamics_model.state_aux_dims_in_dataset] = obs_dataset
		return datax_new

	def _compute_hdot_auto(self, data_x: torch.Tensor, u: torch.Tensor) -> Optional[torch.Tensor]:
		if not self.use_gphi_chain:
			return None
		q_obs = self.dynamics_model.get_obstacle_q_from_datax(data_x)
		qdot_obs, _, _ = self.dynamics_model.get_obstacle_meta_from_datax(data_x)
		if q_obs is None or qdot_obs is None:
			return None

		hdot_vals = []
		for i in range(data_x.shape[0]):
			base = data_x[i].detach()
			qe0 = data_x[i, : self.dynamics_model.n_dims]
			qo0 = q_obs[i]
			qde = u[i]
			qdo = qdot_obs[i]

			def H(qe_s: torch.Tensor, qo_s: torch.Tensor) -> torch.Tensor:
				row = base.clone()
				row[: self.dynamics_model.n_dims] = qe_s
				# overwrite q_obs in aux (if present)
				meta = row[-self.dynamics_model.state_aux_dims_in_dataset :]
				if self.dynamics_model.obstacle_q_dim > 0:
					q_start = self.dynamics_model.sensor_aux_dims
					meta[q_start : q_start + self.dynamics_model.obstacle_q_dim] = qo_s
				row[-self.dynamics_model.state_aux_dims_in_dataset :] = meta
				return self.h(row.unsqueeze(0)).squeeze()

			_, hdot_i = jvp(H, (qe0, qo0), (qde, qdo), strict=False)
			hdot_vals.append(hdot_i)
		return torch.stack(hdot_vals).reshape(-1, 1)

	def V_with_lie_derivatives(
			self,
			x: torch.Tensor,
			data_jacobian: tuple=(),
			scenarios: Optional[ScenarioList] = None,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
		"""Compute CBF Lie derivatives through H(qe, qo)=h(qe, g_phi(qe, qo)).

		Safe_Dual's dynamic-obstacle CBF uses the chain rule through both ego and
		obstacle joint states. The parent implementation finite-differences only
		the ego state and treats the observation as static; keep it as fallback
		when g_phi metadata is unavailable.
		"""
		if not self.use_gphi_chain:
			return super().V_with_lie_derivatives(x, data_jacobian=data_jacobian, scenarios=scenarios)
		q_obs = self.dynamics_model.get_obstacle_q_from_datax(x)
		qdot_obs, _, _ = self.dynamics_model.get_obstacle_meta_from_datax(x)
		if q_obs is None or qdot_obs is None:
			return super().V_with_lie_derivatives(x, data_jacobian=data_jacobian, scenarios=scenarios)

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
				qo0 = q_obs[i].detach().clone().requires_grad_(True)
				qdo = qdot_obs[i].detach()

				def H(qe_s: torch.Tensor, qo_s: torch.Tensor) -> torch.Tensor:
					row = base.clone()
					row[: self.dynamics_model.n_dims] = qe_s
					meta = row[-self.dynamics_model.state_aux_dims_in_dataset :].clone()
					if self.dynamics_model.obstacle_q_dim > 0:
						q_start = self.dynamics_model.sensor_aux_dims
						meta[q_start : q_start + self.dynamics_model.obstacle_q_dim] = qo_s
					row[-self.dynamics_model.state_aux_dims_in_dataset :] = meta
					return self.h(row.unsqueeze(0)).squeeze()

				h_i = H(qe0, qo0)
				grad_qe, grad_qo = torch.autograd.grad(
					h_i,
					(qe0, qo0),
					create_graph=bool(self.training),
					retain_graph=bool(self.training),
				)
				dynamic_lf = torch.dot(grad_qo, qdo)
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
		"""
		Compute odot_jvp vs odot_fd (p/m/n split) and hdot_auto vs hdot_fd.
		Returns None when gphi chain is disabled or obstacle metadata is unavailable.
		"""
		if not self.use_gphi_chain:
			return None
		q_obs = self.dynamics_model.get_obstacle_q_from_datax(data_x)
		qdot_obs, _, _ = self.dynamics_model.get_obstacle_meta_from_datax(data_x)
		if q_obs is None or qdot_obs is None:
			return None

		qe = data_x[:, : self.dynamics_model.n_dims]
		qde = u
		device = data_x.device
		if next(self.g_phi.parameters()).device != device:
			self.g_phi = self.g_phi.to(device)
		with torch.no_grad():
			m_now = self.g_phi(qe, q_obs).get("m_hat", None)
			if m_now is not None:
				hit_count_pred = int((m_now.squeeze(-1) > 0.5).sum(dim=1).float().mean().item())
			else:
				hit_count_pred = 0

		odot_jvp_list, odot_fd_list = [], []
		for i in range(data_x.shape[0]):
			def gflat(qe_s: torch.Tensor, qo_s: torch.Tensor) -> torch.Tensor:
				out = self.g_phi(qe_s.unsqueeze(0), qo_s.unsqueeze(0))
				p = out["p_hat"].reshape(-1)
				n = out["n_hat"].reshape(-1)
				m = out["m_hat"].reshape(-1) if out.get("m_hat", None) is not None else torch.zeros((out["p_hat"].numel() // 3,), device=qe_s.device, dtype=qe_s.dtype)
				return torch.cat([p, n, m], dim=0)

			_, odot_jvp_i = jvp(gflat, (qe[i], q_obs[i]), (qde[i], qdot_obs[i]), strict=False)
			g0 = gflat(qe[i], q_obs[i])
			g1 = gflat(qe[i] + fd_eps * qde[i], q_obs[i] + fd_eps * qdot_obs[i])
			odot_fd_i = (g1 - g0) / fd_eps
			odot_jvp_list.append(odot_jvp_i)
			odot_fd_list.append(odot_fd_i)

		odot_jvp = torch.stack(odot_jvp_list)
		odot_fd = torch.stack(odot_fd_list)

		# split by channels: [p(3R), n(3R), m(R)]
		R = self.g_phi.rays
		p_end = 3 * R
		n_end = 6 * R
		p_jvp = odot_jvp[:, :p_end]
		n_jvp = odot_jvp[:, p_end:n_end]
		m_jvp = odot_jvp[:, n_end:]
		p_fd = odot_fd[:, :p_end]
		n_fd = odot_fd[:, p_end:n_end]
		m_fd = odot_fd[:, n_end:]
		p_err = (p_jvp - p_fd).abs().mean()
		n_err = (n_jvp - n_fd).abs().mean()
		m_err = (m_jvp - m_fd).abs().mean()

		hdot_auto = self._compute_hdot_auto(data_x, u)
		if hdot_auto is None:
			return None
		datax_next = self.dynamics_model.batch_lookahead(data_x, u * self.dynamics_model.controller_dt, data_jacobian=())
		hdot_fd = (self.h(datax_next) - self.h(data_x)) / self.dynamics_model.controller_dt
		hdot_err = (hdot_auto - hdot_fd).abs().mean()

		return {
			"odot_err_p": float(p_err.detach().cpu()),
			"odot_err_n": float(n_err.detach().cpu()),
			"odot_err_m": float(m_err.detach().cpu()),
			"odot_jvp_p_meanabs": float(p_jvp.abs().mean().detach().cpu()),
			"odot_jvp_n_meanabs": float(n_jvp.abs().mean().detach().cpu()),
			"odot_jvp_m_meanabs": float(m_jvp.abs().mean().detach().cpu()),
			"odot_fd_p_meanabs": float(p_fd.abs().mean().detach().cpu()),
			"odot_fd_n_meanabs": float(n_fd.abs().mean().detach().cpu()),
			"odot_fd_m_meanabs": float(m_fd.abs().mean().detach().cpu()),
			"hdot_auto_mean": float(hdot_auto.mean().detach().cpu()),
			"hdot_fd_mean": float(hdot_fd.mean().detach().cpu()),
			"hdot_err": float(hdot_err.detach().cpu()),
			"hit_count_pred": hit_count_pred,
		}

	# @torch.autocast('cuda' if torch.cuda.is_available() else 'cpu')
	def h(self, datax: torch.Tensor):
		"""Return the CBF value for the observations o

		args:
			x: bs x self.n_dims_extended tensor of state and observation
		returns:
			h: bs x 1 tensor of BF values
		"""
		datax_used = self._replace_datax_obs_with_gphi(datax)
		x = self.dynamics_model.datax_to_x(datax_used)
		bs = x.shape[0]
		assert x.shape[1] == self.n_dims_extended

		state = x[:, :self.dynamics_model.n_dims]
		z = self.encode_observation(x, datax_used)
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
			qdot_obs, _, _ = self.dynamics_model.get_obstacle_meta_from_datax(datax)
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
