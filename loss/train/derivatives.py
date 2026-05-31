from __future__ import annotations

from typing import Tuple

import torch
from torch.func import jvp, vjp, vmap


@torch.enable_grad()
def composite_jvp_h_and_hdot(
    g_phi,
    h_theta,
    q_ego: torch.Tensor,
    q_obs: torch.Tensor,
    qdot_ego: torch.Tensor,
    qdot_obs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute h and hdot via one composite JVP: H(qe,qo)=h_theta(qe,g_phi(qe,qo))."""

    def single(qe: torch.Tensor, qo: torch.Tensor, qde: torch.Tensor, qdo: torch.Tensor):
        def H(qe_s: torch.Tensor, qo_s: torch.Tensor) -> torch.Tensor:
            o_hat = g_phi(qe_s.unsqueeze(0), qo_s.unsqueeze(0))["o_hat"].squeeze(0)
            h_val = h_theta(qe_s.unsqueeze(0), o_hat.unsqueeze(0)).squeeze(0)
            return h_val

        h_val, hdot_val = jvp(H, (qe, qo), (qde, qdo), strict=False)
        return h_val, hdot_val

    h, hdot = vmap(single)(q_ego, q_obs, qdot_ego, qdot_obs)
    return h, hdot


@torch.enable_grad()
def decomposed_h_and_hdot(
    g_phi,
    h_theta,
    q_ego: torch.Tensor,
    q_obs: torch.Tensor,
    qdot_ego: torch.Tensor,
    qdot_obs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute h and hdot with explicit decomposition.

    hdot = <∂h/∂q_e, qdot_e> + <∂h/∂o, odot>
    odot via two JVPs on g_phi output flatten.
    """

    B = q_ego.shape[0]
    o_hat = g_phi(q_ego, q_obs)["o_hat"]
    h = h_theta(q_ego, o_hat)

    def g_flat_single(qe: torch.Tensor, qo: torch.Tensor) -> torch.Tensor:
        return g_phi(qe.unsqueeze(0), qo.unsqueeze(0))["o_hat"].reshape(-1)

    def odot_single(qe: torch.Tensor, qo: torch.Tensor, qde: torch.Tensor, qdo: torch.Tensor) -> torch.Tensor:
        _, j_qe = jvp(g_flat_single, (qe, qo), (qde, torch.zeros_like(qo)), strict=False)
        _, j_qo = jvp(g_flat_single, (qe, qo), (torch.zeros_like(qe), qdo), strict=False)
        return j_qe + j_qo

    odot_flat = vmap(odot_single)(q_ego, q_obs, qdot_ego, qdot_obs)
    odot = odot_flat.view_as(o_hat)

    def hdot_single(qe: torch.Tensor, o: torch.Tensor, qde: torch.Tensor, od: torch.Tensor) -> torch.Tensor:
        def h_fn(qe_s: torch.Tensor, o_s: torch.Tensor) -> torch.Tensor:
            return h_theta(qe_s.unsqueeze(0), o_s.unsqueeze(0)).squeeze(0)

        _, vjp_fn = vjp(h_fn, qe, o)
        grad_qe, grad_o = vjp_fn(torch.ones((), device=qe.device, dtype=qe.dtype))
        return (grad_qe * qde).sum() + (grad_o * od).sum()

    hdot = vmap(hdot_single)(q_ego, o_hat, qdot_ego, odot)
    return h, hdot


@torch.enable_grad()
def composite_h_and_min_cbf_derivative(
    g_phi,
    h_theta,
    q_ego: torch.Tensor,
    q_obs: torch.Tensor,
    qdot_obs: torch.Tensor,
    u_min: torch.Tensor,
    u_max: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute h and min_u hdot for H(qe, qo)=h_theta(qe, g_phi(qe, qo)).

    The manipulator model used in this repo is qdot = u, so the paper's
    infimum over the box-constrained control set is the closed-form minimum
    of a linear function grad_qe(H) @ u over [u_min, u_max].
    """

    u_min = u_min.to(device=q_ego.device, dtype=q_ego.dtype)
    u_max = u_max.to(device=q_ego.device, dtype=q_ego.dtype)

    def single(qe: torch.Tensor, qo: torch.Tensor, qdo: torch.Tensor):
        def H(qe_s: torch.Tensor, qo_s: torch.Tensor) -> torch.Tensor:
            o_hat = g_phi(qe_s.unsqueeze(0), qo_s.unsqueeze(0))["o_hat"].squeeze(0)
            return h_theta(qe_s.unsqueeze(0), o_hat.unsqueeze(0)).squeeze(0)

        h_val, vjp_fn = vjp(H, qe, qo)
        grad_qe, grad_qo = vjp_fn(torch.ones((), device=qe.device, dtype=qe.dtype))
        u_star = torch.where(grad_qe >= 0, u_min, u_max)
        hdot_min = (grad_qe * u_star).sum() + (grad_qo * qdo).sum()
        return h_val, hdot_min

    h, hdot_min = vmap(single)(q_ego, q_obs, qdot_obs)
    return h, hdot_min
