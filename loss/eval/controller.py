from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.func import vjp

from loss.models.g_phi import SurrogateObservationNet
from loss.models.h_theta import NeuralCBF
from loss.utils.config import load_config


@dataclass
class ConstraintResult:
    h: torch.Tensor  # scalar
    A: torch.Tensor  # (1, n_ego)
    b: torch.Tensor  # (1,)


def clamp_u(u: torch.Tensor, u_min: torch.Tensor, u_max: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(u, u_max), u_min)


def project_to_halfspace(u_nom: torch.Tensor, A: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Solve min ||u-u_nom||^2 s.t. A u >= b for single linear constraint.

    Closed-form projection on half-space.
    """
    lhs = (A @ u_nom).squeeze(0)
    if lhs >= b:
        return u_nom
    denom = (A * A).sum() + eps
    step = (b - lhs) / denom
    return u_nom + step * A.squeeze(0)


class CBFController:
    """Online controller wrapper using g_phi + h_theta with VJP-derived linear CBF constraint."""

    def __init__(self, g_phi: SurrogateObservationNet, h_theta: NeuralCBF, alpha: float) -> None:
        self.g_phi = g_phi
        self.h_theta = h_theta
        self.alpha = alpha

    @torch.enable_grad()
    def compute_constraint(self, qe: torch.Tensor, qo: torch.Tensor, qdot_o: torch.Tensor) -> ConstraintResult:
        """Compute A,b for constraint A*u >= b where u = qdot_ego.

        H(qe, qo) = h_theta(qe, g_phi(qe, qo)).
        Then grad_qe, grad_qo from one VJP call with cotangent=1.
        """

        def H(qe_s: torch.Tensor, qo_s: torch.Tensor) -> torch.Tensor:
            o_hat = self.g_phi(qe_s.unsqueeze(0), qo_s.unsqueeze(0))["o_hat"]
            return self.h_theta(qe_s.unsqueeze(0), o_hat).squeeze(0)

        h, vjp_fn = vjp(H, qe, qo)
        grad_qe, grad_qo = vjp_fn(torch.ones((), device=qe.device, dtype=qe.dtype))

        A = grad_qe.unsqueeze(0)
        b = -(grad_qo * qdot_o).sum().unsqueeze(0) - self.alpha * h.unsqueeze(0)
        return ConstraintResult(h=h, A=A, b=b)

    def step(
        self,
        qe: torch.Tensor,
        qo: torch.Tensor,
        qdot_o: torch.Tensor,
        u_nom: torch.Tensor,
        u_min: torch.Tensor,
        u_max: torch.Tensor,
    ) -> Tuple[torch.Tensor, ConstraintResult, bool]:
        c = self.compute_constraint(qe, qo, qdot_o)
        u_qp = project_to_halfspace(u_nom, c.A, c.b)
        u_qp = clamp_u(u_qp, u_min, u_max)
        satisfied = ((c.A @ u_qp).squeeze(0) >= c.b.squeeze(0)).item()
        return u_qp, c, bool(satisfied)


def build_models(cfg, device: torch.device):
    sys_cfg = cfg["system"]

    gcfg = cfg["model"]["g_phi"]
    g_phi = SurrogateObservationNet(
        n_ego=int(sys_cfg["n_ego"]),
        n_obs=int(sys_cfg["n_obs"]),
        rays=int(sys_cfg["rays"]),
        hidden_dims=list(gcfg["hidden_dims"]),
        activation=str(gcfg.get("activation", "silu")),
        predict_hit_prob=bool(gcfg.get("predict_hit_prob", True)),
        norm_eps=float(gcfg.get("norm_eps", 1e-6)),
    ).to(device)

    hcfg = cfg["model"]["h_theta"]
    obs_dim_per_ray = 7 if bool(hcfg.get("include_hit_prob", True)) else 6
    h_theta = NeuralCBF(
        n_ego=int(sys_cfg["n_ego"]),
        rays=int(sys_cfg["rays"]),
        obs_dim_per_ray=obs_dim_per_ray,
        encoder=str(hcfg["encoder"]),
        point_feat_dim=int(hcfg["point_feat_dim"]),
        global_feat_dim=int(hcfg["global_feat_dim"]),
        hidden_dims=list(hcfg["hidden_dims"]),
        activation=str(hcfg.get("activation", "silu")),
    ).to(device)

    g_ckpt = Path(cfg["paths"]["g_phi_ckpt"])
    h_ckpt = Path(cfg["paths"]["h_theta_ckpt"])
    if g_ckpt.exists():
        g_phi.load_state_dict(torch.load(g_ckpt, map_location="cpu")["model"])
    if h_ckpt.exists():
        h_theta.load_state_dict(torch.load(h_ckpt, map_location="cpu")["h_model"])

    g_phi.eval()
    h_theta.eval()
    return g_phi, h_theta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo online CBF controller with A*u>=b from VJP/JVP-friendly derivatives.")
    p.add_argument("--config", type=str, default="loss/configs/config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() and str(cfg.get("device", "cuda")).startswith("cuda") else "cpu")

    g_phi, h_theta = build_models(cfg, device)
    ctrl = CBFController(g_phi, h_theta, alpha=float(cfg["eval"]["qp"]["alpha"]))

    n_ego = int(cfg["system"]["n_ego"])
    n_obs = int(cfg["system"]["n_obs"])
    qe = torch.randn(n_ego, device=device) * 0.2
    qo = torch.randn(n_obs, device=device) * 0.2
    qdot_o = torch.randn(n_obs, device=device) * 0.3
    u_nom = torch.randn(n_ego, device=device) * 0.5

    u_min = torch.tensor(cfg["eval"]["qp"]["u_min"], dtype=torch.float32, device=device)
    u_max = torch.tensor(cfg["eval"]["qp"]["u_max"], dtype=torch.float32, device=device)

    u_qp, c, ok = ctrl.step(qe, qo, qdot_o, u_nom, u_min, u_max)
    lhs_nom = (c.A @ u_nom).item()
    lhs_qp = (c.A @ u_qp).item()

    print(f"[controller] h={c.h.item():.6f}")
    print(f"[controller] A shape={tuple(c.A.shape)}, b={c.b.item():.6f}")
    print(f"[controller] nominal lhs={lhs_nom:.6f}, constraint={c.b.item():.6f}")
    print(f"[controller] qp lhs={lhs_qp:.6f}, constraint_satisfied={ok}")
    print(f"[controller] ||u_nom||={u_nom.norm().item():.4f}, ||u_qp||={u_qp.norm().item():.4f}")


if __name__ == "__main__":
    main()
