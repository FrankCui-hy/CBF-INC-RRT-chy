# Differentiable Observation Surrogate + Neural CBF (loss/)

This module provides a runnable PyTorch>=2.0 pipeline:

1. Collect episode data (`q_ego, q_obs, qdot_ego, qdot_obs, p_gt, n_gt, m, ray geometry, y`).
2. Train surrogate observation network `g_phi`.
3. Train neural CBF `h_theta` using analytic `hdot` with `torch.func.jvp`/`vjp`.
4. Run online controller demo to build CBF linear constraint `A u >= b`.

See `loss/configs/config.yaml` for all hyper-parameters.
