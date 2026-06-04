# FK-aware Ray-Link Observation Surrogate (loss/)

This module keeps the current g_phi pipeline only:

1. Generate all-rays-at-once Panda/Panda raycast data with `scripts/generate_neural_raycast_dataset.py`.
2. Train the FK-aware ray-link surrogate `RayLinkMLPGPhi` with `python -m loss.training.train_raylink_g_phi`.
3. Evaluate the trained checkpoint with `python -m loss.eval.eval_raylink_g_phi`.

See `loss/configs/config_raylink_g_phi.yaml` for hyper-parameters.
