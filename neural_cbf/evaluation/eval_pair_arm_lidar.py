from __future__ import annotations

import argparse
import json
import os

import yaml

from neural_cbf.evaluation.eval_arm_lidar import (
    controller_method_metadata,
    eval_metrics_offline,
    init_val,
)


def _resolve_hparams_path(ckpt_path: str, explicit_path: str | None) -> str:
    if explicit_path:
        return explicit_path
    candidate = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), "hparams.yaml")
    if os.path.exists(candidate):
        return candidate
    candidate = os.path.join(os.path.dirname(ckpt_path), "hparams.yaml")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"hparams.yaml not found for checkpoint: {ckpt_path}")


def _load_args(ckpt_path: str, hparams_path: str | None, cli_args: argparse.Namespace) -> argparse.Namespace:
    resolved = _resolve_hparams_path(ckpt_path, hparams_path)
    with open(resolved, "r") as f:
        args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))
    args.accelerator = "cpu"
    args.gui = 0
    if cli_args.robot_name is not None:
        args.robot_name = cli_args.robot_name
    if cli_args.obstacle_horizon_s is not None:
        args.obstacle_horizon_s = float(cli_args.obstacle_horizon_s)
    return args


def _set_expected(args: argparse.Namespace, key: str, expected, label: str, force: bool):
    cur = getattr(args, key, None)
    if force or cur is None:
        setattr(args, key, expected)
        return
    if cur != expected:
        raise ValueError(
            f"{label} checkpoint has {key}={cur!r}, expected {expected!r}. "
            "Use --force_method_flags only when intentionally overriding old/incomplete hparams."
        )


def _enforce_method(
    args: argparse.Namespace,
    label: str,
    force: bool,
    gphi_ckpt_override: str | None = None,
) -> argparse.Namespace:
    if label == "baseline":
        _set_expected(args, "baseline", True, label, force)
        _set_expected(args, "obs_backend", "raw", label, force)
        _set_expected(args, "train_use_fd", True, label, force)
        if force and not getattr(args, "gphi_ckpt", None):
            args.gphi_ckpt = ""
        return args

    if label == "ours":
        _set_expected(args, "baseline", False, label, force)
        _set_expected(args, "obs_backend", "gphi", label, force)
        _set_expected(args, "train_use_fd", False, label, force)
        if gphi_ckpt_override:
            args.gphi_ckpt = gphi_ckpt_override
        if not getattr(args, "gphi_ckpt", None):
            raise ValueError("ours checkpoint requires gphi_ckpt in hparams or --ours_gphi_ckpt.")
        return args

    raise ValueError(f"Unknown method label: {label}")


def _comparison(baseline: dict, ours: dict) -> dict:
    keys = [
        "relax_mean",
        "relax_p95",
        "relax_zero_rate",
        "relax_auto_mean",
        "relax_auto_p95",
        "relax_auto_zero_rate",
        "infeasible_rate",
    ]
    out = {}
    for key in keys:
        b = baseline.get(key)
        o = ours.get(key)
        if isinstance(b, (int, float)) and isinstance(o, (int, float)):
            out[f"{key}_ours_minus_baseline"] = float(o) - float(b)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired offline metrics for baseline vs Safe_Dual on the same eval settings.")
    p.add_argument("--baseline_ckpt", required=True)
    p.add_argument("--ours_ckpt", required=True)
    p.add_argument("--baseline_hparams", default=None)
    p.add_argument("--ours_hparams", default=None)
    p.add_argument("--ours_gphi_ckpt", default=None)
    p.add_argument("--force_method_flags", action="store_true")
    p.add_argument("--robot_name", default=None)
    p.add_argument("--obstacle_horizon_s", type=float, default=None)
    p.add_argument("--num_samples", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--u_clamp", type=float, default=2.5)
    p.add_argument("--near_ratio", type=float, default=0.0)
    p.add_argument("--near_mode", default="boundary_or_unsafe", choices=["boundary_or_unsafe", "unsafe_only", "boundary_only"])
    p.add_argument("--fd_eps_list", default="1e-2,5e-3,1e-3")
    p.add_argument("--fd_obs_source", default="model", choices=["model", "raw"])
    p.add_argument("--out", default="paired_metrics.json")
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    baseline_args = _load_args(cli.baseline_ckpt, cli.baseline_hparams, cli)
    ours_args = _load_args(cli.ours_ckpt, cli.ours_hparams, cli)
    baseline_args = _enforce_method(baseline_args, "baseline", bool(cli.force_method_flags))
    ours_args = _enforce_method(ours_args, "ours", bool(cli.force_method_flags), cli.ours_gphi_ckpt)

    baseline_controller = init_val(cli.baseline_ckpt, baseline_args)
    ours_controller = init_val(cli.ours_ckpt, ours_args)

    metric_kwargs = dict(
        num_samples=cli.num_samples,
        batch_size=cli.batch_size,
        seed=cli.seed,
        alpha=cli.alpha,
        u_clamp=cli.u_clamp,
        near_ratio=cli.near_ratio,
        near_mode=cli.near_mode,
        fd_eps_list=cli.fd_eps_list,
        fd_obs_source=cli.fd_obs_source,
    )
    baseline_metrics = eval_metrics_offline(baseline_controller, **metric_kwargs)
    ours_metrics = eval_metrics_offline(ours_controller, **metric_kwargs)
    baseline_metrics["method_metadata"] = controller_method_metadata(baseline_controller, cli.baseline_ckpt)
    ours_metrics["method_metadata"] = controller_method_metadata(ours_controller, cli.ours_ckpt)

    payload = {
        "paired_eval": True,
        "seed": int(cli.seed),
        "settings": metric_kwargs,
        "baseline": baseline_metrics,
        "ours": ours_metrics,
        "comparison": _comparison(baseline_metrics, ours_metrics),
    }
    print(json.dumps(payload, indent=2))
    if cli.out:
        out_dir = os.path.dirname(cli.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(cli.out, "w") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
