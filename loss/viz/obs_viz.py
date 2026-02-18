from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np



def _save(fig: plt.Figure, path_png: Path, save_pdf: bool = False) -> None:
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_png, dpi=180)
    if save_pdf:
        fig.savefig(path_png.with_suffix(".pdf"))
    plt.close(fig)


def plot_summary_metrics(summary: dict[str, float], out_dir: Path, save_pdf: bool = False) -> None:
    keys = [
        "pos_rmse_hit",
        "pos_mae_hit",
        "normal_angle_mean_deg_hit",
        "normal_cos_mean_hit",
    ]
    vals = [float(summary.get(k, np.nan)) for k in keys]

    if "hit_acc" in summary:
        keys += ["hit_acc", "hit_precision", "hit_recall"]
        vals += [float(summary.get("hit_acc", np.nan)), float(summary.get("hit_precision", np.nan)), float(summary.get("hit_recall", np.nan))]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(keys)), vals)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_title("Observation Fit Summary Metrics")
    ax.grid(True, alpha=0.25)
    _save(fig, out_dir / "summary_metrics.png", save_pdf=save_pdf)


def plot_per_ray(metric_vals: Iterable[float], title: str, ylabel: str, out_path: Path, save_pdf: bool = False) -> None:
    vals = np.asarray(list(metric_vals), dtype=np.float32)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(vals.shape[0]), vals, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("ray index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    _save(fig, out_path, save_pdf=save_pdf)


def plot_hist(values: np.ndarray, title: str, xlabel: str, out_path: Path, bins: int = 50, save_pdf: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if values.size > 0:
        ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    _save(fig, out_path, save_pdf=save_pdf)


def plot_scatter_hit_points(
    p_gt: np.ndarray,
    p_pred: np.ndarray,
    m: np.ndarray,
    out_path: Path,
    max_points: int = 5000,
    save_pdf: bool = False,
) -> None:
    """2D projected scatter comparison for hit points."""
    assert p_gt.shape == p_pred.shape
    hit = m.reshape(-1) > 0.5
    gt = p_gt.reshape(-1, 3)[hit]
    pr = p_pred.reshape(-1, 3)[hit]

    if gt.shape[0] == 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "No hit points in selected subset", ha="center", va="center")
        ax.set_axis_off()
        _save(fig, out_path, save_pdf=save_pdf)
        return

    if gt.shape[0] > max_points:
        idx = np.random.choice(gt.shape[0], size=max_points, replace=False)
        gt = gt[idx]
        pr = pr[idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(gt[:, 0], gt[:, 1], s=5, alpha=0.5, label="gt")
    axes[0].scatter(pr[:, 0], pr[:, 1], s=5, alpha=0.5, label="pred")
    axes[0].set_title("XY projection")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(gt[:, 0], gt[:, 2], s=5, alpha=0.5, label="gt")
    axes[1].scatter(pr[:, 0], pr[:, 2], s=5, alpha=0.5, label="pred")
    axes[1].set_title("XZ projection")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)

    _save(fig, out_path, save_pdf=save_pdf)


def plot_sweep_curve(
    q_values: np.ndarray,
    rmse_vals: np.ndarray,
    angle_vals: np.ndarray,
    hit_counts: np.ndarray | None,
    joint_idx: int,
    out_path: Path,
    save_pdf: bool = False,
) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 4))
    rmse = np.asarray(rmse_vals, dtype=np.float32)
    ang = np.asarray(angle_vals, dtype=np.float32)
    qv = np.asarray(q_values, dtype=np.float32)

    rmse_mask = np.isfinite(rmse)
    ang_mask = np.isfinite(ang)

    ax1.plot(
        qv[rmse_mask],
        rmse[rmse_mask],
        label="RMSE(hit)",
        linewidth=1.6,
        color="tab:blue",
        linestyle="-",
        marker="o",
        markersize=3.0,
        markevery=max(1, max(int(rmse_mask.sum()), 1) // 20),
    )
    ax1.set_xlabel(f"q_ego[{joint_idx}]")
    ax1.set_ylabel("RMSE")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(
        qv[ang_mask],
        ang[ang_mask],
        label="Angle Mean(hit)",
        linewidth=1.6,
        color="tab:orange",
        linestyle="--",
        marker="s",
        markersize=3.0,
        markevery=max(1, max(int(ang_mask.sum()), 1) // 20),
    )
    ax2.set_ylabel("Angle (deg)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines = lines1 + lines2
    labels = labels1 + labels2

    if hit_counts is not None:
        hc = np.asarray(hit_counts, dtype=np.float32)
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 48))
        ax3.plot(
            qv,
            hc,
            label="Hit Count",
            linewidth=1.2,
            color="tab:green",
            linestyle=":",
            alpha=0.85,
        )
        ax3.set_ylabel("Hit Count")
        l3, lb3 = ax3.get_legend_handles_labels()
        lines += l3
        labels += lb3

    ax1.legend(lines, labels, loc="best")
    ax1.set_title("Sweep over q_ego joint")
    _save(fig, out_path, save_pdf=save_pdf)


def plot_sweep_hit_count(
    q_values: np.ndarray,
    hit_counts: np.ndarray,
    joint_idx: int,
    out_path: Path,
    save_pdf: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        q_values,
        hit_counts,
        linewidth=1.5,
        color="tab:green",
        marker="^",
        markersize=3.0,
        markevery=max(1, len(q_values) // 20),
    )
    ax.set_xlabel(f"q_ego[{joint_idx}]")
    ax.set_ylabel("Hit Count")
    ax.set_title("Sweep Hit Count")
    ax.grid(True, alpha=0.25)
    _save(fig, out_path, save_pdf=save_pdf)
