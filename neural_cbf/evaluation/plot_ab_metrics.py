import argparse
import json
import os
import glob
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def _load_group(pattern: str) -> List[Dict]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    data = []
    for f in files:
        with open(f, "r") as fh:
            d = json.load(fh)
        d["_file"] = f
        data.append(d)
    return data


def _mean_std(values: List[float]):
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _collect_metric(group: List[Dict], key: str) -> List[float]:
    vals = []
    for d in group:
        if key in d and d[key] is not None:
            vals.append(float(d[key]))
    return vals


def _collect_fd_eps(group: List[Dict]) -> Dict[str, List[float]]:
    out = {}
    for d in group:
        scan = d.get("fd_eps_scan", {})
        for eps, stats in scan.items():
            val = stats.get("mae_vs_hdot_auto", None)
            if val is None:
                continue
            out.setdefault(eps, []).append(float(val))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a_glob", type=str, required=True, help="Glob for A json files, e.g. /tmp/ab/A_seed*.json")
    parser.add_argument("--b_glob", type=str, required=True, help="Glob for B/STD json files")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--a_label", type=str, default="A_explicit")
    parser.add_argument("--b_label", type=str, default="STD_baseline")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    A = _load_group(args.a_glob)
    B = _load_group(args.b_glob)

    core_keys = [
        "relax_mean",
        "relax_p95",
        "relax_zero_rate",
        "hdot_err_mean",
        "odot_err_p_mean",
        "odot_err_n_mean",
        "odot_err_m_mean",
        "hdot_auto_fd_dt_mae",
    ]

    # 1) bar chart (mean +- std)
    names, a_mean, a_std, b_mean, b_std = [], [], [], [], []
    for k in core_keys:
        av = _collect_metric(A, k)
        bv = _collect_metric(B, k)
        if not av or not bv:
            continue
        am, asd = _mean_std(av)
        bm, bsd = _mean_std(bv)
        names.append(k)
        a_mean.append(am)
        a_std.append(asd)
        b_mean.append(bm)
        b_std.append(bsd)

    if names:
        x = np.arange(len(names))
        w = 0.38
        plt.figure(figsize=(12, 5))
        plt.bar(x - w / 2, a_mean, w, yerr=a_std, label=args.a_label)
        plt.bar(x + w / 2, b_mean, w, yerr=b_std, label=args.b_label)
        plt.xticks(x, names, rotation=30, ha="right")
        plt.title("A/B Metrics Mean±Std")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "ab_metrics_bar.png"), dpi=160)
        plt.close()

    # 2) per-seed line plots for key control metrics
    seed_metrics = ["relax_mean", "relax_p95", "relax_zero_rate"]
    for k in seed_metrics:
        av = _collect_metric(A, k)
        bv = _collect_metric(B, k)
        if not av or not bv:
            continue
        n = min(len(av), len(bv))
        plt.figure(figsize=(8, 4))
        plt.plot(range(n), av[:n], marker="o", label=args.a_label)
        plt.plot(range(n), bv[:n], marker="s", label=args.b_label)
        plt.xlabel("seed index")
        plt.ylabel(k)
        plt.title(f"{k} per seed")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{k}_per_seed.png"), dpi=160)
        plt.close()

    # 3) FD eps scan (mae_vs_hdot_auto)
    A_eps = _collect_fd_eps(A)
    B_eps = _collect_fd_eps(B)
    common_eps = sorted(set(A_eps.keys()) & set(B_eps.keys()), key=lambda s: float(s))
    if common_eps:
        a_y, a_e, b_y, b_e = [], [], [], []
        for eps in common_eps:
            am, asd = _mean_std(A_eps[eps])
            bm, bsd = _mean_std(B_eps[eps])
            a_y.append(am)
            a_e.append(asd)
            b_y.append(bm)
            b_e.append(bsd)
        x = np.array([float(e) for e in common_eps], dtype=np.float64)
        plt.figure(figsize=(8, 4))
        plt.errorbar(x, a_y, yerr=a_e, marker="o", label=args.a_label)
        plt.errorbar(x, b_y, yerr=b_e, marker="s", label=args.b_label)
        plt.xscale("log")
        plt.xlabel("FD eps")
        plt.ylabel("MAE( hdot_auto vs hdot_fd )")
        plt.title("FD Eps Scan")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "fd_eps_scan.png"), dpi=160)
        plt.close()

    summary = {
        "A_count": len(A),
        "B_count": len(B),
        "keys_plotted": names,
        "figures_dir": fig_dir,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

