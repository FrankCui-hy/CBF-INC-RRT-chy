from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F


SAMPLE_TYPE_NAMES = {
    0: "far_random",
    1: "medium_close",
    2: "near_boundary",
    3: "collision_unsafe",
}


def raylink_loss(
    hit_logits: torch.Tensor,
    depth_norm_pred: torch.Tensor,
    hit_mask: torch.Tensor,
    depth_norm: torch.Tensor,
    pos_weight: Optional[torch.Tensor],
    lambda_hit: float,
    lambda_depth: float,
) -> Dict[str, torch.Tensor]:
    hit_target = hit_mask.float()
    if pos_weight is not None:
        pos_weight = pos_weight.to(device=hit_logits.device, dtype=hit_logits.dtype).reshape(1)
    bce = F.binary_cross_entropy_with_logits(hit_logits, hit_target, pos_weight=pos_weight)
    hit = hit_target > 0.5
    if hit.any():
        depth_loss = F.smooth_l1_loss(depth_norm_pred[hit], depth_norm.float()[hit], reduction="mean")
    else:
        depth_loss = depth_norm_pred.sum() * 0.0
    total = float(lambda_hit) * bce + float(lambda_depth) * depth_loss
    return {"total": total, "hit": bce, "depth": depth_loss}


def binary_counts(y_true: torch.Tensor, y_prob: torch.Tensor, threshold: float) -> Dict[str, float]:
    y_true = y_true.bool()
    y_pred = y_prob >= float(threshold)
    tp = torch.logical_and(y_pred, y_true).sum().item()
    tn = torch.logical_and(~y_pred, ~y_true).sum().item()
    fp = torch.logical_and(y_pred, ~y_true).sum().item()
    fn = torch.logical_and(~y_pred, y_true).sum().item()
    total = max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "hit_acc": float((tp + tn) / total),
        "hit_precision": float(precision),
        "hit_recall": float(recall),
        "hit_f1": float(f1),
        "hit_false_negative_rate": float(fn / max(tp + fn, 1)),
        "hit_false_positive_rate": float(fp / max(fp + tn, 1)),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def depth_metrics(depth_pred: torch.Tensor, depth_true: torch.Tensor, hit_mask: torch.Tensor, r_max: float) -> Dict[str, float]:
    hit = hit_mask.bool()
    if not hit.any():
        return {
            "depth_mae_hit_norm": float("nan"),
            "depth_rmse_hit_norm": float("nan"),
            "depth_mae_hit_meter": float("nan"),
            "depth_rmse_hit_meter": float("nan"),
        }
    err = (depth_pred.float()[hit] - depth_true.float()[hit]).abs()
    rmse = torch.sqrt((err.pow(2)).mean())
    return {
        "depth_mae_hit_norm": float(err.mean().item()),
        "depth_rmse_hit_norm": float(rmse.item()),
        "depth_mae_hit_meter": float((err * float(r_max)).mean().item()),
        "depth_rmse_hit_meter": float((rmse * float(r_max)).item()),
    }


def summarize_predictions(
    hit_logits: torch.Tensor,
    depth_pred: torch.Tensor,
    hit_mask: torch.Tensor,
    depth_true: torch.Tensor,
    sample_type: torch.Tensor,
    r_max: float,
    threshold: float,
) -> Dict[str, float]:
    hit_prob = torch.sigmoid(hit_logits.float())
    summary: Dict[str, float] = {}
    summary.update(binary_counts(hit_mask.flatten(), hit_prob.flatten(), threshold))
    summary.update(depth_metrics(depth_pred, depth_true, hit_mask, r_max))
    for type_code, type_name in SAMPLE_TYPE_NAMES.items():
        rows = sample_type == int(type_code)
        if not rows.any():
            continue
        prefix = f"{type_name}/"
        type_hit = hit_mask[rows]
        type_prob = hit_prob[rows]
        type_depth_pred = depth_pred[rows]
        type_depth_true = depth_true[rows]
        bm = binary_counts(type_hit.flatten(), type_prob.flatten(), threshold)
        dm = depth_metrics(type_depth_pred, type_depth_true, type_hit, r_max)
        for key, value in {**bm, **dm}.items():
            summary[prefix + key] = value
    return summary


def select_safety_threshold(
    hit_logits: torch.Tensor,
    hit_mask: torch.Tensor,
    sample_type: torch.Tensor,
    thresholds: Iterable[float],
    near_recall_target: float = 0.95,
    collision_recall_target: float = 0.98,
) -> tuple[float, Dict[str, Dict[str, float]]]:
    thresholds = [float(x) for x in thresholds]
    hit_prob = torch.sigmoid(hit_logits.float())
    rows_near = sample_type == 2
    rows_collision = sample_type == 3
    table: Dict[str, Dict[str, float]] = {}
    candidates = []
    for threshold in thresholds:
        overall = binary_counts(hit_mask.flatten(), hit_prob.flatten(), float(threshold))
        near = binary_counts(hit_mask[rows_near].flatten(), hit_prob[rows_near].flatten(), float(threshold)) if rows_near.any() else {}
        collision = binary_counts(hit_mask[rows_collision].flatten(), hit_prob[rows_collision].flatten(), float(threshold)) if rows_collision.any() else {}
        row = {
            "overall_hit_recall": overall.get("hit_recall", 0.0),
            "overall_hit_precision": overall.get("hit_precision", 0.0),
            "overall_hit_false_positive_rate": overall.get("hit_false_positive_rate", 1.0),
            "near_boundary_hit_recall": near.get("hit_recall", 0.0),
            "near_boundary_hit_false_negative_rate": near.get("hit_false_negative_rate", 1.0),
            "collision_unsafe_hit_recall": collision.get("hit_recall", 0.0),
            "collision_unsafe_hit_false_negative_rate": collision.get("hit_false_negative_rate", 1.0),
        }
        table[f"{float(threshold):.3f}"] = row
        if row["near_boundary_hit_recall"] >= float(near_recall_target) and row["collision_unsafe_hit_recall"] >= float(collision_recall_target):
            candidates.append((float(threshold), row))

    if candidates:
        best = min(candidates, key=lambda x: x[1]["overall_hit_false_positive_rate"])
        return best[0], table
    best_threshold = max(
        (float(t) for t in thresholds),
        key=lambda t: table[f"{float(t):.3f}"]["near_boundary_hit_recall"] + table[f"{float(t):.3f}"]["collision_unsafe_hit_recall"],
    )
    return best_threshold, table
