"""Novelty 6 — Oracle subset analysis.

For each note, assume access to all six stored views and evaluate every
non-empty subset under ground truth. Used ONLY for analysis.
Never used as a training target or as a deployed policy.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np


def all_subsets(n_views: int = 6) -> list[tuple[int, ...]]:
    idx = list(range(n_views))
    out: list[tuple[int, ...]] = []
    for k in range(1, n_views + 1):
        out.extend(combinations(idx, k))
    return out


def oracle_for_note(
    predict_subset: Callable[[tuple[int, ...]], tuple[int, float]],
    y_true: int,
    n_views: int = 6,
) -> dict:
    """predict_subset(idx) -> (pred_label, genuine_score)."""
    best = None
    rows = []
    for subset in all_subsets(n_views):
        pred, score = predict_subset(subset)
        correct = int(pred == y_true)
        row = {
            "subset": list(subset),
            "k": len(subset),
            "pred": int(pred),
            "score": float(score),
            "correct": correct,
        }
        rows.append(row)
        if best is None:
            best = row
            continue
        # prefer correct, then fewer views, then higher |score-0.5|
        if (correct, -len(subset), abs(score - 0.5)) > (
            best["correct"],
            -best["k"],
            abs(best["score"] - 0.5),
        ):
            best = row
    any_correct = any(r["correct"] for r in rows)
    min_k_correct = min((r["k"] for r in rows if r["correct"]), default=None)
    return {
        "oracle_correct": int(best["correct"]) if best else 0,
        "oracle_k": int(best["k"]) if best else n_views,
        "oracle_subset": best["subset"] if best else [],
        "any_subset_correct": int(any_correct),
        "min_k_for_correct": min_k_correct,
        "theoretical_subset_potential": 1.0 if any_correct else 0.0,
        "n_subsets_evaluated": len(rows),
    }


def summarise_oracle(per_note: list[dict], learned_k: list[int], learned_correct: list[int]) -> dict:
    pot = np.array([r["theoretical_subset_potential"] for r in per_note], dtype=np.float64)
    ora_acc = np.array([r["oracle_correct"] for r in per_note], dtype=np.float64)
    ora_k = np.array([r["oracle_k"] for r in per_note], dtype=np.float64)
    learn_acc = np.array(learned_correct, dtype=np.float64)
    learn_k = np.array(learned_k, dtype=np.float64)
    return {
        "n": len(per_note),
        "oracle_accuracy": float(ora_acc.mean()) if len(ora_acc) else None,
        "oracle_mean_views": float(ora_k.mean()) if len(ora_k) else None,
        "theoretical_subset_potential": float(pot.mean()) if len(pot) else None,
        "learned_accuracy": float(learn_acc.mean()) if len(learn_acc) else None,
        "learned_mean_views": float(learn_k.mean()) if len(learn_k) else None,
        "accuracy_gap_oracle_minus_learned": float(ora_acc.mean() - learn_acc.mean()) if len(ora_acc) else None,
        "view_gap_learned_minus_oracle": float(learn_k.mean() - ora_k.mean()) if len(ora_k) else None,
        "notes_unsolvable_by_any_subset": int((pot == 0).sum()),
        "use": "analysis_only_never_for_training_or_deployment",
    }
