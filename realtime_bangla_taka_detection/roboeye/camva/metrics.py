"""Note-level classification metrics, bootstrap CIs, calibration, timing."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(np.float64)
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n = max(len(y_true), 1)
    acc = (tp + tn) / n
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    far = fp / max(fp + tn, 1)
    frr = fn / max(fn + tp, 1)
    cm = [[tn, fp], [fn, tp]]
    return {
        "n": int(len(y_true)),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "sensitivity": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
        "false_acceptance_rate": float(far),
        "false_rejection_rate": float(frr),
        "confusion_matrix": cm,
        "roc_auc": float(_roc_auc(y_true, y_score)),
        "pr_auc": float(_pr_auc(y_true, y_score)),
        "brier": float(np.mean((y_score - y_true) ** 2)),
        "ece": float(expected_calibration_error(y_true, y_score)),
    }


def _roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney
    ranks = np.argsort(np.argsort(np.concatenate([neg, pos])))
    pos_ranks = ranks[len(neg) :]
    u = pos_ranks.sum() - len(pos) * (len(pos) - 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def _pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(-s)
    y = y[order]
    tp = 0
    fp = 0
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    precs, recs = [], []
    for label in y:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precs.append(tp / max(tp + fp, 1))
        recs.append(tp / n_pos)
    recs = np.array([0.0] + recs)
    precs = np.array([1.0] + precs)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(precs, recs))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(len(y), 1)
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if not mask.any():
            continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_diagram(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        rows.append(
            {
                "bin_left": float(bins[i]),
                "bin_right": float(bins[i + 1]),
                "n": int(mask.sum()),
                "accuracy": float(y[mask].mean()) if mask.any() else None,
                "confidence": float(p[mask].mean()) if mask.any() else None,
            }
        )
    return {"n_bins": n_bins, "bins": rows}


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "accuracy",
    n_boot: int = 1000,
    seed: int = 42,
    threshold: float = 0.5,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = binary_metrics(y_true[idx], y_score[idx], threshold=threshold)
        stats.append(m[metric])
    arr = np.array(stats, dtype=np.float64)
    return {
        "metric": metric,
        "mean": float(np.nanmean(arr)),
        "ci95_low": float(np.nanpercentile(arr, 2.5)),
        "ci95_high": float(np.nanpercentile(arr, 97.5)),
        "n_boot": n_boot,
        "unit": "physical_note",
    }


def paired_bootstrap(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    metric: str = "accuracy",
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """score_b minus score_a. Independent unit = note (row)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ma = binary_metrics(y_true[idx], score_a[idx])[metric]
        mb = binary_metrics(y_true[idx], score_b[idx])[metric]
        diffs.append(mb - ma)
    arr = np.array(diffs, dtype=np.float64)
    # two-sided p: fraction of diffs on the other side of 0, doubled
    p = 2 * min(float((arr >= 0).mean()), float((arr <= 0).mean()))
    p = min(1.0, p)
    return {
        "metric": metric,
        "mean_difference": float(arr.mean()),
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
        "p_value": float(p),
        "n_boot": n_boot,
        "unit": "physical_note",
        "test": "paired_bootstrap",
    }


def roc_curve_data(y: np.ndarray, s: np.ndarray) -> dict:
    order = np.argsort(-s)
    y = y[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = tps / max(int(y.sum()), 1)
    fpr = fps / max(int((1 - y).sum()), 1)
    return {"fpr": fpr.astype(float).tolist(), "tpr": tpr.astype(float).tolist()}


def pr_curve_data(y: np.ndarray, s: np.ndarray) -> dict:
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    rec = tp / max(int(y.sum()), 1)
    prec = tp / np.maximum(tp + fp, 1)
    return {"recall": rec.astype(float).tolist(), "precision": prec.astype(float).tolist()}


class Timer:
    def __init__(self) -> None:
        self.times: list[float] = []

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.times.append((time.perf_counter() - self._t0) * 1000.0)

    def summary(self) -> dict:
        if not self.times:
            return {"n": 0, "median_ms": None, "p95_ms": None, "mean_ms": None}
        arr = np.array(self.times)
        return {
            "n": int(len(arr)),
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "mean_ms": float(arr.mean()),
            "sum_ms": float(arr.sum()),
        }
