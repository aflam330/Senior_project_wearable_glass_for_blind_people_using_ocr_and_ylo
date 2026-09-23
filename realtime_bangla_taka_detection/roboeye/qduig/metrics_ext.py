"""Extended metrics: balanced accuracy, McNemar, Wilcoxon, effect size, FDR."""

from __future__ import annotations

from typing import Any

import numpy as np

from .calibration import adaptive_ece, brier, expected_calibration_error, nll_binary


def balanced_accuracy(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    pred = (p >= threshold).astype(int)
    pos = y == 1
    neg = y == 0
    sens = float(pred[pos].mean()) if pos.any() else float("nan")
    spec = float((1 - pred[neg]).mean()) if neg.any() else float("nan")
    return float((sens + spec) / 2.0)


def macro_f1(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    pred = (p >= threshold).astype(int)
    f1s = []
    for cls in (0, 1):
        tp = int(((pred == cls) & (y == cls)).sum())
        fp = int(((pred == cls) & (y != cls)).sum())
        fn = int(((pred != cls) & (y == cls)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-12))
    return float(np.mean(f1s))


def mcnemar(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, Any]:
    """McNemar test on paired correctness. b vs a."""
    a_ok = pred_a == y
    b_ok = pred_b == y
    n01 = int((a_ok & ~b_ok).sum())
    n10 = int((~a_ok & b_ok).sum())
    n = n01 + n10
    if n == 0:
        p = 1.0
        chi2 = 0.0
    else:
        chi2 = (abs(n01 - n10) - 1) ** 2 / n
        # chi-square 1 df survival via erfc approximation
        p = float(_chi2_sf(chi2, df=1))
    return {
        "n01_a_correct_b_wrong": n01,
        "n10_a_wrong_b_correct": n10,
        "chi2_continuity": float(chi2),
        "p_value": p,
        "test": "mcnemar",
    }


def _chi2_sf(x: float, df: int = 1) -> float:
    if df != 1:
        return float("nan")
    # P(Chi^2_1 > x) = erfc(sqrt(x/2))
    from math import erfc, sqrt

    return float(erfc(sqrt(max(x, 0.0) / 2.0)))


def wilcoxon_signed(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank on paired differences b-a. Exact for n<=20 else normal."""
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    d = d[d != 0]
    n = len(d)
    if n < 6:
        return {"n": n, "p_value": None, "statistic": None, "note": "too_few_nonzero_pairs"}
    ranks = _rankdata(np.abs(d))
    w_pos = float(ranks[d > 0].sum())
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    z = (w_pos - mean) / max(var ** 0.5, 1e-12)
    p = float(_norm_sf(abs(z)) * 2.0)
    return {"n": n, "statistic": w_pos, "z": float(z), "p_value": min(1.0, p), "test": "wilcoxon_signed"}


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(x, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    return ranks


def _norm_sf(z: float) -> float:
    from math import erfc, sqrt

    return 0.5 * erfc(z / sqrt(2.0))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    gt = 0
    lt = 0
    for x in a:
        gt += int((b > x).sum())
        lt += int((b < x).sum())
    n = max(len(a) * len(b), 1)
    return float((gt - lt) / n)


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    n = len(pvalues)
    order = np.argsort(pvalues)
    adj = [1.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        raw = pvalues[idx]
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            adj[idx] = float("nan")
            continue
        val = min(1.0, raw * (n - rank))
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def full_binary_report(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(np.float64)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    n = max(len(y), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "n": int(len(y)),
        "accuracy": float((tp + tn) / n),
        "balanced_accuracy": float((rec + spec) / 2.0),
        "precision": float(prec),
        "recall": float(rec),
        "sensitivity": float(rec),
        "specificity": float(spec),
        "f1": float(2 * prec * rec / max(prec + rec, 1e-12)),
        "macro_f1": macro_f1(y, p, threshold),
        "false_acceptance_rate": float(fp / max(fp + tn, 1)),
        "false_rejection_rate": float(fn / max(fn + tp, 1)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "roc_auc": _auc(y, p),
        "pr_auc": _pr_auc(y, p),
        "brier": brier(y, p),
        "ece": expected_calibration_error(y, p),
        "adaptive_ece": adaptive_ece(y, p),
        "nll": nll_binary(y, p),
    }


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([neg, pos])))
    u = ranks[len(neg) :].sum() - len(pos) * (len(pos) - 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def _pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(-s)
    y = y[order]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    tp = fp = 0
    precs, recs = [1.0], [0.0]
    for lab in y:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        precs.append(tp / max(tp + fp, 1))
        recs.append(tp / n_pos)
    return float(np.trapezoid(np.array(precs), np.array(recs)))
