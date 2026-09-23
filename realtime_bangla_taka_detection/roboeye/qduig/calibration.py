"""Novelty 3 — Heteroscedastic Evidence Recalibration (HER) and baselines.

HER is not temperature scaling: a 4-parameter map adds quality- and
entropy-conditioned residuals to the temperature-scaled logit.
All parameters are fit on validation only.

Also implements: temperature scaling, entropy mapping, predictive-confidence
identity, and Monte-Carlo dropout averaging.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def binary_entropy(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def nll_binary(y: np.ndarray, p: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(len(y), 1)
    for i in range(n_bins):
        right = bins[i + 1] if i < n_bins - 1 else bins[i + 1] + 1e-12
        mask = (p >= bins[i]) & (p < right if i < n_bins - 1 else p <= bins[i + 1])
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def adaptive_ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Equal-mass (adaptive) ECE."""
    n = len(p)
    if n == 0:
        return float("nan")
    order = np.argsort(p)
    y_s, p_s = y[order], p[order]
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    ece = 0.0
    for i in range(n_bins):
        sl = slice(edges[i], edges[i + 1])
        if edges[i + 1] <= edges[i]:
            continue
        ece += (len(p_s[sl]) / n) * abs(float(y_s[sl].mean()) - float(p_s[sl].mean()))
    return float(ece)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


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


def confidence_histogram(p: np.ndarray, n_bins: int = 20) -> dict:
    hist, edges = np.histogram(p, bins=n_bins, range=(0.0, 1.0))
    return {"counts": hist.astype(int).tolist(), "edges": edges.astype(float).tolist()}


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    t = torch.nn.Parameter(torch.ones(1))
    y_t = torch.tensor(y, dtype=torch.long)
    logits_t = torch.tensor(logits, dtype=torch.float32)
    opt = torch.optim.LBFGS([t], lr=0.25, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits_t / t.clamp(min=1e-3), y_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.detach().clamp(min=0.05, max=20.0).item())


def apply_temperature(logits: np.ndarray, t: float) -> np.ndarray:
    return _softmax_np(logits / max(t, 1e-6))


def entropy_mapped_confidence(logits: np.ndarray) -> np.ndarray:
    """Map 1 - normalised binary entropy of softmax to a genuine score.

    Keeps the predicted class direction; only the *magnitude* is entropy-based.
    """
    p = _softmax_np(logits)[:, 1]
    h = binary_entropy(p)
    h_max = float(np.log(2.0))
    conf = 1.0 - h / h_max
    pred = (p >= 0.5).astype(np.float64)
    # reconstruct a genuine probability that is high when pred=1 and confident
    return np.where(pred == 1, 0.5 + 0.5 * conf, 0.5 - 0.5 * conf)


class HERCalibrator:
    """Heteroscedastic Evidence Recalibration.

    logit_cal = logit/T + α (q − q0) + β (H − H0) + γ · margin

    Fit on validation NLL. Never fit on test.
    """

    def __init__(self) -> None:
        self.params = {
            "T": 1.0,
            "alpha": 0.0,
            "beta": 0.0,
            "gamma": 0.0,
            "q0": 0.5,
            "H0": float(np.log(2.0) * 0.5),
        }

    def _cal_logit(self, logit_diff: np.ndarray, q: np.ndarray, h: np.ndarray, margin: np.ndarray, p) -> np.ndarray:
        return (
            logit_diff / max(p["T"], 1e-6)
            + p["alpha"] * (q - p["q0"])
            + p["beta"] * (h - p["H0"])
            + p["gamma"] * margin
        )

    def transform(self, logits: np.ndarray, quality: np.ndarray, entropy: np.ndarray | None = None) -> np.ndarray:
        p1 = _softmax_np(logits)[:, 1]
        h = binary_entropy(p1) if entropy is None else entropy
        logit_diff = logits[:, 1] - logits[:, 0]
        margin = np.abs(logits[:, 1] - logits[:, 0])
        z = self._cal_logit(logit_diff, quality, h, margin, self.params)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, logits: np.ndarray, y: np.ndarray, quality: np.ndarray) -> dict[str, float]:
        y = np.asarray(y).astype(np.float64)
        q = np.asarray(quality).astype(np.float64)
        p1 = _softmax_np(logits)[:, 1]
        h = binary_entropy(p1)
        logit_diff = logits[:, 1] - logits[:, 0]
        margin = np.abs(logit_diff)
        self.params["q0"] = float(np.mean(q))
        self.params["H0"] = float(np.mean(h))

        vec = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32))
        y_t = torch.tensor(y, dtype=torch.float32)
        ld = torch.tensor(logit_diff, dtype=torch.float32)
        qt = torch.tensor(q - self.params["q0"], dtype=torch.float32)
        ht = torch.tensor(h - self.params["H0"], dtype=torch.float32)
        mt = torch.tensor(margin, dtype=torch.float32)
        opt = torch.optim.LBFGS([vec], lr=0.2, max_iter=80)

        def closure():
            opt.zero_grad()
            t, a, b, g = vec[0].clamp(0.05, 20.0), vec[1], vec[2], vec[3]
            z = ld / t + a * qt + b * ht + g * mt
            loss = F.binary_cross_entropy_with_logits(z, y_t)
            loss.backward()
            return loss

        opt.step(closure)
        t, a, b, g = [float(x) for x in vec.detach().cpu().tolist()]
        self.params["T"] = float(np.clip(t, 0.05, 20.0))
        self.params["alpha"] = a
        self.params["beta"] = b
        self.params["gamma"] = g
        return dict(self.params)


def evaluate_calibration_suite(
    logits: np.ndarray,
    y: np.ndarray,
    quality: np.ndarray,
    her: HERCalibrator,
    temperature: float,
    mc_probs: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare temperature, entropy, confidence, MC dropout, and HER on the same notes."""
    y = np.asarray(y).astype(np.float64)
    raw = _softmax_np(logits)[:, 1]
    methods = {
        "predictive_confidence": raw,
        "temperature": apply_temperature(logits, temperature)[:, 1],
        "entropy": entropy_mapped_confidence(logits),
        "her": her.transform(logits, quality),
    }
    if mc_probs is not None:
        methods["mc_dropout"] = np.asarray(mc_probs)
    out = {}
    for name, p in methods.items():
        out[name] = {
            "ece": expected_calibration_error(y, p),
            "adaptive_ece": adaptive_ece(y, p),
            "brier": brier(y, p),
            "nll": nll_binary(y, p),
            "reliability": reliability_diagram(y, p),
            "confidence_histogram": confidence_histogram(p),
            "mean_confidence": float(np.mean(np.maximum(p, 1.0 - p))),
        }
    return out
