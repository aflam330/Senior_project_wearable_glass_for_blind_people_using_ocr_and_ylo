"""PAC-Bayes on a linear head fit to half of the training notes.

The published PRMVT checkpoint is not the posterior: it was trained on every
training note. Features come from that frozen encoder. The head is fit on D1.
D2 and the test set are not used to choose the weight decay.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roboeye.camva.engine import make_loader
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE
from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import load_qduig

OUT = ROOT / "results" / "theory" / "pacbayes_head_d1d2_seed42.json"
DECAYS = (1.0, 0.1, 0.01, 0.001)
DELTA = 0.05
GIBBS_SAMPLES = 8


def _kl_bernoulli(p: float, q: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    q = min(max(q, 1e-6), 1.0 - 1e-6)
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def _kl_bound(emp: float, kl: float, n: int) -> float:
    rhs = (kl + math.log(2.0 * math.sqrt(n) / DELTA)) / n
    lo, hi = emp, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _kl_bernoulli(emp, mid) <= rhs:
            lo = mid
        else:
            hi = mid
    return lo


def _fit(x: torch.Tensor, y: torch.Tensor, decay: float) -> torch.Tensor:
    weight = torch.zeros(x.size(1), device=x.device, requires_grad=True)
    bias = torch.zeros((), device=x.device, requires_grad=True)
    opt = torch.optim.Adam([weight, bias], lr=0.01)
    yf = y.float()
    for _ in range(300):
        opt.zero_grad()
        logit = x @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yf)
        loss = loss + 0.5 * decay * weight.pow(2).sum()
        loss.backward()
        if not torch.isfinite(loss):
            break
        torch.nn.utils.clip_grad_norm_([weight, bias], 1.0)
        opt.step()
    return torch.cat([weight.detach(), bias.detach().view(1)])


def _risk(x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor) -> float:
    logit = x @ theta[:-1] + theta[-1]
    pred = (logit >= 0).long()
    return float((pred != y).float().mean())


def main() -> None:
    splits, records = load_splits()
    model = load_qduig(
        ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "checkpoint.pt",
        load_config(ROOT / "configs" / "proposed_prefix_ft.yaml"),
    )
    model.eval()
    loader = make_loader(splits["train"], records, n_views=6, train=False, batch=8, workers=0)
    feats, labels = [], []
    for batch in loader:
        views = batch["views"].to(DEVICE)
        mask = torch.ones(views.size(0), views.size(1), dtype=torch.long, device=DEVICE)
        with torch.inference_mode():
            feats.append(model(views, mask)["fused_h"].detach().cpu())
        labels.append(batch["label"])
    x = torch.cat(feats).float()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.cat(labels)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(y))
    mid = len(y) // 2
    d1, d2 = order[:mid], order[mid:]
    mu = x[d1].mean(dim=0)
    std = x[d1].std(dim=0).clamp_min(1e-3)
    x = (x - mu) / std
    inner = rng.permutation(len(d1))
    cut = max(1, int(0.8 * len(d1)))
    fit_idx, sel_idx = d1[inner[:cut]], d1[inner[cut:]]
    held = []
    for decay in DECAYS:
        theta = _fit(x[fit_idx].to(DEVICE), y[fit_idx].to(DEVICE), decay).cpu()
        held.append((decay, _risk(x[sel_idx], y[sel_idx], theta)))
    decay = min(held, key=lambda item: (item[1], item[0]))[0]
    theta = _fit(x[d1].to(DEVICE), y[d1].to(DEVICE), decay).cpu()
    kl = 0.5 * float(theta.pow(2).sum())
    n = int(len(d2))
    det = _risk(x[d2], y[d2], theta)
    gibbs = []
    gen = torch.Generator().manual_seed(42)
    for _ in range(GIBBS_SAMPLES):
        draw = theta + torch.randn(theta.shape, generator=gen)
        gibbs.append(_risk(x[d2], y[d2], draw))
    emp = float(np.mean(gibbs))
    penalty = math.sqrt((kl + math.log(2.0 * math.sqrt(n) / DELTA)) / (2.0 * n))
    mcallester = emp + penalty
    kl_bound = _kl_bound(emp, kl, n)
    payload = {
        "posterior": "N(theta, I) on a linear head",
        "prior": "N(0, I), fixed before D2",
        "features": "frozen PRMVT fused_h, 6 views",
        "d1_notes": int(len(d1)),
        "d2_notes": n,
        "test_used": False,
        "weight_decay_grid": list(DECAYS),
        "weight_decay_chosen_on": "inner 20% of D1",
        "weight_decay": decay,
        "kl": kl,
        "d2_deterministic_risk": det,
        "d2_gibbs_risk_mean": emp,
        "gibbs_samples": GIBBS_SAMPLES,
        "mcallester": mcallester,
        "pacbayes_kl": kl_bound,
        "non_vacuous": bool(min(mcallester, kl_bound) < 1.0),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("kl", "d2_gibbs_risk_mean", "mcallester", "pacbayes_kl", "non_vacuous")}, indent=2))


if __name__ == "__main__":
    main()
