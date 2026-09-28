"""Data-dependent PAC-Bayes bound for the D1-trained network.

The prior and the posterior are the same Gaussian, N(w_D1, rho^2 I), with
rho fixed at 1e-4 before D2 is scored. KL is therefore 0. Gibbs risk is the
mean 0-1 error of four posterior draws on D2. The test split is not read.
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

CKPT = ROOT / "results" / "theory" / "pacbayes_d1_model" / "seed42" / "checkpoint.pt"
SPLIT = ROOT / "results" / "theory" / "pacbayes_d1_model" / "seed42" / "split_ids.json"
CFG = ROOT / "configs" / "ablation_prefix" / "her_base.yaml"
OUT = ROOT / "results" / "theory" / "pacbayes_d1_bound_seed42.json"
RHO = 1e-4
DELTA = 0.05
GIBBS_SAMPLES = 4


def _errors(model, loader) -> np.ndarray:
    wrong = []
    for batch in loader:
        views = batch["views"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        y = batch["label"].to(DEVICE)
        with torch.inference_mode():
            pred = model(views, mask)["logits"].argmax(dim=1)
        wrong.append((pred != y).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(wrong)


def _hoeffding(emp: float, n: int) -> float:
    return emp + math.sqrt(math.log(1.0 / DELTA) / (2.0 * n))


def _empirical_bernstein(losses: np.ndarray) -> float:
    n = int(losses.size)
    emp = float(losses.mean())
    var = float(np.sum((losses - emp) ** 2) / n)
    log_term = math.log(2.0 / DELTA)
    return emp + math.sqrt(2.0 * var * log_term / n) + (7.0 * log_term) / (3.0 * (n - 1))


def main() -> None:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    d2 = list(split["d2"])
    _, records = load_splits()
    model = load_qduig(CKPT, load_config(CFG))
    model.eval()
    loader = make_loader(d2, records, n_views=6, train=False, batch=8, workers=0)
    det_losses = _errors(model, loader)
    det = float(det_losses.mean())
    base = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    gen = torch.Generator().manual_seed(42)
    gibbs = []
    for sample_i in range(GIBBS_SAMPLES):
        noisy = {}
        for key, value in base.items():
            if torch.is_floating_point(value):
                noise = torch.randn(value.shape, generator=gen, dtype=value.dtype)
                noisy[key] = value + RHO * noise
            else:
                noisy[key] = value.clone()
        model.load_state_dict({k: v.to(DEVICE) for k, v in noisy.items()})
        err = float(_errors(model, loader).mean())
        gibbs.append(err)
        print(json.dumps({"gibbs_sample": sample_i, "d2_error": err}), flush=True)
    model.load_state_dict({k: v.to(DEVICE) for k, v in base.items()})
    n = int(det_losses.size)
    emp = float(np.mean(gibbs))
    penalty = math.sqrt(math.log(2.0 * math.sqrt(n) / DELTA) / (2.0 * n))
    layer_rows = []
    for key, value in base.items():
        if torch.is_floating_point(value):
            layer_rows.append({
                "name": key,
                "n_parameters": int(value.numel()),
                "displacement_l2_sq": 0.0,
                "kl": 0.0,
            })
    payload = {
        "checkpoint": str(CKPT),
        "split": "held-out training half D2",
        "n_d2": n,
        "n_d1": len(split["d1"]),
        "test_used": False,
        "posterior": "N(w_D1, rho^2 I)",
        "prior": "N(w_D1, rho^2 I), the same mean, fit without D2",
        "rho": RHO,
        "sigma0": RHO,
        "kl": 0.0,
        "delta": DELTA,
        "d2_deterministic_error": det,
        "d2_gibbs_errors": gibbs,
        "d2_gibbs_mean": emp,
        "mcallester": emp + penalty,
        "mcallester_penalty": penalty,
        "hoeffding_upper": _hoeffding(det, n),
        "empirical_bernstein_upper": _empirical_bernstein(det_losses),
        "layerwise_kl_sum": 0.0,
        "layerwise_note": "Posterior mean equals the prior mean, so every layer KL is 0. The sum is the same McAllester bound.",
        "n_layers_with_float_weights": len(layer_rows),
        "rademacher_full_network": "NOT_MEASURED",
        "note": "This bound is for the D1-trained network. It is not a bound on the published PRMVT test accuracy.",
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "d2_deterministic_error": det,
        "d2_gibbs_mean": emp,
        "mcallester": payload["mcallester"],
        "hoeffding_upper": payload["hoeffding_upper"],
        "empirical_bernstein_upper": payload["empirical_bernstein_upper"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
