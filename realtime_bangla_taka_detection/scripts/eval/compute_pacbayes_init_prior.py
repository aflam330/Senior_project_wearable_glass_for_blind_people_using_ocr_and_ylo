"""PAC-Bayes penalty when the prior mean is the initialization, not zero.

CUDA is hidden. The test set is not used. This does not train.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import build_model

CKPT = ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "checkpoint.pt"
OUT = ROOT / "results" / "theory" / "pacbayes_init_prior_seed42.json"
DELTA = 0.05


def kl_shift(delta2: float, d: int, sigma0: float, rho: float) -> float:
    s2 = sigma0 * sigma0
    r2 = rho * rho
    return 0.5 * (delta2 / s2 + d * r2 / s2 - d + d * math.log(s2 / r2))


def main() -> None:
    cfg = load_config(ROOT / "configs" / "proposed_prefix_ft.yaml")
    model = build_model(cfg)
    init = {k: v.detach().float().clone() for k, v in model.state_dict().items() if torch.is_floating_point(v)}
    try:
        blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(CKPT, map_location="cpu")
    state = blob["model"]
    delta2 = 0.0
    d = 0
    missing = []
    for key, w0 in init.items():
        if key not in state:
            missing.append(key)
            continue
        w = state[key].detach().float()
        if w.shape != w0.shape:
            missing.append(key)
            continue
        delta2 += float((w - w0).pow(2).sum().item())
        d += int(w.numel())
    meta = json.loads((ROOT / "results" / "camva" / "splits" / "split_metadata.json").read_text(encoding="utf-8"))
    n = int(meta["train"]["n_notes"])
    # rho = sigma0 makes the variance part of the KL zero. Sweep sigma0.
    rows = []
    for sigma0 in (0.01, 0.1, 1.0, 10.0, 100.0):
        kl = kl_shift(delta2, d, sigma0, sigma0)
        penalty = math.sqrt((kl + math.log(2 * math.sqrt(n) / DELTA)) / (2 * n))
        rows.append({"sigma0": sigma0, "rho": sigma0, "kl": kl, "penalty": penalty, "penalty_below_1": penalty < 1.0})
    # Smallest sigma0 such that the penalty is under 1, if any, on a log grid.
    best = None
    for i in range(-4, 6):
        sigma0 = 10 ** (i / 2)
        kl = kl_shift(delta2, d, sigma0, sigma0)
        penalty = math.sqrt((kl + math.log(2 * math.sqrt(n) / DELTA)) / (2 * n))
        row = {"sigma0": sigma0, "rho": sigma0, "kl": kl, "penalty": penalty}
        if best is None or penalty < best["penalty"]:
            best = row
    payload = {
        "checkpoint": str(CKPT),
        "prior": "N(w_init, sigma0^2 I), w_init is the network before JaalTaka training",
        "posterior": "N(w, rho^2 I) with rho = sigma0, so the variance contribution to KL is 0",
        "n_parameters_matched": d,
        "missing_keys": missing,
        "displacement_squared_l2": delta2,
        "n_train_notes": n,
        "delta": DELTA,
        "empirical_gibbs_risk": "NOT_MEASURED",
        "note": "A penalty below 1 is not a full bound. The Gibbs risk at that posterior variance was not measured. The test set was not used.",
        "rows": rows,
        "smallest_penalty_on_log_grid": best,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"d": d, "delta2": delta2, "best_penalty": best["penalty"], "best_sigma0": best["sigma0"], "missing": len(missing)}, indent=2))


if __name__ == "__main__":
    main()
