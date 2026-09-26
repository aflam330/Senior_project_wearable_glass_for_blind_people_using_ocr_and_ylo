"""KL and McAllester complexity for a Gaussian posterior on the saved PRMVT weights.

The prior grid is fixed in this file. The test set is not used.
CUDA is hidden so this can run beside a GPU eval.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "checkpoint.pt"
OUT = ROOT / "results" / "theory" / "pacbayes_prmvt_seed42.json"

SIGMA0 = (0.1, 1.0, 10.0)
RHO = (0.001, 0.01, 0.1)
DELTA = 0.05


def _kl(w2: float, d: int, sigma0: float, rho: float) -> float:
    s2 = sigma0 * sigma0
    r2 = rho * rho
    return 0.5 * (w2 / s2 + d * r2 / s2 - d + d * math.log(s2 / r2))


def main() -> None:
    try:
        blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(CKPT, map_location="cpu")
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    w2 = 0.0
    d = 0
    for value in state.values():
        if torch.is_floating_point(value):
            flat = value.detach().float()
            w2 += float(flat.pow(2).sum().item())
            d += int(flat.numel())
    meta = json.loads((ROOT / "results" / "camva" / "splits" / "split_metadata.json").read_text(encoding="utf-8"))
    n = int(meta["train"]["n_notes"])
    m = len(SIGMA0) * len(RHO)
    rows = []
    for sigma0 in SIGMA0:
        for rho in RHO:
            kl = _kl(w2, d, sigma0, rho)
            penalty = math.sqrt((kl + math.log(m) + math.log(2 * math.sqrt(n) / DELTA)) / (2 * n))
            rows.append({
                "sigma0": sigma0,
                "rho": rho,
                "kl": kl,
                "penalty": penalty,
                "vacuous": penalty >= 1.0,
            })
    best = min(rows, key=lambda row: row["penalty"])
    payload = {
        "checkpoint": str(CKPT),
        "posterior": "N(w, rho^2 I) centered at the saved weights",
        "prior": "N(0, sigma0^2 I)",
        "n_parameters": d,
        "weight_squared_l2": w2,
        "n_train_notes": n,
        "delta": DELTA,
        "grid_size": m,
        "union_bound": "ln(grid size) added inside the McAllester square root",
        "empirical_gibbs_risk": "NOT_MEASURED",
        "note": "The penalty is the second term of the McAllester bound. If it is at least 1, every bound that adds a risk in [0, 1] exceeds 1.",
        "rows": rows,
        "smallest_penalty": best,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"d": d, "w2": w2, "smallest_penalty": best["penalty"], "vacuous": best["vacuous"]}, indent=2))


if __name__ == "__main__":
    main()
