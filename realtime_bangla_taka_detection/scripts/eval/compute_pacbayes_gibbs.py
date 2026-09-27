"""Gibbs risk of N(w, rho^2 I) on the training split, prior centered at initialization.

The test set is not loaded. CUDA is used for the forwards.
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
from roboeye.qduig.engine import build_model, load_qduig

CKPT = ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "checkpoint.pt"
CFG = ROOT / "configs" / "proposed_prefix_ft.yaml"
OUT = ROOT / "results" / "theory" / "pacbayes_gibbs_seed42.json"
DELTA = 0.05
SIGMAS = (3.0, 5.0, 10.0)
SAMPLES = 4


def penalty(delta2: float, sigma0: float, n: int) -> tuple[float, float]:
    kl = 0.5 * delta2 / (sigma0 * sigma0)
    extra = math.log(2 * math.sqrt(n) / DELTA)
    return kl, math.sqrt((kl + extra) / (2 * n))


def main() -> None:
    cfg = load_config(CFG)
    init_model = build_model(cfg)
    init = {k: v.detach().float().cpu().clone() for k, v in init_model.state_dict().items() if torch.is_floating_point(v)}
    del init_model
    model = load_qduig(CKPT, cfg).to(DEVICE)
    model.eval()
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}
    delta2 = 0.0
    for key, w0 in init.items():
        if key in base and torch.is_floating_point(base[key]) and base[key].shape == w0.shape:
            delta2 += float((base[key].detach().float().cpu() - w0).pow(2).sum().item())
    splits, records = load_splits()
    loader = make_loader(splits["train"], records, n_views=6, train=False, batch=8, workers=0)
    meta = json.loads((ROOT / "results" / "camva" / "splits" / "split_metadata.json").read_text(encoding="utf-8"))
    n = int(meta["train"]["n_notes"])
    rows = []
    g = torch.Generator(device="cpu")
    for sigma in SIGMAS:
        errors = []
        for sample in range(SAMPLES):
            g.manual_seed(42 + sample * 17 + int(sigma * 10))
            noisy = {}
            for key, value in base.items():
                if torch.is_floating_point(value):
                    noise = torch.randn(value.shape, generator=g, dtype=torch.float32) * sigma
                    noisy[key] = (value.detach().float().cpu() + noise).to(device=value.device, dtype=value.dtype)
                else:
                    noisy[key] = value
            model.load_state_dict(noisy)
            wrong = 0
            seen = 0
            for batch in loader:
                views = batch["views"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)
                y = batch["label"].numpy()
                with torch.inference_mode():
                    prob = torch.softmax(model(views, mask)["logits"], dim=1)[:, 1]
                pred = (prob >= 0.5).detach().cpu().numpy()
                wrong += int(np.sum(pred != y))
                seen += len(y)
            errors.append(wrong / seen)
            print(f"sigma {sigma} sample {sample} err {errors[-1]:.4f}", flush=True)
        emp = float(sum(errors) / len(errors))
        kl, pen = penalty(delta2, sigma, n)
        rows.append({
            "sigma0": sigma,
            "rho": sigma,
            "samples": SAMPLES,
            "gibbs_errors": errors,
            "empirical_gibbs_risk": emp,
            "kl": kl,
            "penalty": pen,
            "mcallester": emp + pen,
            "below_1": (emp + pen) < 1.0,
        })
    model.load_state_dict(base)
    payload = {
        "split": "train",
        "n_train_notes": n,
        "test_used": False,
        "prior": "N(w_init, sigma0^2 I)",
        "posterior": "N(w, sigma0^2 I), rho = sigma0",
        "displacement_squared_l2": delta2,
        "delta": DELTA,
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps([{k: r[k] for k in ("sigma0", "empirical_gibbs_risk", "penalty", "mcallester")} for r in rows], indent=2))


if __name__ == "__main__":
    main()
