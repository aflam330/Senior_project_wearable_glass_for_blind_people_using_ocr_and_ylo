"""Prefix-only stopping on the saved PRMVT checkpoint.

The threshold grid is applied on validation. Test is scored once per
pre-specified lambda. View order is the file order, never a learned permutation.
"""
from __future__ import annotations

import json
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

LAMBDAS = (0.0, 0.01, 0.02, 0.05, 0.1)
PRIMARY_LAMBDA = 0.02
OUT = ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "prefix_stop_policy.json"


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _collect(model, ids, records) -> dict[str, np.ndarray]:
    loader = make_loader(ids, records, n_views=6, train=False, batch=8, workers=0)
    ys, probs, ents = [], {k: [] for k in range(1, 7)}, {k: [] for k in range(1, 7)}
    for batch in loader:
        views = batch["views"].to(DEVICE)
        ys.append(batch["label"].numpy())
        for k in range(1, 7):
            v = views[:, :k]
            mask = torch.ones(v.size(0), k, dtype=torch.long, device=DEVICE)
            with torch.inference_mode():
                out = model(v, mask)
                p = torch.softmax(out["logits"], dim=1)[:, 1].detach().cpu().numpy()
            probs[k].append(p)
            ents[k].append(_entropy(p))
    return {
        "y": np.concatenate(ys),
        "prob": np.stack([np.concatenate(probs[k]) for k in range(1, 7)], axis=1),
        "entropy": np.stack([np.concatenate(ents[k]) for k in range(1, 7)], axis=1),
    }


def _apply(pack: dict[str, np.ndarray], tau: float) -> dict[str, float]:
    ent = pack["entropy"]
    chosen = np.full(len(pack["y"]), 5, dtype=np.int64)
    for k in range(6):
        take = (chosen == 5) & (ent[:, k] <= tau)
        chosen[take] = k
    pred = (pack["prob"][np.arange(len(chosen)), chosen] >= 0.5).astype(np.int64)
    views = chosen + 1
    return {
        "tau": float(tau),
        "accuracy": float(np.mean(pred == pack["y"])),
        "mean_views": float(np.mean(views)),
        "n": int(len(pack["y"])),
    }


def _fixed(pack: dict[str, np.ndarray], k: int) -> dict[str, float]:
    pred = (pack["prob"][:, k - 1] >= 0.5).astype(np.int64)
    return {
        "rule": f"always_prefix_{k}",
        "accuracy": float(np.mean(pred == pack["y"])),
        "mean_views": float(k),
        "n": int(len(pack["y"])),
    }


def _candidates(val: dict[str, np.ndarray]) -> list[float]:
    flat = val["entropy"].reshape(-1)
    qs = np.quantile(flat, [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9])
    return sorted(set([0.0, float(np.log(2.0)), *[float(q) for q in qs]]))


def _pick(val_rows: list[dict], lamb: float) -> dict:
    return max(val_rows, key=lambda row: (row["accuracy"] - lamb * row["mean_views"], -row["mean_views"]))


def main() -> None:
    splits, records = load_splits()
    model = load_qduig(
        ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "checkpoint.pt",
        load_config(ROOT / "configs" / "proposed_prefix_ft.yaml"),
    )
    model.eval()
    val = _collect(model, splits["val"], records)
    test = _collect(model, splits["test"], records)
    taus = _candidates(val)
    val_rows = [_apply(val, tau) for tau in taus]
    test_by_tau = {row["tau"]: _apply(test, row["tau"]) for row in val_rows}
    by_lambda = []
    for lamb in LAMBDAS:
        chosen = _pick(val_rows, lamb)
        tested = test_by_tau[chosen["tau"]]
        by_lambda.append({
            "lambda": lamb,
            "val": chosen,
            "test": tested,
            "primary": lamb == PRIMARY_LAMBDA,
        })
    payload = {
        "checkpoint": "results/qduig/prefix_ft/seed42/checkpoint.pt",
        "selection": "validation accuracy minus lambda times mean views; test scored after selection",
        "primary_lambda": PRIMARY_LAMBDA,
        "always_prefix_1": {"val": _fixed(val, 1), "test": _fixed(test, 1)},
        "always_prefix_3": {"val": _fixed(val, 3), "test": _fixed(test, 3)},
        "lambda_grid": by_lambda,
        "oracle_test": {
            "source": "results/qduig/eval/seed42/oracle.json",
            "note": "analysis only; not used to choose this policy",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    primary = next(row for row in by_lambda if row["primary"])
    print(json.dumps({"primary_test": primary["test"], "always_1_test": payload["always_prefix_1"]["test"]}, indent=2))


if __name__ == "__main__":
    main()
