"""Prefix network (trained on the serial-disjoint split) + watermark window, on unseen counterfeit prints.

Inputs: the prefix-network predictions of results/serial_split/runs/prefix_shbn/s2/{val_views,test_views}
and the watermark classifier of watermark_serial_split.py (refit the same way: TRAIN crops, C on VAL).
HYBRID_k: logistic regression fitted on VAL over [network logit at k views, watermark logit, watermark missing].
TEST read once. Paired exact McNemar against the network alone. Output: results/watermark/hybrid_serial_split.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from roboeye.camva.notes import load_splits  # noqa: E402
from watermark_serial_split import crop_feats, summary  # noqa: E402

RUNS = {42: ROOT / "results/serial_split/runs/prefix_shbn/s2",
        43: ROOT / "results/serial_split/runs/prefix_shbn_seed43/s2",
        44: ROOT / "results/serial_split/runs/prefix_shbn_seed44/s2"}


def preds(run, kind, k):
    p = json.loads((run / kind / f"{k}view" / "test_predictions.json").read_text(encoding="utf-8"))
    return dict(zip(p["note_id"], p["genuine_score"]))


def mcnemar(a, b):
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    n = n01 + n10
    return n01, n10, (1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n))


def main() -> None:
    splits, records = load_splits(ROOT / "results" / "serial_split")
    ids = {s: list(splits[s]) for s in ("train", "val", "test")}
    y = {s: np.array([int(records[n]["label"]) for n in ids[s]]) for s in ids}
    cf = crop_feats()
    has = {s: np.array([n in cf for n in ids[s]]) for s in ids}
    W = {s: np.stack([cf.get(n, np.zeros(2048, np.float32)) for n in ids[s]]) for s in ids}
    best = None
    for c in (0.01, 0.1, 1.0, 10.0, 100.0):
        m = LogisticRegression(C=c, max_iter=3000).fit(W["train"][has["train"]], y["train"][has["train"]])
        a = roc_auc_score(y["val"][has["val"]], m.predict_proba(W["val"][has["val"]])[:, 1])
        if best is None or a > best[0]:
            best = (a, c, m)
    wp = {s: np.where(has[s], best[2].predict_proba(W[s])[:, 1], 0.5) for s in ("val", "test")}
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - np.asarray(p), 1e-6, 1))
    all_seeds = {}
    for seed, run in RUNS.items():
      if not (run / "val_views" / "6view" / "test_predictions.json").is_file():
        print("seed", seed, "not available"); continue
      res = {}
      for k in (1, 6):
        net = {"val": preds(run, "val_views", k), "test": preds(run, "test_views", k)}
        P = {s: np.array([net[s][n] for n in ids[s]]) for s in ("val", "test")}
        F = {s: np.column_stack([lg(P[s]), lg(wp[s]), (~has[s]).astype(float)]) for s in ("val", "test")}
        hy = LogisticRegression(max_iter=2000).fit(F["val"], y["val"])
        ph = hy.predict_proba(F["test"])[:, 1]
        right_h = (ph >= 0.5) == (y["test"] == 1)
        right_n = (P["test"] >= 0.5) == (y["test"] == 1)
        res[f"k{k}"] = {"network": summary(P["test"], y["test"]), "hybrid": summary(ph, y["test"]),
                        "coef [network, watermark, wm_missing]": hy.coef_[0].round(3).tolist(),
                        "mcnemar_hybrid_vs_network (n01 hybrid-only right, n10 network-only right, p)": mcnemar(right_h, right_n)}
        for name in ("network", "hybrid"):
            v = res[f"k{k}"][name]
            print(f"seed {seed} k={k} {name:8s} acc {v['accuracy']:.4f} auc {v['auc']:.4f} FCR {v['false_counterfeit_on_genuine']}/{v['genuine_n']} miss {v['counterfeit_missed']}/{v['counterfeit_n']}")
        print("   McNemar", res[f"k{k}"]["mcnemar_hybrid_vs_network (n01 hybrid-only right, n10 network-only right, p)"])
      all_seeds[seed] = res
      if seed == 42:
        (ROOT / "results/watermark/hybrid_serial_split.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    summ = {}
    for k in ("k1", "k6"):
      for m in ("network", "hybrid"):
        acc = [all_seeds[s][k][m]["accuracy"] for s in all_seeds]
        summ[f"{k}/{m}"] = {"seeds": list(all_seeds), "mean": float(np.mean(acc)), "sd": float(np.std(acc, ddof=1)) if len(acc) > 1 else None,
                            "fcr": [all_seeds[s][k][m]["false_counterfeit_on_genuine"] for s in all_seeds],
                            "missed": [all_seeds[s][k][m]["counterfeit_missed"] for s in all_seeds]}
    (ROOT / "results/watermark/hybrid_serial_split_seeds.json").write_text(json.dumps({"summary": summ, "per_seed": all_seeds}, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
