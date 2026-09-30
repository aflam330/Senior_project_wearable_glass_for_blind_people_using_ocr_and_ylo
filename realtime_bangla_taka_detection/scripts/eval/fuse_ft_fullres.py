"""Precommitted hybrid: prefix network + MobileNetV2 watermark + full-resolution fine-tuned ResNet-50.

The combiner is a logistic regression fitted on VAL only, over
[network logit, watermark logit, watermark-missing flag, fine-tuned ResNet logit].
The published three-feature hybrid is recomputed in the same run as the paired baseline.
Nothing is chosen from TEST. Output: results/serial_split/hybrid_ft_fullres.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from roboeye.camva.notes import load_splits  # noqa: E402
from hybrid_serial_split import RUNS, mcnemar, preds  # noqa: E402
import hybrid_v2_serial_split as hv  # noqa: E402
from watermark_serial_split import summary  # noqa: E402

hv.WM_TAG = "watermark_mobilenetv2"
FT = ROOT / "results" / "serial_split" / "ft_resnet50_fullres"
OUT = ROOT / "results" / "serial_split" / "hybrid_ft_fullres.json"


def lg(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> None:
    splits, records = load_splits(ROOT / "results" / "serial_split")
    ids = {s: list(splits[s]) for s in ("val", "test")}
    y = {s: np.array([int(records[n]["label"]) for n in ids[s]]) for s in ids}
    ok = {r["note_id"] for r in json.loads((hv.WM / "features.json").read_text(encoding="utf-8")) if r["ok"]}
    has = {s: np.array([n in ok for n in ids[s]]) for s in ids}
    wp = {s: hv.wm_scores(ids[s], ok) for s in ids}
    per_seed = {}
    for seed, run in RUNS.items():
        fj = json.loads((FT / f"seed{seed}.json").read_text(encoding="utf-8"))
        ft = {}
        for split, key_ids, key_p in (("val", "val_note_ids", "val_probs"), ("test", "test_note_ids", "test_probs")):
            order = {n: i for i, n in enumerate(fj[key_ids])}
            ft[split] = np.array(fj[key_p], dtype=np.float64)[[order[n] for n in ids[split]]]
        res = {"ft_best_val_mean": fj["best_val_mean"], "ft_val": fj["val"], "ft_test": fj["test"]}
        for k in (1, 6):
            net = {s: np.array([preds(run, f"{s}_views", k)[n] for n in ids[s]]) for s in ids}
            ft_k = {s: ft[s][:, :k].mean(1) for s in ids}
            base = {s: np.column_stack([lg(net[s]), lg(wp[s]), (~has[s]).astype(float)]) for s in ids}
            plus = {s: np.column_stack([base[s], lg(ft_k[s])]) for s in ids}
            row = {}
            ph = {}
            for name, feat in (("hybrid", base), ("hybrid_ft", plus)):
                model = LogisticRegression(max_iter=2000).fit(feat["val"], y["val"])
                prob = {s: model.predict_proba(feat[s])[:, 1] for s in ids}
                ph[name] = prob
                row[name] = {"val": summary(prob["val"], y["val"]), "test": summary(prob["test"], y["test"]),
                             "coef": model.coef_[0].round(4).tolist()}
            right = {name: (ph[name]["test"] >= 0.5) == (y["test"] == 1) for name in ph}
            right["ft"] = (ft_k["test"] >= 0.5) == (y["test"] == 1)
            right["network"] = (net["test"] >= 0.5) == (y["test"] == 1)
            row["network"] = summary(net["test"], y["test"])
            row["ft"] = summary(ft_k["test"], y["test"])
            row["mcnemar_hybrid_ft_vs_hybrid"] = mcnemar(right["hybrid_ft"], right["hybrid"])
            row["mcnemar_hybrid_ft_vs_ft"] = mcnemar(right["hybrid_ft"], right["ft"])
            row["mcnemar_ft_vs_network"] = mcnemar(right["ft"], right["network"])
            res[f"k{k}"] = row
            print(seed, k,
                  "ft", round(row["ft"]["accuracy"], 4),
                  "hybrid", round(row["hybrid"]["test"]["accuracy"], 4),
                  "hybrid_ft", round(row["hybrid_ft"]["test"]["accuracy"], 4),
                  "mcnemar", row["mcnemar_hybrid_ft_vs_hybrid"], flush=True)
        per_seed[seed] = res
    summ = {}
    for k in ("k1", "k6"):
        for m in ("ft", "network"):
            a = [per_seed[s][k][m]["accuracy"] for s in per_seed]
            summ[f"{k}/{m}"] = {"mean": float(np.mean(a)), "sd": float(np.std(a, ddof=1))}
        for m in ("hybrid", "hybrid_ft"):
            a = [per_seed[s][k][m]["test"]["accuracy"] for s in per_seed]
            summ[f"{k}/{m}"] = {"mean": float(np.mean(a)), "sd": float(np.std(a, ddof=1))}
        summ[f"{k}/mcnemar_hybrid_ft_vs_hybrid_per_seed"] = [
            list(per_seed[s][k]["mcnemar_hybrid_ft_vs_hybrid"]) for s in per_seed]
    OUT.write_text(json.dumps({"summary": summ, "per_seed": {str(s): per_seed[s] for s in per_seed}}, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
