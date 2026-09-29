"""Hybrid v2 on unseen counterfeit prints: prefix network + MobileNet watermark model + rejection.

Serial-disjoint split (results/serial_split), seeds 42-44 of the prefix network.
Watermark score: MobileNetV3-Small (models/watermark_mobilenet.pt; trained on this split's TRAIN crops,
epoch chosen on VAL AUC). Notes whose view 6 did not register get a neutral score and a missing flag.
HYBRID_k: logistic regression fitted on VAL over [network logit (k views), watermark logit, missing flag].
Rejection (rule fixed before test): answer only if max(p, 1-p) >= t, where t is the smallest value on
0.50..0.99 (step 0.01) whose VAL error among answered notes is <= 1 %; otherwise "check by hand".
TEST read once per seed. Output: results/watermark/hybrid_v2_seeds.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torchvision import models

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
from roboeye.camva.notes import load_splits  # noqa: E402
from hybrid_serial_split import RUNS, mcnemar, preds  # noqa: E402
from train_watermark_mobilenet import EV, WM  # noqa: E402
from watermark_serial_split import summary  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# watermark model: argv[1] = "watermark_mobilenet" (V3) or "watermark_mobilenetv2" (V2, higher VAL AUC: the final choice)
WM_TAG = sys.argv[1] if len(sys.argv) > 1 else "watermark_mobilenet"
OUT_NAME = "hybrid_v2_seeds.json" if WM_TAG == "watermark_mobilenet" else "hybrid_final_seeds.json"


@torch.inference_mode()
def wm_scores(ids, ok):
    blob = torch.load(ROOT / f"models/{WM_TAG}.pt", map_location="cpu", weights_only=False)
    if blob["arch"] == "mobilenet_v2":
        m = models.mobilenet_v2()
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    else:
        m = models.mobilenet_v3_small()
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    m.load_state_dict(blob["state_dict"])
    m = m.eval().to(DEV)
    out = np.full(len(ids), 0.5)
    for i, n in enumerate(ids):
        if n in ok:
            x = EV(Image.open(WM / "crops" / (n.replace(":", "_") + ".png")).convert("RGB")).unsqueeze(0).to(DEV)
            out[i] = float(torch.softmax(m(x), 1)[0, 1])
    return out


def main() -> None:
    splits, records = load_splits(ROOT / "results" / "serial_split")
    ids = {s: list(splits[s]) for s in ("val", "test")}
    y = {s: np.array([int(records[n]["label"]) for n in ids[s]]) for s in ids}
    ok = {r["note_id"] for r in json.loads((WM / "features.json").read_text(encoding="utf-8")) if r["ok"]}
    has = {s: np.array([n in ok for n in ids[s]]) for s in ids}
    wp = {s: wm_scores(ids[s], ok) for s in ids}
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - np.asarray(p), 1e-6, 1))
    per_seed = {}
    for seed, run in RUNS.items():
        res = {}
        for k in (1, 6):
            P = {s: np.array([preds(run, f"{s}_views", k)[n] for n in ids[s]]) for s in ids}
            F = {s: np.column_stack([lg(P[s]), lg(wp[s]), (~has[s]).astype(float)]) for s in ids}
            hy = LogisticRegression(max_iter=2000).fit(F["val"], y["val"])
            ph = {s: hy.predict_proba(F[s])[:, 1] for s in ids}
            conf_v = np.maximum(ph["val"], 1 - ph["val"])
            right_v = (ph["val"] >= 0.5) == (y["val"] == 1)
            t = next((round(t, 2) for t in np.arange(0.50, 0.995, 0.01)
                      if (conf_v >= t).any() and (~right_v[conf_v >= t]).mean() <= 0.01), 0.99)
            conf_t = np.maximum(ph["test"], 1 - ph["test"])
            ans = conf_t >= t
            right_t = (ph["test"] >= 0.5) == (y["test"] == 1)
            gen = y["test"] == 1
            res[f"k{k}"] = {
                "network": summary(P["test"], y["test"]), "hybrid": summary(ph["test"], y["test"]),
                "mcnemar_hybrid_vs_network": mcnemar(right_t, (P["test"] >= 0.5) == gen),
                "rejection": {"threshold_from_val": float(t), "answered": int(ans.sum()), "n": int(len(ans)),
                              "wrong_among_answered": int((ans & ~right_t).sum()),
                              "genuine_called_counterfeit_answered": int((ans & ~(ph["test"] >= 0.5) & gen).sum()),
                              "counterfeit_passed_answered": int((ans & (ph["test"] >= 0.5) & ~gen).sum()),
                              "counterfeit_n": int((~gen).sum()), "genuine_n": int(gen.sum())}}
            h, r = res[f"k{k}"]["hybrid"], res[f"k{k}"]["rejection"]
            print(f"seed {seed} k={k} hybrid acc {h['accuracy']:.4f} FCR {h['false_counterfeit_on_genuine']}/{h['genuine_n']} "
                  f"miss {h['counterfeit_missed']}/{h['counterfeit_n']} | reject t={r['threshold_from_val']} answered {r['answered']}/{r['n']} "
                  f"wrong {r['wrong_among_answered']} | McNemar {res[f'k{k}']['mcnemar_hybrid_vs_network']}", flush=True)
        per_seed[seed] = res
    summ = {}
    for k in ("k1", "k6"):
        for m in ("network", "hybrid"):
            a = [per_seed[s][k][m]["accuracy"] for s in per_seed]
            summ[f"{k}/{m}"] = {"mean": float(np.mean(a)), "sd": float(np.std(a, ddof=1))}
        rj = [per_seed[s][k]["rejection"] for s in per_seed]
        summ[f"{k}/rejection"] = {"answered_mean_share": float(np.mean([r["answered"] / r["n"] for r in rj])),
                                  "wrong_among_answered": [r["wrong_among_answered"] for r in rj],
                                  "answered": [r["answered"] for r in rj]}
    (WM / OUT_NAME).write_text(json.dumps({"summary": summ, "per_seed": per_seed}, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
