"""Watermark detector and full-view probe on the SERIAL-DISJOINT split (unseen counterfeit prints).

Split: results/serial_split (make_serial_split.py); TEST counterfeits share no serial print with TRAIN.
  W_deep   logistic regression on frozen ResNet-50 features of the back-lit watermark window
           (crops from watermark_features.py), C chosen on this split's VAL; notes whose view 6 did
           not register are reported separately
  PROBE_k  frozen ResNet-50 features of the first k views (cached by eval_backbone_probes.py),
           one logistic regression on single TRAIN views, C chosen on VAL, mean probability over k
  HYBRID   logistic regression fitted on VAL over [PROBE_1 logit, W_deep logit, W missing]
TEST read once. Crop features are cached in results/watermark/crop_feats_resnet50.npz.
Output: results/watermark/serial_split_eval.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from roboeye.camva.notes import load_splits  # noqa: E402
import eval_backbone_probes as probes  # noqa: E402

WM = ROOT / "results" / "watermark"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def crop_feats() -> dict[str, np.ndarray]:
    cache = WM / "crop_feats_resnet50.npz"
    if cache.is_file():
        d = np.load(cache, allow_pickle=True)
        return dict(zip(d["ids"], d["x"]))
    model, tf = probes.backbone("resnet50")
    rows = [r for r in json.loads((WM / "features.json").read_text(encoding="utf-8")) if r["ok"]]
    ids, xs = [], []
    with torch.inference_mode():
        for r in rows:
            img = cv2.imread(str(WM / "crops" / (r["note_id"].replace(":", "_") + ".png")))
            z = model(tf(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEV)).float().cpu().numpy()[0]
            ids.append(r["note_id"])
            xs.append(z / (np.linalg.norm(z) + 1e-8))
    np.savez(cache, ids=np.array(ids), x=np.stack(xs))
    return dict(zip(ids, xs))


def summary(p, y):
    pred = p >= 0.5
    gen = y == 1
    fcr, miss = int((~pred & gen).sum()), int((pred & ~gen).sum())
    return {"n": int(len(y)), "accuracy": float((pred == gen).mean()), "auc": float(roc_auc_score(gen, p)),
            "false_counterfeit_on_genuine": fcr, "genuine_n": int(gen.sum()), "fcr_wilson95": wilson(fcr, int(gen.sum())),
            "counterfeit_missed": miss, "counterfeit_n": int((~gen).sum()), "miss_wilson95": wilson(miss, int((~gen).sum()))}


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
    wd = best[2]
    wp = {s: np.where(has[s], wd.predict_proba(W[s])[:, 1], 0.5) for s in ids}
    res = {"split": "results/serial_split", "counts": {s: {"notes": len(ids[s]), "counterfeit": int((y[s] == 0).sum()),
                                                            "view6_registered": int(has[s].sum())} for s in ids},
           "W_deep": {"C_on_val": best[1], "test_registered": summary(wp["test"][has["test"]], y["test"][has["test"]])}}
    f = np.load(ROOT / "results/sota/feats_resnet50.npz", allow_pickle=True)
    pos = {n: i for i, n in enumerate(f["ids"])}
    X = {s: f["x"][[pos[n] for n in ids[s]]] for s in ids}
    bestp = None
    for c in (0.01, 0.1, 1.0, 10.0, 100.0):
        m = LogisticRegression(C=c, max_iter=3000).fit(X["train"].reshape(-1, 2048), np.repeat(y["train"], 6))
        pv = m.predict_proba(X["val"].reshape(-1, 2048))[:, 1].reshape(-1, 6)
        a = np.mean([((pv[:, :k].mean(1) >= 0.5) == (y["val"] == 1)).mean() for k in range(1, 7)])
        if bestp is None or a > bestp[0]:
            bestp = (a, c, m)
    pm = bestp[2]
    P = {s: pm.predict_proba(X[s].reshape(-1, 2048))[:, 1].reshape(-1, 6) for s in ids}
    res["PROBE"] = {"C_on_val": bestp[1]}
    for k in (1, 2, 6):
        res["PROBE"][f"k{k}"] = summary(P["test"][:, :k].mean(1), y["test"])
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    Fv = np.column_stack([lg(P["val"][:, 0]), lg(wp["val"]), (~has["val"]).astype(float)])
    Ft = np.column_stack([lg(P["test"][:, 0]), lg(wp["test"]), (~has["test"]).astype(float)])
    hy = LogisticRegression(max_iter=2000).fit(Fv, y["val"])
    res["HYBRID_view1_plus_watermark"] = {"coef": hy.coef_[0].round(3).tolist(), "test": summary(hy.predict_proba(Ft)[:, 1], y["test"])}
    (WM / "serial_split_eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["counts"]))
    for k, v in (("W_deep (registered)", res["W_deep"]["test_registered"]), ("PROBE k1", res["PROBE"]["k1"]), ("PROBE k2", res["PROBE"]["k2"]),
                 ("PROBE k6", res["PROBE"]["k6"]), ("HYBRID v1+wm", res["HYBRID_view1_plus_watermark"]["test"])):
        print(f"{k:22s} acc {v['accuracy']:.4f} auc {v['auc']:.4f} FCR {v['false_counterfeit_on_genuine']}/{v['genuine_n']} miss {v['counterfeit_missed']}/{v['counterfeit_n']}")


if __name__ == "__main__":
    main()
