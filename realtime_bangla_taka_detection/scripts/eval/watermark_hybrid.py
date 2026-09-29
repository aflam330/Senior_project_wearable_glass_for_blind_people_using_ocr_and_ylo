"""Watermark + serial + PRMVT hybrid counterfeit detector on JaalTaka (seed-42 split). Test read once.

Components (all fitted on TRAIN, combined on VAL):
  W_hand   logistic regression on the 4 hand-crafted watermark features (watermark_features.py)
  W_deep   logistic regression on frozen ResNet-50 features of the watermark crop; C chosen on VAL
  S_list   serial blacklist: the note's serial (first 6 digits) occurs among TRAIN counterfeits
  S_dup    bundle anomaly: the full serial occurs on >= 2 different TRAIN notes
  PRMVT    p(genuine), prefix_ft seed 42, first k views (k = 1 and 6)
  HYBRID_k logistic regression fitted on VAL over [logit PRMVT_k, W_deep logit, W missing flag, S_list]
Reported on TEST: accuracy, false-counterfeit rate on genuine (FCR), counterfeit miss rate, and the
catch rate on counterfeits whose serial is unseen in TRAIN (results/jaal_whole/serial_split.json).
A note whose view 6 did not register has no watermark score (flag = 1, logit = 0).
Output: results/watermark/hybrid.json
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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from roboeye.authenticity import default_transform  # noqa: E402
from roboeye.camva.notes import SPLIT_DIR, load_splits  # noqa: E402
from roboeye.qduig.engine import load_qduig  # noqa: E402
import eval_backbone_probes as probes  # noqa: E402

WM = ROOT / "results" / "watermark"
PRMVT = ROOT / "results/qduig/prefix_ft/seed42"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HAND = ["lap_var", "std", "edges", "rel_mean"]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


@torch.inference_mode()
def prmvt_scores(ids, records, k):
    net = load_qduig(PRMVT / "checkpoint.pt").to(DEV).eval()
    tf = default_transform(train=False)
    out = {}
    for nid in ids:
        v = torch.stack([tf(Image.fromarray(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))) for p in records[nid]["view_paths"][:k]])
        out[nid] = float(net(v.unsqueeze(0).to(DEV), torch.ones(1, k, dtype=torch.long, device=DEV))["prob"].reshape(-1)[0])
    return out


def report(p_gen, ids, records, unseen):
    y = np.array([int(records[n]["label"]) for n in ids])
    pred = np.asarray(p_gen) >= 0.5
    gen, cf = y == 1, y == 0
    fcr = int((~pred & gen).sum())
    miss = int((pred & cf).sum())
    un = np.array([n in unseen for n in ids])
    return {"n": len(ids), "accuracy": float((pred == gen).mean()),
            "auc": float(roc_auc_score(gen, p_gen)) if 0 < gen.sum() < len(gen) else None,
            "false_counterfeit_on_genuine": fcr, "genuine_n": int(gen.sum()), "fcr_wilson95": wilson(fcr, int(gen.sum())),
            "counterfeit_missed": miss, "counterfeit_n": int(cf.sum()),
            "unseen_serial_counterfeit_caught": int((~pred & cf & un).sum()), "unseen_serial_counterfeit_n": int((cf & un).sum())}


def main() -> None:
    splits, records = load_splits(SPLIT_DIR)
    feats = {r["note_id"]: r for r in json.loads((WM / "features.json").read_text(encoding="utf-8"))}
    audit = {r["note_id"]: r for r in json.loads((ROOT / "results/jaal_whole/serial_audit.json").read_text(encoding="utf-8"))["rows"]}
    unseen = set(json.loads((ROOT / "results/jaal_whole/serial_split.json").read_text(encoding="utf-8"))["unseen_note_ids"])
    ids = {s: list(splits[s]) for s in ("train", "val", "test")}
    y = {s: np.array([int(records[n]["label"]) for n in ids[s]]) for s in ids}
    res = {"registered_view6": {s: sum(feats.get(n, {}).get("ok", False) for n in ids[s]) for s in ids}}

    # --- watermark, hand-crafted ---
    def hand(s):
        return np.array([[feats[n][f] if feats.get(n, {}).get("ok") else np.nan for f in HAND] for n in ids[s]])
    ok = {s: ~np.isnan(hand(s)).any(1) for s in ids}
    sc = StandardScaler().fit(hand("train")[ok["train"]])
    wh = LogisticRegression(max_iter=2000).fit(sc.transform(hand("train")[ok["train"]]), y["train"][ok["train"]])
    res["W_hand"] = {s: {"n_registered": int(ok[s].sum()),
                         "auc": float(roc_auc_score(y[s][ok[s]], wh.predict_proba(sc.transform(hand(s)[ok[s]]))[:, 1])),
                         "coef": dict(zip(HAND, wh.coef_[0].round(3).tolist()))} for s in ("val", "test")}

    # --- watermark, deep ---
    model, tf = probes.backbone("resnet50")
    def deep(s):
        out = np.zeros((len(ids[s]), 2048), np.float32)
        with torch.inference_mode():
            for i, n in enumerate(ids[s]):
                if ok[s][i]:
                    img = cv2.imread(str(WM / "crops" / (n.replace(":", "_") + ".png")))
                    z = model(tf(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEV)).float().cpu().numpy()[0]
                    out[i] = z / (np.linalg.norm(z) + 1e-8)
        return out
    X = {s: deep(s) for s in ids}
    best = None
    for c in (0.01, 0.1, 1.0, 10.0, 100.0):
        m = LogisticRegression(C=c, max_iter=3000).fit(X["train"][ok["train"]], y["train"][ok["train"]])
        a = roc_auc_score(y["val"][ok["val"]], m.predict_proba(X["val"][ok["val"]])[:, 1])
        if best is None or a > best[0]:
            best = (a, c, m)
    wd = best[2]
    wscore = {s: np.where(ok[s], wd.predict_proba(X[s])[:, 1], 0.5) for s in ids}
    res["W_deep"] = {"C_on_val": best[1]}
    for s in ("val", "test"):
        r = report(wscore[s][ok[s]], [n for n, o in zip(ids[s], ok[s]) if o], records, unseen)
        res["W_deep"][s] = r

    # --- serial ---
    ser = lambda n: (audit.get(n) or {}).get("serial")
    list_cf = {ser(n)[:6] for n in ids["train"] if int(records[n]["label"]) == 0 and ser(n)}
    from collections import Counter
    cnt = Counter(ser(n) for n in ids["train"] if ser(n))
    dup = {s for s, c in cnt.items() if c >= 2}
    s_list = {s: np.array([bool(ser(n)) and ser(n)[:6] in list_cf for n in ids[s]]) for s in ids}
    s_dup = {s: np.array([bool(ser(n)) and ser(n) in dup for n in ids[s]]) for s in ids}
    res["S_list"] = {"test": report(np.where(s_list["test"], 0.0, 1.0), ids["test"], records, unseen)}
    res["S_dup"] = {"test": report(np.where(s_dup["test"], 0.0, 1.0), ids["test"], records, unseen),
                    "train_duplicate_serials": len(dup)}

    # --- PRMVT and hybrid ---
    for k in (1, 6):
        pv = prmvt_scores(ids["val"], records, k)
        pt = json.loads((PRMVT / f"test_views_20260928/{k}view/test_predictions.json").read_text(encoding="utf-8"))
        pt = dict(zip(pt["note_id"], pt["genuine_score"]))
        P = {"val": np.array([pv[n] for n in ids["val"]]), "test": np.array([pt[n] for n in ids["test"]])}
        res[f"PRMVT_k{k}"] = {"test": report(P["test"], ids["test"], records, unseen)}
        F = {s: np.column_stack([logit(P[s]), logit(wscore[s]), (~ok[s]).astype(float), s_list[s].astype(float)]) for s in ("val", "test")}
        hyb = LogisticRegression(C=1.0, max_iter=2000).fit(F["val"], y["val"])
        ph = hyb.predict_proba(F["test"])[:, 1]
        res[f"HYBRID_k{k}"] = {"fitted_on": "val", "coef [prmvt, watermark, wm_missing, serial_list]": hyb.coef_[0].round(3).tolist(),
                               "test": report(ph, ids["test"], records, unseen)}
        # without the serial list (for deployment on prints never seen before)
        F2 = {s: F[s][:, :3] for s in F}
        hyb2 = LogisticRegression(C=1.0, max_iter=2000).fit(F2["val"], y["val"])
        res[f"HYBRID_NO_SERIAL_k{k}"] = {"test": report(hyb2.predict_proba(F2["test"])[:, 1], ids["test"], records, unseen)}
    (WM / "hybrid.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, v in res.items():
        if isinstance(v, dict) and isinstance(v.get("test"), dict):
            t = v["test"]
            print(f"{k:22s} acc {t['accuracy']:.4f} auc {t['auc']} FCR {t['false_counterfeit_on_genuine']}/{t['genuine_n']} "
                  f"miss {t['counterfeit_missed']}/{t['counterfeit_n']} unseen-caught {t['unseen_serial_counterfeit_caught']}/{t['unseen_serial_counterfeit_n']}")
    print("W_hand", res["W_hand"], "registered", res["registered_view6"])


if __name__ == "__main__":
    main()
