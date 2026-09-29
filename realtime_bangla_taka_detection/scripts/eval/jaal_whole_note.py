"""Whole-note counterfeit check for the glass: view synthesis, probes, shortcut test, safe policy.

Scope: 500 and 1000 BDT, the only denominations with counterfeit data on disk.

Inputs (whole-note photos; none used to train PRMVT or the ResNet-50 probe)
  cf  data set/Bangladeshi Counterfeit Currency Image Dataset (genuine + counterfeit)
  bm  Bangla Money (Kaggle) Training/500, /1000 (genuine)
  bt  BanglaTaka raw 500, 1000 (genuine)
Every photo goes through the app's Taka YOLO; the top box is cropped, resized to 640 px wide and
re-encoded as JPEG (so file format and resolution do not reach the checkers).

Checkers
  S0     deployed PRMVT, whole crop as 1 view (the configuration switched off on 2026-09-29)
  S1-S4  PRMVT on views 1..k cut from the crop with the windows measured on JaalTaka TRAIN notes
         (results/jaal_whole/view_geometry.json)
  R1-R4  frozen ResNet-50 + logistic regression fitted on JaalTaka TRAIN single views
         (C = 100, chosen on JaalTaka VAL in eval_backbone_probes.py), same cut views, mean prob
  B      whole-crop ResNet-50 + logistic regression trained on the counterfeit set itself,
         leave-one-counterfeit-group-out (all genuine bursts split 50/50 by burst)
  META   logistic regression on file metadata only (size, aspect, format, bytes/pixel),
         leave-one-group-out: measures how separable the counterfeit set is without the note

Grouping of the counterfeit set (it has no note IDs): phone photos split into bursts where the
filename timestamps are more than 3 s apart; the two 1000 BDT screenshots and all 1000 BDT
augmented_* copies form one group (same stamped note, same 1118 x 781 size); the 500 BDT NoteN
photos and their augmented_* copies form one group (one serial number).

Policy E (the one that can ship): never say "counterfeit". Say "likely genuine" only if the
score is >= tau, else "check by hand". tau is fixed on JaalTaka VAL notes (real views 1..k) as the
highest score of any VAL counterfeit note, so no VAL counterfeit would be passed.

Output: results/jaal_whole/{items.json, scores.json, summary.json}
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORK / "savior_glass"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from roboeye.authenticity import default_transform  # noqa: E402
from roboeye.camva.notes import load_splits  # noqa: E402
from roboeye.qduig.engine import load_qduig  # noqa: E402
import eval_backbone_probes as probes  # noqa: E402

OUT = ROOT / "results" / "jaal_whole"
CF = WORK / "data set" / "Bangladeshi Counterfeit Currency Image Dataset" / "Denomination-wise Distribution"
BM = WORK / "data set for comparison" / "Bangla Money dataset Kaggle" / "bangla_banknote_v2" / "Training"
BT = WORK / "data set" / "Bangladeshi_Paper_Currency_Raw" / "Bangladeshi_Paper_Currency_Raw"
PRMVT = ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "checkpoint.pt"
IMG_EXT = {".jpg", ".jpeg", ".png"}
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def stamp(name: str) -> tuple[str, float] | None:
    m = re.match(r"(?:PXL|IMG)_(\d{8})_(\d{6})(\d*)", name)
    if not m:
        return None
    d, t, ms = m.groups()
    return d, int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6]) + (int(ms[:3]) / 1000 if ms else 0.0)


def collect() -> list[dict]:
    items = []
    for folder in sorted(p for p in CF.iterdir() if p.is_dir()):
        denom = folder.name.split()[0]
        truth = "counterfeit" if "Counterfeit" in folder.name else "genuine"
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)
        timed = sorted((stamp(p.name), p) for p in files if stamp(p.name))
        prev, g = None, -1
        group_of = {}
        for (d, s), p in timed:
            if prev is None or d != prev[0] or s - prev[1] > 3:
                g += 1
            group_of[p] = f"cf{denom}_{truth}_burst{g:03d}"
            prev = (d, s)
        for p in files:
            if p in group_of:
                grp = group_of[p]
            elif denom == "1000" and truth == "counterfeit":
                grp = "cf1000_counterfeit_stamped"  # screenshots + augmented copies
            elif denom == "500" and truth == "counterfeit":
                grp = "cf500_counterfeit_notes"  # NoteN + augmented copies
            else:
                grp = f"cf{denom}_{truth}_single_{p.stem}"
            items.append({"set": "cf", "path": p, "denom": denom, "truth": truth, "group": grp,
                          "augmented": p.name.startswith("augmented")})
    for denom in ("500", "1000"):
        for p in sorted((BM / denom).iterdir()):
            if p.suffix.lower() in IMG_EXT:
                items.append({"set": "bm", "path": p, "denom": denom, "truth": "genuine", "group": f"bm_{p.stem}", "augmented": False})
        for p in sorted((BT / denom).iterdir()):
            if p.suffix.lower() in IMG_EXT:
                items.append({"set": "bt", "path": p, "denom": denom, "truth": "genuine", "group": f"bt_{p.stem}", "augmented": False})
    return items


def cut_views(crop: np.ndarray, geom: dict) -> list[np.ndarray]:
    h, w = crop.shape[:2]
    out = []
    for k in ("1", "2", "3", "4"):
        x0, y0, x1, y1 = geom["views"][k]["median_xyxy"]
        out.append(crop[int(y0 * h):max(int(y1 * h), int(y0 * h) + 2), int(x0 * w):max(int(x1 * w), int(x0 * w) + 2)])
    return out


@torch.inference_mode()
def prmvt_prob(net, tf, views: list[np.ndarray]) -> float:
    x = torch.stack([tf(Image.fromarray(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))) for v in views]).unsqueeze(0).to(DEV)
    m = torch.ones(1, len(views), dtype=torch.long, device=DEV)
    return float(net(x, m)["prob"].reshape(-1)[0].item())


@torch.inference_mode()
def resnet_feats(model, tf, imgs: list[np.ndarray]) -> np.ndarray:
    x = torch.stack([tf(Image.fromarray(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))) for v in imgs]).to(DEV)
    z = model(x).float().cpu().numpy()
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)


def jaaltaka_val_tau(net, tf, lr, rmodel, rtf) -> dict:
    """Highest VAL-counterfeit score per checker, from real JaalTaka views 1..k (no whole-note data)."""
    splits, records = load_splits()
    taus = defaultdict(float)
    for nid in splits["val"]:
        r = records[nid]
        if int(r["label"]) == 1:
            continue
        views = [cv2.imread(p) for p in r["view_paths"][:4]]
        feats = resnet_feats(rmodel, rtf, views)
        pr = lr.predict_proba(feats)[:, 1]
        for k in range(1, 5):
            taus[f"S{k}"] = max(taus[f"S{k}"], prmvt_prob(net, tf, views[:k]))
            taus[f"R{k}"] = max(taus[f"R{k}"], float(pr[:k].mean()))
    return dict(taus)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geom = json.loads((OUT / "view_geometry.json").read_text(encoding="utf-8"))
    from modes.currency_mode import CurrencyMode
    mode = CurrencyMode()
    mode._load_yolo()
    net = load_qduig(PRMVT).to(DEV).eval()
    tf = default_transform(train=False)
    rmodel, rtf = probes.backbone("resnet50")

    # ResNet-50 probe fitted on JaalTaka TRAIN single views (cached features), C from VAL
    splits, records = load_splits()
    cache = np.load(ROOT / "results" / "sota" / "feats_resnet50.npz", allow_pickle=True)
    ids = list(cache["ids"])
    pos = {n: i for i, n in enumerate(ids)}
    tr = [pos[n] for n in splits["train"]]
    xtr = cache["x"][tr].reshape(-1, cache["x"].shape[2])
    ytr = np.repeat([int(records[n]["label"]) for n in splits["train"]], 6)
    lr = LogisticRegression(C=100.0, max_iter=3000).fit(xtr, ytr)

    tau = jaaltaka_val_tau(net, tf, lr, rmodel, rtf)
    print("tau from JaalTaka VAL", {k: round(v, 4) for k, v in tau.items()}, flush=True)

    items = collect()
    rows = []
    for i, it in enumerate(items):
        p = it["path"]
        raw = cv2.imread(str(p))
        row = {k: (str(v.relative_to(WORK)) if k == "path" else v) for k, v in it.items()}
        row["meta"] = [math.log(raw.shape[1]), math.log(raw.shape[0]), raw.shape[1] / raw.shape[0],
                       float(p.suffix.lower() == ".png"), p.stat().st_size / (raw.shape[0] * raw.shape[1])]
        hits = mode.detect_live(raw)
        if not hits:
            row["yolo"] = None
            rows.append(row)
            continue
        x, y, w, h = hits[0]["bbox"]
        crop = raw[max(0, y):y + h, max(0, x):x + w]
        s = 640 / crop.shape[1]
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        crop = cv2.imdecode(cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR)
        if crop.shape[0] > crop.shape[1]:  # portrait note: rotate so the long side is horizontal
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        row["yolo"] = hits[0]["name"]
        views = cut_views(crop, geom)
        row["S0"] = prmvt_prob(net, tf, [crop])
        for k in range(1, 5):
            row[f"S{k}"] = prmvt_prob(net, tf, views[:k])
        f = resnet_feats(rmodel, rtf, views + [crop])
        pr = lr.predict_proba(f[:4])[:, 1]
        for k in range(1, 5):
            row[f"R{k}"] = float(pr[:k].mean())
        row["whole_feat"] = f[4].round(5).tolist()
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(i + 1, "/", len(items), flush=True)

    # --- B: whole-crop probe trained on the counterfeit set, leave-one-counterfeit-group-out ---
    cf = [r for r in rows if r["set"] == "cf" and r.get("yolo")]
    gen_groups = sorted({r["group"] for r in cf if r["truth"] == "genuine"})
    rng = np.random.default_rng(42)
    rng.shuffle(gen_groups)
    half = set(gen_groups[: len(gen_groups) // 2])
    cf_groups = sorted({r["group"] for r in cf if r["truth"] == "counterfeit"})
    scores = {"B": defaultdict(list), "META": defaultdict(list)}
    for held in cf_groups:
        for fold_gen in (half, set(gen_groups) - half):  # each genuine burst is scored when its half is held out
            train = [r for r in cf if (r["truth"] == "counterfeit" and r["group"] != held)
                     or (r["truth"] == "genuine" and r["group"] not in fold_gen)]
            test = [r for r in cf if r["group"] == held or (r["truth"] == "genuine" and r["group"] in fold_gen)]
            y = np.array([r["truth"] == "genuine" for r in train], dtype=int)
            wts = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
            for name, key in (("B", "whole_feat"), ("META", "meta")):
                clf = LogisticRegression(C=1.0, max_iter=3000).fit(np.array([r[key] for r in train]), y, sample_weight=wts)
                for r, v in zip(test, clf.predict_proba(np.array([r[key] for r in test]))[:, 1]):
                    scores[name][r["path"]].append(float(v))
    for r in cf:  # counterfeit: scored twice (same held-out group); genuine: once per counterfeit group
        for name in scores:
            r[name] = float(np.mean(scores[name][r["path"]]))
    # B on independent genuine sets: one model on the whole counterfeit set
    y = np.array([r["truth"] == "genuine" for r in cf], dtype=int)
    wts = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(np.array([r["whole_feat"] for r in cf]), y, sample_weight=wts)
    for r in rows:
        if r["set"] != "cf" and r.get("yolo"):
            r["B"] = float(clf.predict_proba(np.array([r["whole_feat"]]))[0, 1])

    # --- summary ---
    checkers = ["S0", "S1", "S2", "S3", "S4", "R1", "R2", "R3", "R4", "B", "META"]
    summ = {"tau_from_jaaltaka_val": tau, "view_geometry": geom["views"], "sets": {}, "auc_cf_originals": {}, "policy_E": {}}
    for setname in ("cf", "bm", "bt"):
        for truth in ("genuine", "counterfeit"):
            rs = [r for r in rows if r["set"] == setname and r["truth"] == truth]
            if not rs:
                continue
            det = [r for r in rs if r.get("yolo")]
            ent = {"n_images": len(rs), "detected": len(det),
                   "groups": len({r["group"] for r in rs}),
                   "augmented": sum(r["augmented"] for r in rs)}
            for c in checkers:
                v = [r[c] for r in det if r.get(c) is not None]
                if not v:
                    continue
                wrong = sum((x < 0.5) if truth == "genuine" else (x >= 0.5) for x in v)
                ent[c] = {"n": len(v), "median_p_genuine": float(np.median(v)),
                          "wrong_at_0.5": wrong, "wrong_rate": wrong / len(v), "wilson95": wilson(wrong, len(v))}
            summ["sets"][f"{setname}/{truth}"] = ent
    orig = [r for r in rows if r["set"] == "cf" and r.get("yolo") and not r["augmented"]]
    yy = [r["truth"] == "genuine" for r in orig]
    for c in checkers:
        v = [r.get(c) for r in orig]
        if all(x is not None for x in v) and 0 < sum(yy) < len(yy):
            summ["auc_cf_originals"][c] = float(roc_auc_score(yy, v))
    summ["auc_cf_originals_n"] = {"genuine": sum(yy), "counterfeit": len(yy) - sum(yy),
                                  "counterfeit_groups": len({r["group"] for r in orig if r["truth"] == "counterfeit"})}
    for c in [k for k in tau]:
        pol = {"tau": tau[c]}
        for setname, truth in (("cf", "counterfeit"), ("cf", "genuine"), ("bm", "genuine"), ("bt", "genuine")):
            det = [r for r in rows if r["set"] == setname and r["truth"] == truth and r.get("yolo") and not r["augmented"]]
            passed = [r for r in det if r[c] >= tau[c]]
            e = {"n": len(det), "said_likely_genuine": len(passed), "rate": len(passed) / len(det) if det else None,
                 "wilson95": wilson(len(passed), len(det))}
            if truth == "counterfeit":
                grp = defaultdict(lambda: [0, 0])
                for r in det:
                    grp[r["group"]][0] += 1
                    grp[r["group"]][1] += r[c] >= tau[c]
                e["by_group_n_passed"] = {g: v for g, v in grp.items()}
            pol[f"{setname}/{truth}"] = e
        summ["policy_E"][c] = pol
    for r in rows:
        r.pop("whole_feat", None)
    (OUT / "scores.json").write_text(json.dumps(rows, indent=0), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps({k: summ[k] for k in ("auc_cf_originals", "auc_cf_originals_n")}, indent=1))


if __name__ == "__main__":
    main()
