"""UV authentication evaluation: manifest, note-grouped splits, baselines, metrics, quality checks.

It measures AUTHENTICATION (genuine vs counterfeit), not only whether the UV module runs. With the data on this
machine it cannot: no counterfeit note has been photographed under UV (data set/uv_notes holds 118 genuine notes).
Every metric that needs both classes is then written as "not computable" with the reason; nothing is invented.

1. Manifest (one row per image), data set/uv_auth/manifest.csv, columns:
     physical_note_id, denomination (200/500/1000), genuine_or_counterfeit (genuine/counterfeit), uv_path,
     visible_path (may be empty), capture_session, counterfeit_print_id (may be empty), note_box (x1 y1 x2 y2, may be
     empty), note_id_known (1 if the photographer recorded which physical note it is)
   `--build` writes it from what exists: uv_notes (genuine; the file does not record which physical note each photo
   shows, so each photo is its own note and note_id_known = 0 - a split by photo then, flagged in the output).
   New photos: add rows by hand, keeping one physical_note_id per physical note and one counterfeit_print_id per print.
2. Splits by group, never by image: group = counterfeit_print_id for counterfeits that have one, else physical_note_id.
   A fixed 60/20/20 train/validation/test split by group (seed 0), or `--cv K`: K-fold cross-validation by group.
   The script asserts that no group is in two splits.
3. Baselines (each scored only when its inputs exist):
     UV-only       fibre count of the deployed check (modes/uv_check.py) on the note crop; decision count >= UV_MIN_FIBRES
     visible-only  p(genuine) of the counterfeit checker (roboeye BDCounterfeitAuth) on the visible photo
     visible + UV  logistic regression on [log(1 + fibres), logit p] fitted on the training groups only
4. Metrics: accuracy, precision, recall, F1 (positive class = genuine), AUROC, AUPRC, false acceptance rate
   (counterfeits accepted), false rejection rate (genuine rejected), and a confusion matrix per denomination.
5. Quality: blur (Laplacian variance < 50), exposure (mean < 30 or > 225, or > 5 % clipped pixels), missing UV
   response (not a UV picture by the glass's own blue/green rule, modes/currency_mode.is_uv_frame).

    python scripts/uv_auth_eval.py --build          # write the manifest from the existing data
    python scripts/uv_auth_eval.py [--cv 5]
Output: results/uv_auth/{manifest_checked.csv, rows.csv, evaluation.json}
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from modes.currency_mode import is_uv_frame  # noqa: E402
from modes.uv_check import UVChecker  # noqa: E402

DS = ROOT.parent / "data set"
MANIFEST = DS / "uv_auth" / "manifest.csv"
OUT = ROOT / "results" / "uv_auth"
COLS = ["physical_note_id", "denomination", "genuine_or_counterfeit", "uv_path", "visible_path", "capture_session",
        "counterfeit_print_id", "note_box", "note_id_known"]


def build():
    rows = []
    boxes = {r["file"]: r for r in csv.DictReader((DS / "uv_notes" / "boxes.csv").open(encoding="utf-8"))}
    for p in sorted((DS / "uv_notes").glob("*/uv_*.png")):
        b = boxes.get(p.name, {})
        box = " ".join(b[k] for k in ("x1", "y1", "x2", "y2")) if b.get("x1") else ""
        rows.append({"physical_note_id": p.stem, "denomination": p.parent.name, "genuine_or_counterfeit": "genuine",
                     "uv_path": str(p.relative_to(DS)), "visible_path": "", "capture_session": "uv_notes_2026-10-12",
                     "counterfeit_print_id": "", "note_box": box, "note_id_known": "0"})
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {MANIFEST} ({len(rows)} rows)")


def group_of(r):
    return ("print:" + r["counterfeit_print_id"]) if r["genuine_or_counterfeit"] == "counterfeit" and r["counterfeit_print_id"] \
        else ("note:" + r["physical_note_id"])


def split_groups(groups, seed=0, cv=0):
    g = sorted(set(groups))
    random.Random(seed).shuffle(g)
    if cv:
        return [{x: ("test" if i % cv == k else "train") for i, x in enumerate(g)} for k in range(cv)]
    n = len(g)
    return [{x: ("train" if i < 0.6 * n else "val" if i < 0.8 * n else "test") for i, x in enumerate(g)}]


def quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean, clip = float(gray.mean()), float(((gray <= 2) | (gray >= 253)).mean())
    return {"blur_var": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1), "mean": round(mean, 1),
            "clipped": round(clip, 4), "blurry": float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 50,
            "bad_exposure": mean < 30 or mean > 225 or clip > 0.05, "uv_response_missing": not is_uv_frame(img)}


def metrics(y, score, decision):
    """y: 1 genuine / 0 counterfeit."""
    y, d = np.asarray(y), np.asarray(decision).astype(int)
    out = {"n": int(len(y)), "n_genuine": int(y.sum()), "n_counterfeit": int((1 - y).sum())}
    tp, fp = int(((d == 1) & (y == 1)).sum()), int(((d == 1) & (y == 0)).sum())
    fn = int(((d == 0) & (y == 1)).sum())
    out["genuine_acceptance"] = tp / max(out["n_genuine"], 1) if out["n_genuine"] else None
    out["false_rejection_rate"] = fn / out["n_genuine"] if out["n_genuine"] else None
    if out["n_counterfeit"] == 0 or out["n_genuine"] == 0:
        why = "no counterfeit images" if out["n_counterfeit"] == 0 else "no genuine images"
        for k in ("accuracy", "precision", "recall", "f1", "auroc", "auprc", "false_acceptance_rate"):
            out[k] = f"not computable: {why}"
        return out
    from sklearn.metrics import average_precision_score, roc_auc_score
    prec = tp / max(tp + fp, 1)
    rec = tp / out["n_genuine"]
    out.update(accuracy=float((d == y).mean()), precision=prec, recall=rec,
               f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
               auroc=float(roc_auc_score(y, score)), auprc=float(average_precision_score(y, score)),
               false_acceptance_rate=fp / out["n_counterfeit"])
    return out


def confusion_by_denom(rows, key):
    cm = defaultdict(Counter)
    for r in rows:
        if r.get(key) is None:
            continue
        cm[r["denomination"]][(r["genuine_or_counterfeit"], "accepted" if r[key] else "rejected")] += 1
    return {d: {f"{a}->{b}": n for (a, b), n in sorted(c.items())} for d, c in sorted(cm.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--cv", type=int, default=0)
    a = ap.parse_args()
    if a.build:
        return build()
    if not MANIFEST.is_file():
        sys.exit(f"no manifest at {MANIFEST}; run with --build or write it (columns: {', '.join(COLS)})")
    man = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    bad = [r for r in man if r["genuine_or_counterfeit"] not in ("genuine", "counterfeit") or r["denomination"] not in ("200", "500", "1000")]
    assert not bad, f"{len(bad)} manifest rows with an unknown label or denomination, first: {bad[0]}"

    uv = UVChecker()
    auth = None
    rows = []
    for r in man:
        img = cv2.imread(str(DS / r["uv_path"]))
        if img is None:
            rows.append({**r, "error": "UV image not readable"})
            continue
        if r["note_box"]:
            x1, y1, x2, y2 = (int(float(v)) for v in r["note_box"].split())
            crop = img[max(y1, 0):y2, max(x1, 0):x2]
        else:
            crop = img
        res = uv.check(crop)
        row = {**r, **quality(img), "box_used": bool(r["note_box"]), "uv_outcome": res["outcome"], "fibres": res["fibres"],
               "uv_accept": res["outcome"] == "uv_marks_found", "p_visible": None, "visible_accept": None}
        if r["visible_path"]:
            if auth is None:
                sys.path.insert(0, str(ROOT.parent / "realtime_bangla_taka_detection"))
                from roboeye.bd_counterfeit_auth import BDCounterfeitAuth
                auth = BDCounterfeitAuth(min_conf=0.0)
            vis = cv2.imread(str(DS / r["visible_path"]))
            if vis is not None:
                row["p_visible"] = float(auth.predict_bgr(vis)["genuine_prob"])
                row["visible_accept"] = row["p_visible"] >= float(getattr(config, "BD_AUTH_MIN_CONF", 0.95))
        rows.append(row)
    ok = [r for r in rows if "error" not in r]
    groups = [group_of(r) for r in ok]
    folds = split_groups(groups, cv=a.cv)
    for f in folds:                                     # no group (note or print) in two splits
        by = defaultdict(set)
        for r, g in zip(ok, groups):
            by[g].add(f[g])
        assert all(len(s) == 1 for s in by.values())

    y = [int(r["genuine_or_counterfeit"] == "genuine") for r in ok]
    res = {"manifest": str(MANIFEST), "n_images": len(rows), "n_unreadable": len(rows) - len(ok),
           "n_groups": len(set(groups)), "class_counts": dict(Counter(r["genuine_or_counterfeit"] for r in ok)),
           "per_denomination": dict(Counter(f"{r['denomination']} {r['genuine_or_counterfeit']}" for r in ok)),
           "physical_note_ids_recorded": all(r["note_id_known"] == "1" for r in ok),
           "split": f"{a.cv}-fold cross-validation by group" if a.cv else "60/20/20 by group, seed 0",
           "quality": {k: int(sum(r[k] for r in ok)) for k in ("blurry", "bad_exposure", "uv_response_missing")},
           "baselines": {}}
    if not res["physical_note_ids_recorded"]:
        res["warning"] = ("the manifest does not record which physical note each photo shows (note_id_known = 0), so "
                          "photos of one note could sit in different splits; record note ids for new photos")
    split_rows = {"all (no fitting)": ok}
    if not a.cv:
        split_rows["test split"] = [r for r, g in zip(ok, groups) if folds[0][g] == "test"]
    res["split_sizes"] = {s: len(v) for s, v in split_rows.items()}
    for name, key, score in (("UV-only", "uv_accept", "fibres"), ("visible-only", "visible_accept", "p_visible")):
        have = [r for r in ok if r[key] is not None]
        if not have:
            res["baselines"][name] = "not computable: no images for this input"
            continue
        res["baselines"][name] = {s: metrics([int(r["genuine_or_counterfeit"] == "genuine") for r in rs if r[key] is not None],
                                             [r[score] for r in rs if r[key] is not None],
                                             [r[key] for r in rs if r[key] is not None])
                                  for s, rs in split_rows.items()}
        res["baselines"][name]["confusion_by_denomination"] = confusion_by_denom(have, key)
    both = [r for r in ok if r["p_visible"] is not None]
    if not both or len({r["genuine_or_counterfeit"] for r in both}) < 2:
        res["baselines"]["visible + UV"] = ("not computable: needs visible and UV photos of the same notes, of both "
                                            "classes" if not both else "not computable: only one class has both photos")
    else:
        from sklearn.linear_model import LogisticRegression
        X = np.array([[np.log1p(r["fibres"]), np.log(max(r["p_visible"], 1e-6) / max(1 - r["p_visible"], 1e-6))] for r in both])
        yy = np.array([int(r["genuine_or_counterfeit"] == "genuine") for r in both])
        gg = [group_of(r) for r in both]
        fold_res = []
        for f in (folds if a.cv else [{g: ("test" if folds[0][g] == "test" else "train") for g in gg}]):
            tr = np.array([f[g] == "train" for g in gg])
            te = np.array([f[g] == "test" for g in gg])
            if len(set(yy[tr])) < 2 or not te.any():
                continue
            lr = LogisticRegression().fit(X[tr], yy[tr])
            s = lr.predict_proba(X[te])[:, 1]
            fold_res.append(metrics(yy[te], s, s >= 0.5))
        res["baselines"]["visible + UV"] = fold_res or "not computable: a training fold lacks one class"
    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(ok[0].keys()) if ok else COLS
    with (OUT / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(ok)
    (OUT / "evaluation.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str)[:3500])


if __name__ == "__main__":
    main()
