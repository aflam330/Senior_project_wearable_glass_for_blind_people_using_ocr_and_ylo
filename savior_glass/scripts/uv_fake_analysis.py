"""UV check against the first counterfeit notes photographed under UV (data set/fake_bd_cur/uv_fake_cur, 36 pictures,
12 serials; added to data set/uv_auth/manifest.csv by uv_auth_eval.py --build). Run uv_auth_eval.py first.

1. Per class: fibre count of the deployed check, the paper's green/blue and red/blue ratios, overall brightness and
   colour saturation.
2. A two-sided fibre rule, accept if UV_MIN_FIBRES <= fibres <= K, with K chosen on the training folds only (the K
   that maximises genuine accepted minus counterfeits accepted), scored on the held-out fold: 5-fold cross-validation
   by group (serial for counterfeit fronts, the photo otherwise), seed 0. Pooled over the five held-out folds.
3. Capture-source check, same protocol: one threshold on a feature that has nothing to do with UV fibres (overall
   brightness; colour saturation). If such a feature separates the classes as well as the fibre rule, this data
   cannot show that the separation comes from the notes rather than from how they were photographed (genuine: glass
   camera, PNG, background visible; counterfeit: CamScanner, cropped and enhanced JPEG).

    python scripts/uv_fake_analysis.py
Output: results/uv_auth/fake_analysis.json
"""
from __future__ import annotations

import csv
import json
import random
import statistics as st
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from modes.uv_check import UVChecker  # noqa: E402
from scripts.uv_auth_eval import DS, OUT, group_of  # noqa: E402

config.UV_ISOLATION_PX = 0     # this analysis documents the check as it was before scripts/uv_fibre_v2.py (2026-11-24)

K_FOLDS = 5


def features(r):
    img = cv2.imread(str(DS / r["uv_path"]))
    if r["note_box"]:
        x1, y1, x2, y2 = (int(float(v)) for v in r["note_box"].split())
        img = img[max(y1, 0):y2, max(x1, 0):x2]
    res = UVChecker().check(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return {"fibres": res["fibres"], "paper_gb": res.get("paper_gb"), "paper_rb": res.get("paper_rb"),
            "brightness": float(hsv[..., 2].mean()), "saturation": float(hsv[..., 1].mean())}


def summary(v):
    v = [x for x in v if x is not None]
    return {"n": len(v), "min": round(min(v), 3), "median": round(st.median(v), 3), "max": round(max(v), 3)}


def cv_threshold(rows, key, two_sided_from=None):
    """Grouped K-fold CV of a one-threshold rule on `key`. two_sided_from: accept if two_sided_from <= x <= t (the
    fibre rule); otherwise the direction (accept above or below t) is also chosen on the training folds."""
    groups = sorted({r["group"] for r in rows})
    random.Random(0).shuffle(groups)
    fold = {g: i % K_FOLDS for i, g in enumerate(groups)}
    tp = fp = n_g = n_c = 0
    chosen = []
    for k in range(K_FOLDS):
        tr = [r for r in rows if fold[r["group"]] != k]
        te = [r for r in rows if fold[r["group"]] == k]
        cands = sorted({r[key] for r in tr})
        best = None
        for t in cands:
            for sign in ((1,) if two_sided_from is not None else (1, -1)):
                def acc(x, t=t, sign=sign):
                    return two_sided_from <= x <= t if two_sided_from is not None else (sign * x >= sign * t)
                u = sum(acc(r[key]) for r in tr if r["y"]) - sum(acc(r[key]) for r in tr if not r["y"])
                if best is None or u > best[0]:
                    best = (u, t, sign, acc)
        _, t, sign, acc = best
        chosen.append({"threshold": round(float(t), 3), "accept": "between" if two_sided_from is not None else ("above" if sign == 1 else "below")})
        tp += sum(acc(r[key]) for r in te if r["y"])
        fp += sum(acc(r[key]) for r in te if not r["y"])
        n_g += sum(r["y"] for r in te)
        n_c += sum(not r["y"] for r in te)
    return {"genuine_accepted": f"{tp}/{n_g}", "genuine_acceptance": round(tp / n_g, 4),
            "counterfeits_accepted": f"{fp}/{n_c}", "false_acceptance_rate": round(fp / n_c, 4),
            "per_fold_threshold": chosen}


def main():
    man = list(csv.DictReader((DS / "uv_auth" / "manifest.csv").open(encoding="utf-8")))
    rows = []
    for r in man:
        f = features(r)
        rows.append({**f, "y": r["genuine_or_counterfeit"] == "genuine", "group": group_of(r),
                     "denomination": r["denomination"]})
    res = {"date": "2026-11-24", "n": len(rows), "n_genuine": sum(r["y"] for r in rows),
           "n_counterfeit": sum(not r["y"] for r in rows), "n_groups": len({r["group"] for r in rows}),
           "per_class": {c: {k: summary([r[k] for r in rows if r["y"] == (c == "genuine")])
                             for k in ("fibres", "paper_gb", "paper_rb", "brightness", "saturation")}
                         for c in ("genuine", "counterfeit")}}
    m = int(config.UV_MIN_FIBRES)
    res["deployed_rule"] = {"rule": f"fibres >= {m}",
                            "genuine_accepted": f"{sum(r['fibres'] >= m for r in rows if r['y'])}/{res['n_genuine']}",
                            "counterfeits_accepted": f"{sum(r['fibres'] >= m for r in rows if not r['y'])}/{res['n_counterfeit']}"}
    res["two_sided_fibre_rule_cv"] = cv_threshold(rows, "fibres", two_sided_from=m)
    res["capture_source_check_cv"] = {k: cv_threshold(rows, k) for k in ("brightness", "saturation")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fake_analysis.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
