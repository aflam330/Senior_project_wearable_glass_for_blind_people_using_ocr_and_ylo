"""UV fibre check: blocked 5-fold cross-validation over capture order.

Why. The first evaluation (uv_dataset.py evaluate) set the threshold on the first 60 % of each value's photographs
and tested on the last 40 %. The last 40 % of the 500 and 1,000 Taka photographs show the back of the note and
were taken from further away, which the calibration part did not contain, so the threshold was set for a different
kind of photograph. That single test split has been read; this script does not tune on it.

Protocol (declared before it was run, 2026-10-13):
  * Each value's photographs, in capture order, are cut into 5 contiguous blocks. Neighbouring photographs (the
    same note, seconds apart) stay in one block.
  * For each block, the threshold is set on the other four blocks only, with the SAME rule as before: the largest
    threshold that accepts at least 95 % of the training notes, never below (most fibres on a plain-surface
    patch in the training blocks) + 2.
  * Two variants, both reported, neither chosen by its result:
      A. one threshold for all values;
      B. one threshold per value (the glass knows the value before it runs the UV check).
  * The detector, the fibre finder and UV_REL_THRESHOLD are unchanged.

There are still no counterfeit notes: the "no fibres" side is the plain surface beside the note.

    python scripts/uv_blocked_cv.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from modes.uv_check import fibre_mask, inner  # noqa: E402
from scripts.uv_dataset import DATA, OUT, VALUES, _surface_patch, read  # noqa: E402

K = 5


def rule(note_counts, surface_counts):
    floor = (int(max(surface_counts)) if len(surface_counts) else 0) + 2
    s = np.sort(np.asarray(note_counts))
    return max(floor, int(s[int(np.floor(0.05 * len(s)))]))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 4), round(float(c + h), 4)]


def main():
    rows = [r for r in csv.DictReader(open(DATA / "boxes.csv", encoding="utf-8")) if r["x1"] != ""]
    items = []          # value, order index within value, fibres on the note, fibres on the surface patch (or None)
    for v in VALUES:
        rv = sorted((r for r in rows if r["value"] == v), key=lambda r: r["file"])
        for i, r in enumerate(rv):
            img = read(DATA / v / r["file"])
            x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
            _, n, ab = fibre_mask(inner(img[y1:y2, x1:x2]))
            if ab is None:
                continue
            p = _surface_patch(img, x1, y1, x2, y2)
            items.append({"value": v, "file": r["file"], "block": min(K - 1, i * K // len(rv)), "n": n,
                          "surface": fibre_mask(p)[1] if p is not None else None})
    out = {"A_one_threshold": [], "B_threshold_per_value": []}
    thr_log = {"A": [], "B": []}
    for b in range(K):
        tr = [it for it in items if it["block"] != b]
        te = [it for it in items if it["block"] == b]
        surf = [it["surface"] for it in tr if it["surface"] is not None]
        tA = rule([it["n"] for it in tr], surf)
        tB = {v: rule([it["n"] for it in tr if it["value"] == v], surf) for v in VALUES}
        thr_log["A"].append(tA)
        thr_log["B"].append(tB)
        for it in te:
            out["A_one_threshold"].append((it, tA))
            out["B_threshold_per_value"].append((it, tB[it["value"]]))
    res = {"date": "2026-10-13", "protocol": "blocked 5-fold cross-validation over capture order; threshold from the other 4 blocks",
           "photographs_judged": len(items), "thresholds_per_fold": thr_log}
    for name, pairs in out.items():
        acc = sum(it["n"] >= t for it, t in pairs)
        per = {}
        for v in VALUES:
            s = [(it, t) for it, t in pairs if it["value"] == v]
            k = sum(it["n"] >= t for it, t in s)
            per[v] = {"accepted": k, "n": len(s), "rate": round(k / len(s), 4), "wilson95": wilson(k, len(s))}
        sp = [(it, t) for it, t in pairs if it["surface"] is not None]
        rej = sum(it["surface"] < t for it, t in sp)
        res[name] = {"genuine_accepted": acc, "n": len(pairs), "rate": round(acc / len(pairs), 4),
                     "wilson95": wilson(acc, len(pairs)), "per_value": per,
                     "surface_patches_rejected": rej, "surface_patches": len(sp),
                     "surface_wilson95": wilson(rej, len(sp)),
                     "rejected_files": sorted(it["file"] + f" ({it['n']}<{t})" for it, t in pairs if it["n"] < t)}
    res["not_measured"] = ["counterfeit notes", "printed paper without fibres", "the glass camera"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "uv_blocked_cv.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
