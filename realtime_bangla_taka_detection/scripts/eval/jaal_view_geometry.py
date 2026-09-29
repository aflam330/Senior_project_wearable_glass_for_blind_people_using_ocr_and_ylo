"""Where JaalTaka views 1-4 lie on a whole note, measured on TRAIN notes only.

Each of views 1-4 (front close-ups) is registered to a clean whole-note front template of 500 and
of 1000 BDT from BanglaTaka (SIFT + RANSAC homography). The template with more inliers gives the
note's denomination; the projected view outline gives that view's window in normalised
whole-note coordinates (x, y in [0, 1]). The per-view median window is the geometry used to cut
JaalTaka-like views out of a whole-note crop (scripts/eval/jaal_whole_note.py).

Output: results/jaal_whole/view_geometry.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from roboeye.camva.notes import load_splits  # noqa: E402

RAW = ROOT.parent / "data set" / "Bangladeshi_Paper_Currency_Raw" / "Bangladeshi_Paper_Currency_Raw"
TEMPLATES = {"500": RAW / "500" / "500 Taka_0001.jpg", "1000": RAW / "1000" / "1000  Taka_001.jpg"}
OUT = ROOT / "results" / "jaal_whole" / "view_geometry.json"
W = 1000  # template width after resizing
MIN_INLIERS = 25

sift = cv2.SIFT_create(nfeatures=4000)
matcher = cv2.BFMatcher(cv2.NORM_L2)


def prep(img: np.ndarray, width: int) -> np.ndarray:
    s = width / img.shape[1]
    return cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def register(view_gray, tmpl) -> tuple[int, np.ndarray | None]:
    kp, des = sift.detectAndCompute(view_gray, None)
    if des is None or len(kp) < 10:
        return 0, None
    tkp, tdes = tmpl
    good = [m for m, n in (p for p in matcher.knnMatch(des, tdes, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        return 0, None
    src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([tkp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    return (int(mask.sum()) if mask is not None else 0), H


def main() -> None:
    tm = {}
    for d, p in TEMPLATES.items():
        g = prep(cv2.imread(str(p)), W)
        kp, des = sift.detectAndCompute(g, None)
        tm[d] = ((kp, des), g.shape)
    splits, records = load_splits()
    ids = list(splits["train"])
    random.Random(0).shuffle(ids)
    windows = {k: [] for k in range(4)}
    denoms, used = {}, 0
    for nid in ids[:200]:
        paths = records[nid]["view_paths"][:4]
        per_view = []
        for p in paths:
            g = prep(cv2.imread(p), 700)
            best = max(((d, *register(g, tm[d][0])) for d in tm), key=lambda r: r[1])
            per_view.append((best, g.shape))
        d_votes = [b[0] for (b, _) in per_view if b[1] >= MIN_INLIERS]
        if not d_votes:
            continue
        denom = max(set(d_votes), key=d_votes.count)
        denoms[nid] = denom
        th, tw = tm[denom][1]
        for k, ((d, inl, H), (h, w)) in enumerate(per_view):
            if d != denom or inl < MIN_INLIERS or H is None:
                continue
            c = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), H).reshape(-1, 2)
            x0, y0 = c[:, 0].min() / tw, c[:, 1].min() / th
            x1, y1 = c[:, 0].max() / tw, c[:, 1].max() / th
            if not (-0.3 < x0 < x1 < 1.3 and -0.3 < y0 < y1 < 1.3 and (x1 - x0) > 0.15):
                continue  # degenerate homography
            windows[k].append([x0, y0, x1, y1])
        used += 1
    geom = {}
    for k, ws in windows.items():
        a = np.clip(np.asarray(ws), 0, 1)
        geom[str(k + 1)] = {"median_xyxy": np.median(a, 0).round(4).tolist(),
                            "q25": np.quantile(a, 0.25, 0).round(4).tolist(),
                            "q75": np.quantile(a, 0.75, 0).round(4).tolist(), "n": len(ws)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    blob = {"split": "train", "notes_tried": 200, "notes_registered": used,
            "denomination_counts": {d: list(denoms.values()).count(d) for d in TEMPLATES},
            "templates": {d: str(p.relative_to(ROOT.parent)) for d, p in TEMPLATES.items()},
            "views": geom}
    OUT.write_text(json.dumps(blob, indent=1), encoding="utf-8")
    print(json.dumps(blob, indent=1))


if __name__ == "__main__":
    main()
