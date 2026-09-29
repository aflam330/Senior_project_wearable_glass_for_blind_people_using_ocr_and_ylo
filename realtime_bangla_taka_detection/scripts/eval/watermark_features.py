"""Watermark-window crops and features from back-lit JaalTaka view 6, for every note.

View 6 is photographed against light, so the watermark (portrait + denomination electrotype in the
blank oval) is visible on genuine notes. View 6 is registered (SIFT + RANSAC) to the whole-note front
template of the note's denomination (as in synth_whole_notes.py) and warped onto it; the watermark
window WM_BOX (fractions of the note) is cropped. Hand-crafted features on the grey crop (256 x 256):
  lap_var  variance of the Laplacian (fine structure)
  std      grey-level standard deviation
  edges    Canny edge density
  rel_mean window mean / whole-warped-note mean (brightness of the unprinted window under backlight)
Crops are saved for learned features (watermark_hybrid.py).
Output: results/watermark/crops/<note>.png, features.json
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
import synth_whole_notes as sw  # noqa: E402

OUT = ROOT / "results" / "watermark"
WM_BOX = (0.70, 0.33, 0.93, 0.85)  # blank oval window; starts below the upper-right serial
MIN_INLIERS_BACKLIT = 12  # back-lit views give fewer SIFT matches; window coverage still checked


def feats(item):
    nid, path, label, split, denom = item
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
    s = 700 / img.shape[1]
    col = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)
    cands = [denom] if denom else list(sw.TEMPLATES)
    best = max(((d, *sw._reg(gray, d)) for d in cands), key=lambda r: r[1])
    d, inl, H = best
    rec = {"note_id": nid, "label": label, "split": split, "denom": d, "inliers": inl}
    if H is None or inl < MIN_INLIERS_BACKLIT:
        rec["ok"] = False
        return rec
    th, tw = sw._T[d][1]
    warped = cv2.warpPerspective(col, H, (tw, th))
    cover = cv2.warpPerspective(np.ones(gray.shape, np.uint8), H, (tw, th))
    x0, y0, x1, y1 = WM_BOX
    crop = warped[int(y0 * th):int(y1 * th), int(x0 * tw):int(x1 * tw)]
    cov = cover[int(y0 * th):int(y1 * th), int(x0 * tw):int(x1 * tw)].mean()
    if cov < 0.8:
        rec.update({"ok": False, "window_coverage": float(cov)})
        return rec
    g = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (256, 256)).astype(np.float32)
    note_mean = float(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)[cover > 0].mean())
    rec.update({"ok": True, "window_coverage": float(cov),
                "lap_var": float(cv2.Laplacian(g, cv2.CV_32F).var()), "std": float(g.std()),
                "edges": float((cv2.Canny(g.astype(np.uint8), 50, 150) > 0).mean()),
                "rel_mean": float(g.mean() / max(note_mean, 1.0))})
    cv2.imwrite(str(OUT / "crops" / (nid.replace(":", "_") + ".png")), cv2.resize(crop, (224, 224)))
    return rec


def main() -> None:
    from roboeye.camva.notes import SPLIT_DIR, load_splits
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    splits, records = load_splits(SPLIT_DIR)
    synth = {r["note_id"]: r for r in json.loads((ROOT / "results/jaal_whole/synth_index.json").read_text(encoding="utf-8"))["notes"]}
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    items = [(n, records[n]["view_paths"][5], int(records[n]["label"]), s, synth.get(n, {}).get("denom"))
             for s in ("train", "val", "test") for n in splits[s]][:limit]
    with ProcessPoolExecutor(max_workers=4, initializer=sw._init) as ex:
        rows = list(ex.map(feats, items, chunksize=4))
    (OUT / "features.json").write_text(json.dumps(rows, indent=0), encoding="utf-8")
    print("ok", sum(r["ok"] for r in rows), "of", len(rows))


if __name__ == "__main__":
    main()
