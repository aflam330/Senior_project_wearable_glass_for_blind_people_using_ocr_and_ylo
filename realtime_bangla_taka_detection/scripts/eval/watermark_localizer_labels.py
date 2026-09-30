"""Corner labels for a learned watermark localizer, from the SIFT registration of back-lit view 6.

For every JaalTaka note whose view 6 registers to its denomination template (same rule as
watermark_features.py: >= 12 inliers, window coverage >= 0.8), the window WM_BOX is mapped back
through the inverse homography into the photo. The four corners (TL, TR, BR, BL), as fractions of
the photo width and height, are the label. The photo is cached at 320 x 320 for training.
Notes that do not register are cached too (label null): the localizer is scored on them for coverage only.
Output: results/watermark_localizer/labels.json, cache/wm_loc/<note>.jpg
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
from watermark_features import MIN_INLIERS_BACKLIT, WM_BOX  # noqa: E402

OUT = ROOT / "results" / "watermark_localizer"
CACHE = ROOT / "cache" / "wm_loc"
SIZE = 320


def label(item):
    nid, path, lab, split, denom = item
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
    s = 700 / img.shape[1]
    col = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(CACHE / (nid.replace(":", "_") + ".jpg")), cv2.resize(col, (SIZE, SIZE), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    gray = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)
    cands = [denom] if denom else [d for d in sw._T if d not in ("sift", "bf")]
    d, inl, H = max(((d, *sw._reg(gray, d)) for d in cands), key=lambda r: r[1])
    rec = {"note_id": nid, "label": lab, "split": split, "denom": d, "inliers": inl, "corners": None}
    if H is None or inl < MIN_INLIERS_BACKLIT:
        return rec
    th, tw = sw._T[d][1]
    x0, y0, x1, y1 = WM_BOX
    cover = cv2.warpPerspective(np.ones(gray.shape, np.uint8), H, (tw, th))
    if cover[int(y0 * th):int(y1 * th), int(x0 * tw):int(x1 * tw)].mean() < 0.8:
        return rec
    box = np.float32([[x0 * tw, y0 * th], [x1 * tw, y0 * th], [x1 * tw, y1 * th], [x0 * tw, y1 * th]]).reshape(-1, 1, 2)
    pts = cv2.perspectiveTransform(box, np.linalg.inv(H)).reshape(-1, 2)
    pts[:, 0] /= col.shape[1]
    pts[:, 1] /= col.shape[0]
    rec["corners"] = np.round(pts, 5).tolist()
    return rec


def main() -> None:
    from roboeye.camva.notes import load_splits
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    splits, records = load_splits(ROOT / "results" / "serial_split")
    synth = {r["note_id"]: r for r in json.loads((ROOT / "results/jaal_whole/synth_index.json").read_text(encoding="utf-8"))["notes"]}
    items = [(n, records[n]["view_paths"][5], int(records[n]["label"]), s, synth.get(n, {}).get("denom"))
             for s in ("train", "val", "test") for n in splits[s]]
    with ProcessPoolExecutor(max_workers=4, initializer=sw._init) as ex:
        rows = list(ex.map(label, items, chunksize=4))
    (OUT / "labels.json").write_text(json.dumps({"split": "serial_split", "wm_box": WM_BOX, "size": SIZE, "rows": rows}, indent=0),
                                     encoding="utf-8")
    for s in ("train", "val", "test"):
        r = [x for x in rows if x["split"] == s]
        print(s, len(r), "labelled", sum(x["corners"] is not None for x in r))


if __name__ == "__main__":
    main()
