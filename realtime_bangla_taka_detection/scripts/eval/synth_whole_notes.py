"""Approach A, step 1: reconstruct a whole-note image for every JaalTaka note from its views 1-4.

Each front view is registered (SIFT + RANSAC homography) to a whole-note front template of its
denomination (500 or 1000 BDT, BanglaTaka photos; the denomination is the template with more
inliers). The warped views are averaged on the template canvas. Pixels no view covers are filled
by inpainting from the note's own pixels (never from the genuine template, which would leak
genuine texture into counterfeit notes). A note is kept if at least 90 % of the canvas is covered.

Output: results/jaal_whole/synth_notes/<class>_<note>.jpg (1000 px wide) and synth_index.json
Uses all notes of all splits; the split is carried in the index and respected downstream.
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
from jaal_view_geometry import MIN_INLIERS, TEMPLATES, W, prep  # noqa: E402

OUT = ROOT / "results" / "jaal_whole" / "synth_notes"
_T = {}


def _init():
    cv2.setNumThreads(1)
    sift = cv2.SIFT_create(nfeatures=4000)
    for d, p in TEMPLATES.items():
        img = cv2.imread(str(p))
        s = W / img.shape[1]
        col = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)
        _T[d] = (sift.detectAndCompute(g, None), g.shape)
    _T["sift"] = sift
    _T["bf"] = cv2.BFMatcher(cv2.NORM_L2)


def _reg(gray, d):
    sift, bf = _T["sift"], _T["bf"]
    kp, des = sift.detectAndCompute(gray, None)
    (tkp, tdes), _ = _T[d]
    if des is None or len(kp) < 10:
        return 0, None
    good = [m for m, n in (p for p in bf.knnMatch(des, tdes, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        return 0, None
    src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([tkp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    return (int(mask.sum()) if mask is not None else 0), H


def build(item):
    rec = _build(item)
    (OUT / (item[0].replace(":", "_") + ".json")).write_text(json.dumps(rec), encoding="utf-8")
    return rec


def _build(item):
    nid, paths, label, split = item
    side = OUT / (nid.replace(":", "_") + ".json")
    if side.is_file():  # resumable: a note already processed is not redone
        return json.loads(side.read_text(encoding="utf-8"))
    views = []
    for p in paths[:4]:
        img = cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_2)  # half size: views are ~2000 px, 700 px is used
        s = 700 / img.shape[1]
        col = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        views.append((col, cv2.cvtColor(col, cv2.COLOR_BGR2GRAY)))
    regs = {d: [_reg(g, d) for _, g in views] for d in TEMPLATES}
    denom = max(regs, key=lambda d: sum(r[0] for r in regs[d]))
    th, tw = _T[denom][1]
    acc = np.zeros((th, tw, 3), np.float32)
    cnt = np.zeros((th, tw), np.float32)
    used = 0
    for (col, _), (inl, H) in zip(views, regs[denom]):
        if H is None or inl < MIN_INLIERS:
            continue
        c = cv2.perspectiveTransform(np.float32([[0, 0], [col.shape[1], 0], [col.shape[1], col.shape[0]], [0, col.shape[0]]]).reshape(-1, 1, 2), H).reshape(-1, 2)
        area = cv2.contourArea(c.astype(np.float32))
        if not (0.05 * th * tw < area < 1.5 * th * tw):
            continue  # degenerate homography
        warped = cv2.warpPerspective(col, H, (tw, th), flags=cv2.INTER_LINEAR)
        m = cv2.warpPerspective(np.ones(col.shape[:2], np.float32), H, (tw, th), flags=cv2.INTER_NEAREST)
        acc += warped.astype(np.float32) * m[..., None]
        cnt += m
        used += 1
    cover = float((cnt > 0).mean())
    rec = {"note_id": nid, "label": label, "split": split, "denom": denom, "views_used": used, "coverage": cover}
    if used < 2 or cover < 0.90:
        rec["kept"] = False
        return rec
    img = (acc / np.maximum(cnt, 1)[..., None]).clip(0, 255).astype(np.uint8)
    hole = (cnt == 0).astype(np.uint8) * 255
    if hole.any():
        img = cv2.inpaint(img, hole, 5, cv2.INPAINT_TELEA)
    name = nid.replace(":", "_") + ".jpg"
    cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    rec.update({"kept": True, "file": name})
    return rec


def main() -> None:
    from roboeye.camva.notes import load_splits
    OUT.mkdir(parents=True, exist_ok=True)
    splits, records = load_splits()
    items = [(n, records[n]["view_paths"], int(records[n]["label"]), s) for s in ("train", "val", "test") for n in splits[s]]
    rows = []
    with ProcessPoolExecutor(max_workers=3, initializer=_init) as ex:
        for i, r in enumerate(ex.map(build, items, chunksize=4)):
            rows.append(r)
            if (i + 1) % 100 == 0:
                print(i + 1, "/", len(items), flush=True)
    kept = [r for r in rows if r["kept"]]
    summ = {s: {"kept": sum(r["split"] == s for r in kept), "total": sum(r["split"] == s for r in rows)} for s in ("train", "val", "test")}
    (OUT.parent / "synth_index.json").write_text(json.dumps({"summary": summ, "notes": rows}, indent=0), encoding="utf-8")
    print(summ)


if __name__ == "__main__":
    main()
