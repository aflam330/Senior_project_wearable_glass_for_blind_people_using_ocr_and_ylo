"""Resize every JaalTaka view to 256 px (short side) once, for fast fine-tuning.

cache/views256 was built with cv2.IMREAD_REDUCED_COLOR_4, which decodes the JPEG at
1/4 size first (about 420 px from a 1700 px view) and then resizes. Those files are
kept, because the published fine-tune used them.

cache/views256_full decodes the full JPEG and resizes the short side to 256.
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "cache" / "views256_full"


def one(item):
    nid, k, p = item
    dst = OUT / f"{nid.replace(':', '_')}_{k}.jpg"
    if dst.is_file() and dst.stat().st_size > 0:
        return
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"unreadable view: {p}")
    s = 256 / min(img.shape[:2])
    cv2.imwrite(str(dst), cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == "__main__":
    from roboeye.camva.notes import SPLIT_DIR, load_splits
    OUT.mkdir(parents=True, exist_ok=True)
    _, rec = load_splits(SPLIT_DIR)
    items = [(n, k, p) for n, r in rec.items() for k, p in enumerate(r["view_paths"][:6])]
    with ProcessPoolExecutor(6) as ex:
        list(ex.map(one, items, chunksize=32))
    print("cached", len(list(OUT.glob("*.jpg"))))
