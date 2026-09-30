"""Click the four corners of the watermark window on back-lit photos of a print_capture dataset.

For each backlight photo in manifest.jsonl without an annotation: click TL, TR, BR, BL (any currency,
any denomination; nothing is hardcoded). Keys: u = undo, s = skip (window not visible), q = quit.
Saved after every photo to <root>/wm_corners.json, keyed by the photo's sha256, as fractions of width
and height. Labels are made on TRAIN and VAL photos for training; TEST photos are annotated too, for
scoring only.
Usage: python scripts/dataset/annotate_watermark.py --root DATASET [--split train]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--split", default=None)
    a = ap.parse_args()
    rows = [json.loads(x) for x in (a.root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    out = a.root / "wm_corners.json"
    ann = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    todo = [r for r in rows if r["lighting"] == "backlight" and r["sha256"] not in ann and (a.split is None or r["split"] == a.split)]
    print(len(todo), "photos to annotate")
    for r in todo:
        img = cv2.imread(str(a.root / r["file"]))
        s = min(1.0, 1000 / max(img.shape[:2]))
        view = cv2.resize(img, None, fx=s, fy=s)
        pts = []

        def click(ev, x, y, *_):
            if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
                pts.append((x, y))

        cv2.namedWindow("wm")
        cv2.setMouseCallback("wm", click)
        while True:
            canvas = view.copy()
            for i, p in enumerate(pts):
                cv2.circle(canvas, p, 5, (0, 0, 255), -1)
                cv2.putText(canvas, "TL TR BR BL".split()[i], (p[0] + 6, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(canvas, f"{r['currency']} {r['denomination']}  click TL,TR,BR,BL  u=undo s=skip q=quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("wm", canvas)
            k = cv2.waitKey(30) & 0xFF
            if k == ord("u") and pts:
                pts.pop()
            if k == ord("q"):
                cv2.destroyAllWindows()
                return
            if k == ord("s"):
                ann[r["sha256"]] = None
                break
            if len(pts) == 4:
                ann[r["sha256"]] = [[round(x / view.shape[1], 5), round(y / view.shape[0], 5)] for x, y in pts]
                break
        out.write_text(json.dumps(ann, indent=0), encoding="utf-8")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
