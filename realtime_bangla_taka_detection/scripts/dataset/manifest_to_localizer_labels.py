"""Turn a print_capture dataset into localizer labels (same format as watermark_localizer_labels.py).

Reads <root>/manifest.jsonl (backlight photos only) and <root>/wm_corners.json (annotate_watermark.py,
or --wm-corners given at capture). The split is the dataset's fixed print split. Photos are cached at
320 x 320. Then train with:
  python scripts/train/train_watermark_localizer.py <seed> <out_dir>
Output: <out_dir>/labels.json, <out_dir>/cache/<sha>.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    rows = [json.loads(x) for x in (a.root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    ann_path = a.root / "wm_corners.json"
    ann = json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.exists() else {}
    (a.out / "cache").mkdir(parents=True, exist_ok=True)
    out = []
    for r in rows:
        if r["lighting"] != "backlight":
            continue
        corners = ann.get(r["sha256"])
        if corners is None and "wm_corners" in r:
            c = json.loads(r["wm_corners"])
            corners = [c[i:i + 2] for i in range(0, 8, 2)]
        img = cv2.imread(str(a.root / r["file"]))
        s = 700 / img.shape[1]
        col = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(a.out / "cache" / f"{r['sha256']}.jpg"), cv2.resize(col, (320, 320), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        out.append({"note_id": r["sha256"], "photo": r["file"], "print_id": r["print_id"], "currency": r["currency"],
                    "denom": r["denomination"], "label": 1 if r["label"] == "genuine" else 0, "split": r["split"],
                    "camera": r["camera"], "corners": corners})
    (a.out / "labels.json").write_text(json.dumps({"split": str(a.root / "splits.json"), "dataset_root": str(a.root), "size": 320,
                                                   "rows": out}, indent=0), encoding="utf-8")
    for s in ("train", "val", "test"):
        rs = [r for r in out if r["split"] == s]
        print(s, len(rs), "labelled", sum(r["corners"] is not None for r in rs))


if __name__ == "__main__":
    main()
