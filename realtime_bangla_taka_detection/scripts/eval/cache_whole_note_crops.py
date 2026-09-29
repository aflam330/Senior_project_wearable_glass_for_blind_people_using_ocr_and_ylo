"""Cache the Taka-YOLO note crops of all whole-note photos used in jaal_whole_note.py.

Same items and same crop processing as jaal_whole_note.py (top box, 640 px wide, JPEG q90,
portrait turned landscape), so later checkers can be scored without re-running the detector.
Output: results/jaal_whole/crops/<n>.jpg and crops_index.json (set, truth, denom, group, augmented, yolo).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORK / "savior_glass"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from jaal_whole_note import collect  # noqa: E402

OUT = ROOT / "results" / "jaal_whole" / "crops"


def main() -> None:
    from modes.currency_mode import CurrencyMode
    OUT.mkdir(parents=True, exist_ok=True)
    mode = CurrencyMode()
    mode._load_yolo()
    rows = []
    for i, it in enumerate(collect()):
        raw = cv2.imread(str(it["path"]))
        row = {k: (str(v.relative_to(WORK)) if k == "path" else v) for k, v in it.items()}
        hits = mode.detect_live(raw)
        row["yolo"] = hits[0]["name"] if hits else None
        if hits:
            x, y, w, h = hits[0]["bbox"]
            crop = raw[max(0, y):y + h, max(0, x):x + w]
            crop = cv2.resize(crop, None, fx=640 / crop.shape[1], fy=640 / crop.shape[1], interpolation=cv2.INTER_AREA)
            if crop.shape[0] > crop.shape[1]:
                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            row["crop"] = f"{i:05d}.jpg"
            cv2.imwrite(str(OUT / row["crop"]), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        rows.append(row)
        if (i + 1) % 250 == 0:
            print(i + 1, flush=True)
    (OUT.parent / "crops_index.json").write_text(json.dumps(rows, indent=0), encoding="utf-8")
    print("crops", sum(1 for r in rows if r.get("crop")), "of", len(rows))


if __name__ == "__main__":
    main()
