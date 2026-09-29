"""Serial-number audit of JaalTaka: do counterfeit notes share serial numbers?

The lower-left serial of every reconstructed whole note (results/jaal_whole/synth_notes) is read
with EasyOCR (bn + en, CPU). Bangla digits are mapped to 0-9 and the longest digit run of 6-8 digits is
kept. Counts per class and denomination: notes read, distinct serials, and the share of notes carrying
the most common serial. A shared serial is a shortcut a model can learn instead of authenticity.
Output: results/jaal_whole/serial_audit.json
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
SYN = ROOT / "results" / "jaal_whole"
BN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def main() -> None:
    import easyocr
    reader = easyocr.Reader(["bn", "en"], gpu=False, verbose=False)
    idx = [r for r in json.loads((SYN / "synth_index.json").read_text(encoding="utf-8"))["notes"] if r.get("kept")]
    rows = []
    for i, r in enumerate(idx):
        img = cv2.imread(str(SYN / "synth_notes" / r["file"]))
        h, w = img.shape[:2]
        crop = img[int(0.80 * h):int(0.96 * h), int(0.08 * w):int(0.42 * w)]
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        text = " ".join(t for _, t, _ in reader.readtext(crop)).translate(BN)
        runs = [m for m in re.findall(r"\d+", text.replace(" ", "")) if 6 <= len(m) <= 8]
        rows.append({"note_id": r["note_id"], "label": r["label"], "denom": r["denom"], "split": r["split"],
                     "ocr": text, "serial": max(runs, key=len) if runs else None})
        if (i + 1) % 200 == 0:
            print(i + 1, flush=True)
    summ = {}
    for denom in ("500", "1000"):
        for label, name in ((0, "counterfeit"), (1, "genuine")):
            rs = [x for x in rows if x["denom"] == denom and x["label"] == label]
            ser = [x["serial"] for x in rs if x["serial"]]
            c = collections.Counter(ser)
            top = c.most_common(3)
            summ[f"{denom}/{name}"] = {"notes": len(rs), "serial_read": len(ser), "distinct": len(c),
                                       "top3": top, "share_top1_of_read": (top[0][1] / len(ser)) if ser else None}
    (SYN / "serial_audit.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
