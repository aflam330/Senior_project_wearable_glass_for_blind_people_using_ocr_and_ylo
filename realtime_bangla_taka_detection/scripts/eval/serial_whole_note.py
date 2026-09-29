"""Serial blacklist on real whole-note photos (YOLO crops cached by cache_whole_note_crops.py).

The serial is read (EasyOCR bn + en, GPU) from the lower-left and upper-right serial boxes of the crop,
in both orientations (a note can be upside down). A photo is flagged if any 6-digit prefix read matches
a serial prefix of the JaalTaka TRAIN counterfeits (the blacklist used in watermark_hybrid.py).
Sets: counterfeit-set counterfeit originals (all), counterfeit-set genuine, Bangla Money 500/1000 and BanglaTaka 500/1000 (150 sampled each). Seed 42.
Output: results/watermark/serial_whole_note.json
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from roboeye.camva.notes import SPLIT_DIR, load_splits  # noqa: E402

J = ROOT / "results" / "jaal_whole"
BN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
BOXES = [(0.04, 0.76, 0.46, 0.98), (0.58, 0.10, 0.94, 0.34)]


def main() -> None:
    import easyocr
    reader = easyocr.Reader(["bn", "en"], gpu=True, verbose=False)
    splits, records = load_splits(SPLIT_DIR)
    audit = {r["note_id"]: r for r in json.loads((J / "serial_audit.json").read_text(encoding="utf-8"))["rows"]}
    blacklist = {audit[n]["serial"][:6] for n in splits["train"]
                 if int(records[n]["label"]) == 0 and n in audit and audit[n]["serial"]}
    rows = [r for r in json.loads((J / "crops_index.json").read_text(encoding="utf-8")) if r.get("crop") and not r["augmented"]]
    rng = random.Random(42)
    pick = [r for r in rows if r["set"] == "cf" and r["truth"] == "counterfeit"]
    for s, n in (("cf", 150), ("bm", 150), ("bt", 150)):
        pool = [r for r in rows if r["set"] == s and r["truth"] == "genuine"]
        rng.shuffle(pool)
        pick += pool[:n] if n else pool
    out = []
    for i, r in enumerate(pick):
        img = cv2.imread(str(J / "crops" / r["crop"]))
        reads = []
        for rot in (img, cv2.rotate(img, cv2.ROTATE_180)):
            h, w = rot.shape[:2]
            for x0, y0, x1, y1 in BOXES:
                c = cv2.resize(rot[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                t = " ".join(x for _, x, _ in reader.readtext(c)).translate(BN).replace(" ", "")
                reads += [m for m in re.findall(r"\d+", t) if 6 <= len(m) <= 8]
        hit = any(m[:6] in blacklist for m in reads)
        out.append({"set": r["set"], "truth": r["truth"], "group": r["group"], "reads": reads, "flagged": hit})
        if (i + 1) % 100 == 0:
            print(i + 1, "/", len(pick), flush=True)
    summ = {}
    for s, t in (("cf", "counterfeit"), ("cf", "genuine"), ("bm", "genuine"), ("bt", "genuine")):
        rs = [x for x in out if x["set"] == s and x["truth"] == t]
        summ[f"{s}/{t}"] = {"n": len(rs), "flagged": sum(x["flagged"] for x in rs), "any_serial_read": sum(bool(x["reads"]) for x in rs)}
        if t == "counterfeit":
            g = {}
            for x in rs:
                g.setdefault(x["group"], [0, 0])
                g[x["group"]][0] += 1
                g[x["group"]][1] += int(x["flagged"])
            summ[f"{s}/{t}"]["by_group_n_flagged"] = g
    (ROOT / "results/watermark/serial_whole_note.json").write_text(json.dumps({"blacklist": sorted(blacklist), "summary": summ, "rows": out}, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
