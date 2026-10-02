"""Task 3 runner: other OCR engines on the validation set (3 seeds). Settings (Tesseract page mode) chosen on val.

Tesseract 5.5.2 (tessdata_best ben+eng), whole frame: page modes 3, 6, 11; on the Task 1 text block: modes 6, 7.
TrOCR on the text block with the language given (oracle; easier task than the others).
PaddleOCR has no Bangla model; its English numbers come from paddle_run.py (Python 3.10).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_bench import bench as B, engines as E  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "ocr_bench"


class LangAware:
    """score() passes only the image; this wrapper looks up the language of the current item (oracle)."""

    def __init__(self, name):
        self.items = {}
        for seed in B.SEEDS:
            for it in B.build(name, seed):
                self.items[it["png"]] = it["lang"]


def run(tag, fn, cfg, name="val"):
    r = B.score(fn, name)
    B.save(r, OUT / f"{name}_t3_{tag}.json", {"set": name, "task": 3, "config": cfg})
    print(B.line(tag, r), flush=True)
    return r["overall"]["cer"], tag, cfg


def main() -> None:
    res = []
    for psm in (3, 6, 11):
        res.append(run(f"tesseract_frame_psm{psm}", lambda im, p=psm: E.tesseract(im, p), {"engine": "tesseract", "where": "frame", "psm": psm}))
    for psm in (6, 7):
        res.append(run(f"tesseract_block_psm{psm}", lambda im, p=psm: E.tesseract_on_block(im, p), {"engine": "tesseract", "where": "block", "psm": psm}))
    # TrOCR needs the language: map image bytes -> lang for the val items
    lang = {}
    for seed in B.SEEDS:
        for it in B.build("val", seed):
            lang[B.decode(it).tobytes()] = it["lang"]
    res.append(run("trocr_block_oracle_lang", lambda im: E.trocr_on_block(im, lang[im.tobytes()]),
                   {"engine": "trocr", "where": "block", "lang": "oracle"}))
    (OUT / "task3_val.json").write_text(json.dumps([{"val_cer": c, "tag": t, "config": k} for c, t, k in res], indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
