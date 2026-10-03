"""EAST, OpenCV DB (DBNet) and a YOLO text detector against CRAFT, inside the same final OCR pipeline.

Validation set, 3 image seeds. Then the single best alternative on validation is read once on the held-out test set
(--test). CRAFT's numbers are the stored final-pipeline results.
  python ocr_bench/run_detectors.py          # validation
  python ocr_bench/run_detectors.py --test   # after validation: best alternative on test, once
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_bench import alt_detectors as A, bench as B  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "ocr_bench"
NAMES = ("east", "db_td500", "db_ic15", "yolo")


def main() -> None:
    if "--test" in sys.argv:
        val = json.loads((OUT / "detectors_val.json").read_text(encoding="utf-8"))
        best = min(val["alternatives"], key=val["alternatives"].get)
        r = B.score(A.make_reader(best), "test")
        B.save(r, OUT / f"test_det_{best}.json", {"set": "test", "method": f"final pipeline with {best} detector", "read_once": True,
                                                  "chosen_on": "validation (best alternative detector)"})
        print(B.line(f"test det_{best}", r))
        return
    res = {}
    for name in NAMES:
        r = B.score(A.make_reader(name), "val")
        B.save(r, OUT / f"val_det_{name}.json", {"set": "val", "method": f"final pipeline with {name} detector", "detector_settings": "published defaults"})
        res[name] = r["overall"]["cer"]
        print(B.line(f"val det_{name}", r), flush=True)
    craft = json.loads((OUT / "val_t5_rules.json").read_text(encoding="utf-8"))["overall"]["cer"]
    (OUT / "detectors_val.json").write_text(json.dumps({"craft_final_val_cer": craft, "alternatives": res}, indent=1), encoding="utf-8")
    print("CRAFT (final pipeline) val CER", round(100 * craft, 2), "| alternatives", {k: round(100 * v, 2) for k, v in res.items()})


if __name__ == "__main__":
    main()
