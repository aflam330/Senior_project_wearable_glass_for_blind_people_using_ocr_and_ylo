"""Read the held-out TEST set once per finished method (and the legacy lexicon / select sets). No choice is made here.

Methods (each fixed on val before this runs):
  baseline          the old offline evaluation path (preprocess, EasyOCR beamsearch, boxes in EasyOCR order, no filter)
  glass_app         the app's OCR mode as shipped (adds reading order, confidence >= 0.4, text filter, lexicon repair)
  t1_region         + Task 1 region choice
  t2_prep           + Task 2 preprocessing
  t4_params         + Task 4 parameters (= Task 6 choice: single scale)
  final             + Task 5 post-processing
  tesseract_block   best Task 3 engine other than EasyOCR (Tesseract 5, Task 1 text block, page mode from val)
Usage: python ocr_bench/run_test.py [--finetuned]   (--finetuned: the final pipeline with each Task 7 seed)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr_bench import bench as B, engines as E, final_pipeline as F, pipelines as P, text_region as T  # noqa: E402
from ocr_bench.run_task4 import prep_fn  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "ocr_bench"


def methods():
    c = F.load_choices()
    t1 = json.loads((OUT / "task1_best.json").read_text(encoding="utf-8"))["best"]["config"]
    t3 = min(json.loads((OUT / "task3_val.json").read_text(encoding="utf-8")), key=lambda r: r["val_cer"])
    prep = prep_fn(c["prep"])
    kw = {k: v for k, v in c["params"].items() if k != "min_conf"}
    return {
        "baseline": lambda im: P.baseline(im),
        "glass_app": P.glass_app,
        "t1_region": lambda im: T.read_region(im, t1["detector"], min_conf=t1["min_conf"], second_pass=t1["second_pass"]),
        "t2_prep": lambda im: T.read_region(im, t1["detector"], min_conf=t1["min_conf"], second_pass=t1["second_pass"], prep=prep),
        "t4_params": lambda im: T.read_region(im, t1["detector"], min_conf=c["params"]["min_conf"], second_pass=t1["second_pass"], prep=prep, **kw),
        "final": F.make_reader(c),
        f"t3_{t3['tag']}": (lambda im, p=t3["config"]["psm"]: E.tesseract_on_block(im, p)) if t3["config"]["where"] == "block"
        else (lambda im, p=t3["config"]["psm"]: E.tesseract(im, p)),
    }


def main() -> None:
    if "--finetuned" in sys.argv:
        c = F.load_choices()
        for seed in (42, 43, 44):
            F.use_recognizer(Path(__file__).resolve().parent / "finetuned" / f"recognizer_seed{seed}.pth")
            for name in ("test", "lexicon", "select"):
                r = B.score(F.make_reader(c), name)
                B.save(r, OUT / f"{name}_final_ft_seed{seed}.json", {"set": name, "method": "final + Task 7 recognizer", "seed": seed, "choices": c})
                print(B.line(f"{name} final_ft_seed{seed}", r), flush=True)
        return
    for mname, fn in methods().items():
        for name in ("test", "lexicon", "select"):
            if name != "test" and mname not in ("baseline", "glass_app", "final"):
                continue
            r = B.score(fn, name)
            B.save(r, OUT / f"{name}_{mname}.json", {"set": name, "method": mname, "read_once": True})
            print(B.line(f"{name} {mname}", r), flush=True)


if __name__ == "__main__":
    main()
