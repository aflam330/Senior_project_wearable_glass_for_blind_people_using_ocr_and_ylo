"""The OCR pipeline assembled from the validation choices of Tasks 1-6 (read from results/ocr_bench/task*_best.json).

  Task 1  read the whole frame, keep the dominant text block (CRAFT boxes, confidence >= min_conf), order into lines
  Task 2  preprocessing chosen on val
  Task 4  EasyOCR parameters chosen on val
  Task 6  single scale (multi-scale did not help on val)
  Task 5  post-processing chosen on val
Optional: a fine-tuned recognizer from Task 7 (state dict) swapped into the EasyOCR reader.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import pipelines as P, postprocess as PO, text_region as T
from .run_task4 import prep_fn

RES = Path(__file__).resolve().parents[1] / "results" / "ocr_bench"


def load_choices():
    t4 = json.loads((RES / "task4_best.json").read_text(encoding="utf-8"))
    t5 = json.loads((RES / "task5_best.json").read_text(encoding="utf-8"))
    return {"task1": t4["task1"], "prep": t4["prep"], "params": t4["config"], "post": t5["best"]}


def use_recognizer(state_dict_path=None):
    """Swap fine-tuned weights into the cached CRAFT reader (None restores nothing: call before first use)."""
    import torch
    r = P.reader("craft")
    m = r.recognizer.module if hasattr(r.recognizer, "module") else r.recognizer
    if state_dict_path:
        m.load_state_dict(torch.load(state_dict_path, map_location="cpu"))
    return r


def make_reader(choices=None, post=True):
    c = choices or load_choices()
    prep = prep_fn(c["prep"])
    kw = {k: v for k, v in c["params"].items() if k != "min_conf"}
    fix = PO.METHODS[c["post"]] if post else (lambda t: t)

    def read(bgr):
        return fix(T.read_region(bgr, c["task1"]["detector"], min_conf=c["params"]["min_conf"],
                                 second_pass=c["task1"]["second_pass"], prep=prep, **kw))
    return read
