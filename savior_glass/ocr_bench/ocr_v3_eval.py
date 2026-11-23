"""OCR v3 candidates (crop margin, low-confidence rotation retry), chosen on validation, tested once.

Same data and rule as ocr_bench/real_ft_eval.py (the glass's own reading function OCRMode._read_v2, CPU, the
deployed fine-tuned recogniser):
  real validation: the 150 Bangla + 150 Latin-script ICDAR 2019 MLT words of split() (disjoint from the test words and
    from the words the recogniser was fine-tuned on); rendered validation: the benchmark's val set (432 images).
  Rule, fixed before the run (2026-11-23): a candidate is admissible if its rendered-validation CER is within 0.5
    points of A's; the admissible candidate with the lowest real-validation CER (mean of the two scripts) is chosen if
    it is lower than A's. Only then are A and the chosen one read on the locked test words (500 + 500) and the rendered
    test set, once each, for before/after. If A wins, nothing changes.
  Candidates:
    A  current: margin 0.1, rotation retry off
    M  margin 0.2
    R  rotation retry when nothing is read or the main block's mean confidence is below 0.3
    MR M and R together
Also reported (scoring only, changes nothing): CER with zero-width joiner / non-joiner removed from both sides, to show
whether Unicode handling moves the score, and median latency per word frame on this CPU (4 threads).

    python -m ocr_bench.ocr_v3_eval
Output: results/ocr_bench/ocr_v3_eval.json
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import bench as B
from .page_bench import OUT, _ocr_mode
from .real_words import frame_of
from .real_words_tune import fetch
from .train_real_words import split

RES = OUT / "ocr_v3_eval.json"
CANDIDATES = {"A": (0.1, 0.0), "M": (0.2, 0.0), "R": (0.1, 0.3), "MR": (0.2, 0.3)}
ZW = dict.fromkeys(map(ord, "‌‍"))


def score(read, items, keep_rows=False):
    out, rows = {}, []
    for lang in ("bn", "en"):
        rs = []
        for it in items:
            if it["lang"] != lang:
                continue
            fr = frame_of(it["img"])
            t0 = time.perf_counter()
            h = read(fr)
            ms = (time.perf_counter() - t0) * 1000
            rs.append((it["gt"], h, ms))
        n_words = sum(len(B.nfc(g).split()) for g, _, _ in rs)
        out[lang] = {"n_images": len(rs), "n_words": n_words,
                     "cer": round(float(np.mean([B.cer(g, h) for g, h, _ in rs])), 4),
                     "wer": round(float(np.mean([B.wer(g, h) for g, h, _ in rs])), 4),
                     "cer_without_zero_width": round(float(np.mean([B.cer(g.translate(ZW), h.translate(ZW)) for g, h, _ in rs])), 4),
                     "nothing_read": round(float(np.mean([h.strip() == "" for _, h, _ in rs])), 4),
                     "word_exact": round(float(np.mean([B.nfc(g) == B.nfc(h) for g, h, _ in rs])), 4),
                     "latency_ms_median": round(float(np.median([ms for *_, ms in rs])), 1)}
        rows += [{"lang": lang, "gt": g, "hyp": h, "cer": round(B.cer(g, h), 3)} for g, h, _ in rs]
    out["mean_cer"] = round((out["bn"]["cer"] + out["en"]["cer"]) / 2, 4)
    return (out, rows) if keep_rows else out


def main(threads=4):
    import torch
    import config
    from ocr_text_region import clean_text
    torch.set_num_threads(threads)
    _, va, te = split()
    m = _ocr_mode()
    m.speak_line = None

    def read(frame):
        return clean_text(" ".join(m._read_v2(frame)))

    def use(name):
        config.OCR_ADD_MARGIN, config.OCR_ROTATE_LOWCONF = CANDIDATES[name]
        config.OCR_ROTATE_FALLBACK = False

    res = json.loads(RES.read_text(encoding="utf-8")) if RES.exists() else {"date": "2026-11-23", "validation": {}}
    val_items = fetch(va)
    for name in CANDIDATES:
        if name in res["validation"]:
            continue
        use(name)
        rend = B.score(read, "val")
        res["validation"][name] = {"setting": dict(zip(("add_margin", "rotate_lowconf"), CANDIDATES[name])),
                                   "real_val": score(read, val_items),
                                   "rendered_val_cer": round(rend["overall"]["cer"], 4),
                                   "rendered_val_wer": round(rend["overall"]["wer"], 4)}
        RES.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(name, json.dumps(res["validation"][name], ensure_ascii=False), flush=True)
    V = res["validation"]
    ok = [n for n in CANDIDATES if V[n]["rendered_val_cer"] - V["A"]["rendered_val_cer"] <= 0.005]
    best = min(ok, key=lambda n: V[n]["real_val"]["mean_cer"])
    chosen = best if V[best]["real_val"]["mean_cer"] < V["A"]["real_val"]["mean_cer"] else "A"
    res["decision"] = {"admissible": ok, "chosen": chosen}
    if chosen != "A" and "test" not in res:
        te_items = fetch(te)
        res["test"] = {}
        for name in ("A", chosen):
            use(name)
            real, rows = score(read, te_items, keep_rows=True)
            rend = B.score(read, "test")
            worst = sorted(rows, key=lambda r: -r["cer"])
            res["test"][name] = {"real_words": real,
                                 "rendered": {"cer": round(rend["overall"]["cer"], 4), "cer_bn": round(rend["bn"]["cer"], 4),
                                              "cer_en": round(rend["en"]["cer"], 4), "wer": round(rend["overall"]["wer"], 4),
                                              "n_images": rend["overall"]["n"]},
                                 "error_examples": [r for r in worst if r["hyp"]][:8] + [r for r in worst if not r["hyp"]][:4]}
            RES.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    use("A")
    RES.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "validation"}, indent=1, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
