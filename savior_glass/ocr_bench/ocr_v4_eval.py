"""OCR v4: second text-detection pass and the version-2 recogniser, chosen on validation, tested once.

Same data, reading function and rule as ocr_v3_eval.py (the glass's OCRMode._read_v2 on the CPU):
  admissible = rendered-validation CER within 0.5 points of A; chosen = admissible candidate with the lowest real-
  validation CER (mean of the two scripts) if lower than A's. Only then are A and the chosen candidate read on the
  locked test words (500 + 500) and the rendered test set, once each.
Also reported: on 40 text-free-or-not COCO validation-half photos (the rendered benchmark's background pool), on how
many each candidate speaks any text - the second pass must not make the glass talk about scenes much more often.
Candidates (declared 2026-11-23, after the design probe ocr_bench/probe_second_pass.py on validation words only):
  A    current glass settings (A of ocr_v3_eval.py)
  S    A + second pass when nothing is kept: picture at half size (mag_ratio 0.5), text 0.6, low-text 0.25; kept if
       the mean confidence is >= 0.2
  V    A with the version-2 recogniser (train_real_words_v2.py) blended half and half with the released weights
  SV   S and V together
V and SV are scored only if ocr_bench/finetuned/recognizer_real_v2_seed42.pth exists.

    python -m ocr_bench.ocr_v4_eval
Output: results/ocr_bench/ocr_v4_eval.json
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2

from . import bench as B
from . import train_task7 as T7
from .ocr_v3_eval import score
from .page_bench import OUT, _ocr_mode
from .real_ft_eval import released_weights
from .real_words_tune import fetch
from .train_real_words import split

RES = OUT / "ocr_v4_eval.json"
SECOND = {"mag_ratio": 0.5, "text_threshold": 0.6, "low_text": 0.25}
V2 = T7.FT / "recognizer_real_v2_seed42.pth"
OCT = "models/ocr_recognizer_real_blend"          # A's recogniser (the glass's default until 2026-11-23)


def main(threads=4):
    import torch
    import config
    import ocr_recognizer_ft as FT
    from ocr_text_region import clean_text
    torch.set_num_threads(threads)
    _, va, te = split()
    m = _ocr_mode()
    m.speak_line = None
    import os
    assert FT.install(m._reader, os.path.join(config.BASE_DIR, OCT)) != "released"
    deployed = m._reader.recognizer                      # A's recogniser (October blend 0.5 seed 44)
    base_sd = released_weights(m._reader)
    tmp = Path(tempfile.gettempdir()) / "sara_v2_blend.pth"
    cands = ["A", "S"] + (["V", "SV"] if V2.exists() else [])

    def read(frame):
        return clean_text(" ".join(m._read_v2(frame)))

    def use(name):
        config.OCR_ADD_MARGIN, config.OCR_ROTATE_LOWCONF, config.OCR_ROTATE_FALLBACK = 0.1, 0.0, False
        config.OCR_SECOND_PASS = SECOND if "S" in name else None
        config.OCR_SECOND_PASS_MIN_CONF = 0.2
        if "V" in name:
            ft = torch.load(V2, map_location="cpu")
            sd = {k: ((0.5 * base_sd[k].float() + 0.5 * v.float()).to(v.dtype) if v.is_floating_point() else v) for k, v in ft.items()}
            torch.save(sd, tmp)
            assert FT.install(m._reader, str(tmp)) != "released"
        else:
            m._reader.recognizer = deployed

    scenes = [cv2.resize(cv2.imread(str(p)), (640, 480)) for p in sorted(B.COCO.glob("*.jpg"))[:len(list(B.COCO.glob("*.jpg"))) // 2][:40]]
    res = json.loads(RES.read_text(encoding="utf-8")) if RES.exists() else {"date": "2026-11-23", "validation": {}}
    val_items = fetch(va)
    for name in cands:
        if name in res["validation"]:
            continue
        use(name)
        rend = B.score(read, "val")
        res["validation"][name] = {"real_val": score(read, val_items), "rendered_val_cer": round(rend["overall"]["cer"], 4),
                                   "rendered_val_wer": round(rend["overall"]["wer"], 4),
                                   "coco_scenes_with_spoken_text": sum(bool(read(s)) for s in scenes)}
        RES.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(name, json.dumps(res["validation"][name], ensure_ascii=False), flush=True)
    V = res["validation"]
    ok = [n for n in V if V[n]["rendered_val_cer"] - V["A"]["rendered_val_cer"] <= 0.005]
    best = min(ok, key=lambda n: V[n]["real_val"]["mean_cer"])
    chosen = best if V[best]["real_val"]["mean_cer"] < V["A"]["real_val"]["mean_cer"] else "A"
    res["decision"] = {"candidates": list(V), "admissible": ok, "chosen": chosen}
    RES.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("decision", res["decision"], flush=True)


def test():
    """Read the locked test words once, for A and the chosen candidate (call after main())."""
    import torch
    import config
    res = json.loads(RES.read_text(encoding="utf-8"))
    chosen = res["decision"]["chosen"]
    if chosen == "A" or "test" in res:
        print("nothing to test" if chosen == "A" else "already tested")
        return
    torch.set_num_threads(4)
    _, _, te = split()
    te_items = fetch(te)
    m = _ocr_mode()
    m.speak_line = None
    res["test"] = {}
    from ocr_text_region import clean_text
    import ocr_recognizer_ft as FT
    import os
    assert FT.install(m._reader, os.path.join(config.BASE_DIR, OCT)) != "released"
    deployed = m._reader.recognizer
    base_sd = released_weights(m._reader)
    tmp = Path(tempfile.gettempdir()) / "sara_v2_blend.pth"
    for name in ("A", chosen):
        config.OCR_SECOND_PASS = SECOND if "S" in name else None
        config.OCR_SECOND_PASS_MIN_CONF = 0.2
        if "V" in name:
            ft = torch.load(V2, map_location="cpu")
            sd = {k: ((0.5 * base_sd[k].float() + 0.5 * v.float()).to(v.dtype) if v.is_floating_point() else v) for k, v in ft.items()}
            torch.save(sd, tmp)
            FT.install(m._reader, str(tmp))
        else:
            m._reader.recognizer = deployed

        def read(frame):
            return clean_text(" ".join(m._read_v2(frame)))
        real, rows = score(read, te_items, keep_rows=True)
        rend = B.score(read, "test")
        worst = sorted(rows, key=lambda r: -r["cer"])
        res["test"][name] = {"real_words": real,
                             "rendered": {"cer": round(rend["overall"]["cer"], 4), "cer_bn": round(rend["bn"]["cer"], 4),
                                          "cer_en": round(rend["en"]["cer"], 4), "wer": round(rend["overall"]["wer"], 4),
                                          "n_images": rend["overall"]["n"]},
                             "error_examples": [r for r in worst if r["hyp"]][:8] + [r for r in worst if not r["hyp"]][:4]}
        RES.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(name, json.dumps(res["test"][name]["real_words"], ensure_ascii=False), res["test"][name]["rendered"], flush=True)
    config.OCR_SECOND_PASS = None


if __name__ == "__main__":
    import sys
    test() if sys.argv[1:] == ["test"] else main()
