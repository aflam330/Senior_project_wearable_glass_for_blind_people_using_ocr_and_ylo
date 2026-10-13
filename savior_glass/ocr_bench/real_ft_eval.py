"""Decide on (1) the recognisers fine-tuned on real words and (2) the rotation retry, through the glass's own
reading function (OCRMode._read_v2, CPU, 8-bit LSTM as on the Pi), then test the final choice once.

Rule, fixed before the run (2026-10-13):
  Step 1, recogniser. The released recogniser and the three fine-tuned ones (seeds 42, 43, 44) are scored on the
    real validation words (150 + 150) and on the rendered benchmark's validation set (432 images). A fine-tuned
    model is admissible if its rendered-validation CER is within 0.5 points of the released one's. The admissible
    model with the lowest real-validation CER (mean of the two scripts) wins if it beats the released one there.
  Step 2, rotation retry (when nothing is read, try the picture turned by 90 degrees each way). Scored with the
    recogniser of step 1. Adopted if it lowers the real-validation CER and the rendered-validation CER stays within
    0.5 points of the released recogniser's.
  Test. The final configuration is read once on the 1,000 real test words and once on the rendered test set.

    python -m ocr_bench.real_ft_eval
Output: results/ocr_bench/real_ft_eval.json
"""
from __future__ import annotations

import json

from . import bench as B
from . import train_task7 as T7
from .page_bench import OUT, _ocr_mode
from .real_words_tune import fetch, score_real
from .train_real_words import split

RES = OUT / "real_ft_eval.json"
SEEDS = (42, 43, 44)


def main(threads=4):
    import torch
    import config
    import ocr_recognizer_ft as FT
    from ocr_text_region import clean_text
    torch.set_num_threads(threads)
    _, va, te = split()
    val_items = fetch(va)
    m = _ocr_mode()
    m.speak_line = None
    released = m._reader.recognizer

    def read(frame):
        return clean_text(" ".join(m._read_v2(frame)))

    res = json.loads(RES.read_text(encoding="utf-8")) if RES.exists() else {"date": "2026-10-13", "validation": {}}

    def evaluate(name):
        if name not in res["validation"]:
            real = score_real(read, val_items)
            rend = B.score(read, "val")
            res["validation"][name] = {"real_val": real, "rendered_val_cer": round(rend["overall"]["cer"], 4),
                                       "rendered_val_cer_bn": round(rend["bn"]["cer"], 4), "rendered_val_cer_en": round(rend["en"]["cer"], 4)}
            RES.write_text(json.dumps(res, indent=1), encoding="utf-8")
            print(name, res["validation"][name], flush=True)
        return res["validation"][name]

    def use(name):
        if name == "released":
            m._reader.recognizer = released
        else:
            assert FT.install(m._reader, str(T7.FT / f"recognizer_real_seed{name.split()[-1]}.pth")) != "released"

    config.OCR_ROTATE_FALLBACK = False
    use("released")
    base = evaluate("released")
    for s in SEEDS:
        use(f"fine-tuned seed {s}")
        evaluate(f"fine-tuned seed {s}")
    ok = [f"fine-tuned seed {s}" for s in SEEDS
          if res["validation"][f"fine-tuned seed {s}"]["rendered_val_cer"] - base["rendered_val_cer"] <= 0.005]
    rec = min(ok, key=lambda n: res["validation"][n]["real_val"]["mean_cer"]) if ok else "released"
    if res["validation"][rec]["real_val"]["mean_cer"] >= base["real_val"]["mean_cer"]:
        rec = "released"
    res["step1"] = {"admissible": ok, "recogniser": rec}
    use(rec)
    config.OCR_ROTATE_FALLBACK = True
    rot = evaluate(f"{rec} + rotation retry")
    cur = res["validation"][rec]
    use_rot = bool(rot["real_val"]["mean_cer"] < cur["real_val"]["mean_cer"] and rot["rendered_val_cer"] - base["rendered_val_cer"] <= 0.005)
    res["step2"] = {"rotation_retry": use_rot}
    config.OCR_ROTATE_FALLBACK = use_rot
    if (rec != "released" or use_rot) and "test" not in res:
        real_t = score_real(read, fetch(te))
        t = B.score(read, "test")
        res["test"] = {"configuration": {"recogniser": rec, "rotation_retry": use_rot}, "real_words": real_t,
                       "rendered": {"cer": round(t["overall"]["cer"], 4), "cer_bn": round(t["bn"]["cer"], 4),
                                    "cer_en": round(t["en"]["cer"], 4), "wer": round(t["overall"]["wer"], 4)}}
    use("released")
    config.OCR_ROTATE_FALLBACK = False
    RES.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "validation"}, indent=1))


if __name__ == "__main__":
    main()
