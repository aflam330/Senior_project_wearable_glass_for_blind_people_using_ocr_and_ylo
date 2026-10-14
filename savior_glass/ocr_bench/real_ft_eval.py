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

Added on 2026-10-14, after seed 42 was found NOT admissible (rendered validation 3.82 % against 2.58 %; English
5.84 % against 3.27 %) and before any other fine-tuned model was scored:
  Step 1b, weight blending. If no fully fine-tuned model is admissible, the weights are blended with the released
    ones, theta = (1 - a) * released + a * fine-tuned, for a = 0.5 and, if still none is admissible, a = 0.25,
    for all three seeds. The same admissibility rule and the same choice (lowest real-validation CER) apply.
    This is weight-space interpolation as used to keep accuracy on the original data after fine-tuning.

    python -m ocr_bench.real_ft_eval
Output: results/ocr_bench/real_ft_eval.json
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from . import bench as B
from . import train_task7 as T7
from .page_bench import OUT, _ocr_mode
from .real_words_tune import fetch, score_real
from .train_real_words import split

RES = OUT / "real_ft_eval.json"
SEEDS = (42, 43, 44)


def released_weights(reader):
    """EasyOCR's released recogniser weights as a plain state dict (the file stores them with a 'module.' prefix)."""
    import torch
    path = Path(reader.model_storage_directory) / "bengali.pth"
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


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
    base_sd = released_weights(m._reader)
    tmp = Path(tempfile.gettempdir()) / "sara_blend.pth"

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

    def weights_of(name):
        """State dict for 'fine-tuned seed S' or 'blend a seed S'."""
        parts = name.split()
        ft = torch.load(T7.FT / f"recognizer_real_seed{parts[-1]}.pth", map_location="cpu")
        if parts[0] != "blend":
            return ft
        a = float(parts[1])
        return {k: ((1 - a) * base_sd[k].float() + a * v.float()).to(v.dtype) if v.is_floating_point() else v for k, v in ft.items()}

    def use(name):
        if name == "released":
            m._reader.recognizer = released
        else:
            torch.save(weights_of(name), tmp)
            assert FT.install(m._reader, str(tmp)) != "released", name

    config.OCR_ROTATE_FALLBACK = False
    use("released")
    base = evaluate("released")

    def admissible(names):
        return [n for n in names if res["validation"][n]["rendered_val_cer"] - base["rendered_val_cer"] <= 0.005]

    stages = [[f"fine-tuned seed {s}" for s in SEEDS], [f"blend 0.5 seed {s}" for s in SEEDS], [f"blend 0.25 seed {s}" for s in SEEDS]]
    rec, log = "released", []
    for names in stages:
        for n in names:
            use(n)
            evaluate(n)
        ok = admissible(names)
        log.append({"candidates": names, "admissible": ok})
        if ok:
            best = min(ok, key=lambda n: res["validation"][n]["real_val"]["mean_cer"])
            if res["validation"][best]["real_val"]["mean_cer"] < base["real_val"]["mean_cer"]:
                rec = best
            break
    res["step1"] = {"stages": log, "recogniser": rec}
    RES.write_text(json.dumps(res, indent=1), encoding="utf-8")
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
    if rec != "released":                                   # keep the chosen weights where the glass can load them
        torch.save(weights_of(rec), T7.FT / "recognizer_real_chosen.pth")
    use("released")
    config.OCR_ROTATE_FALLBACK = False
    RES.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "validation"}, indent=1))


if __name__ == "__main__":
    main()
