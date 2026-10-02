"""Score other OCR engines on the validation phrase list only.

Does not open the 24-phrase set or the 80-image set, and does not change
the cited EasyOCR pipeline (slope_ths=0.2).
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_ocr_improve import VAL_BN, VAL_EN, agg, glass_gray, join_text, read_kwargs, samples  # noqa: E402
from eval_ocr_offline import levenshtein  # noqa: E402
from ocr_repair import repair_ocr_text, tesseract_available, tesseract_read  # noqa: E402

OUT = ROOT / "results" / "ocr_engines_val.json"


def cer_rows(pairs):
    rows = []
    for gt, hyp, lang in pairs:
        rows.append({"lang": lang, "cer": levenshtein(gt, hyp) / max(len(gt), 1), "wer": 0.0})
    return agg(rows)


def main() -> None:
    blocks = [samples(VAL_BN, VAL_EN, 24, seed) for seed in (9000, 9100, 9200)]
    data = [row for block in blocks for row in block]
    report = {"n": len(data), "sets_opened": "validation phrases only"}

    # Lexicon repair on the cited EasyOCR read. These phrases are not in the lexicon.
    import easyocr
    from modes.ocr_mode import OCRMode
    glass_gray.ocr = OCRMode()
    reader = easyocr.Reader(["bn", "en"], gpu=True, verbose=False)
    kw = read_kwargs("slope_0_2")
    raw_pairs, fixed_pairs = [], []
    t0 = time.perf_counter()
    for gt, bgr, lang in data:
        hyp = join_text(reader.readtext(glass_gray(bgr), **kw))[0]
        raw_pairs.append((gt, hyp, lang))
        fixed_pairs.append((gt, repair_ocr_text(hyp), lang))
    report["easyocr_slope_lexicon"] = {
        "raw": cer_rows(raw_pairs),
        "repaired": cer_rows(fixed_pairs),
        "seconds": round(time.perf_counter() - t0, 2),
        "phrases_in_lexicon": False,
    }
    print("lexicon", report["easyocr_slope_lexicon"]["raw"]["overall"]["cer"],
          report["easyocr_slope_lexicon"]["repaired"]["overall"]["cer"], flush=True)

    report["tesseract_on_path"] = tesseract_available() or shutil.which("tesseract") is not None
    if report["tesseract_on_path"]:
        pairs = []
        t0 = time.perf_counter()
        for gt, bgr, lang in data:
            hyp = tesseract_read(glass_gray(bgr))
            pairs.append((gt, hyp, lang))
        report["tesseract"] = {"summary": cer_rows(pairs), "seconds": round(time.perf_counter() - t0, 2)}
        print("tesseract", report["tesseract"]["summary"]["overall"]["cer"], flush=True)
    else:
        report["tesseract"] = {"error": "tesseract binary not on PATH; pytesseract is installed"}

    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch
        from PIL import Image
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        pairs = []
        t0 = time.perf_counter()
        for gt, bgr, lang in data:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            pixel = processor(images=image, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                ids = model.generate(pixel, max_new_tokens=32)
            hyp = processor.batch_decode(ids, skip_special_tokens=True)[0]
            pairs.append((gt, hyp, lang))
        report["trocr_base_printed"] = {
            "model": "microsoft/trocr-base-printed",
            "note": "English printed model. It is not a Bangla recognizer.",
            "device": device,
            "summary": cer_rows(pairs),
            "seconds": round(time.perf_counter() - t0, 2),
            "examples": [{"gt": a, "hyp": b, "lang": c} for a, b, c in pairs[:4]],
        }
        print("trocr", report["trocr_base_printed"]["summary"]["overall"]["cer"],
              "bn", report["trocr_base_printed"]["summary"]["bn"]["cer"], flush=True)
    except Exception:
        report["trocr_base_printed"] = {"error": traceback.format_exc()}
        print("trocr failed", flush=True)

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
