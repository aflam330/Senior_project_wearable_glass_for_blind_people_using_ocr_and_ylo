"""Task 3 engines: Tesseract 5 (ben+eng), TrOCR (English printed; community Bangla model). PaddleOCR: paddle_run.py (py3.10)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import pipelines as P, text_region as T

TESSDATA = str(Path(__file__).resolve().parent / "tessdata")


@lru_cache(maxsize=8)
def _tess(psm: int):
    import tesserocr
    return tesserocr.PyTessBaseAPI(path=TESSDATA, lang="ben+eng", psm=psm, oem=tesserocr.OEM.LSTM_ONLY)


def tesseract(gray_or_bgr, psm=6) -> str:
    img = gray_or_bgr if gray_or_bgr.ndim == 2 else cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    api = _tess(psm)
    api.SetImage(Image.fromarray(img))
    return " ".join(api.GetUTF8Text().split())


def block_crop(bgr, min_conf=0.3, margin=0.25):
    """The dominant text block (Task 1, CRAFT) as a grey crop, 1200 px wide, plus its line boxes."""
    img = P.glass_preprocess(bgr)
    det = P.reader("craft").readtext(img, detail=1, paragraph=False, decoder="beamsearch", beamWidth=5,
                                     width_ths=0.7, height_ths=0.7, canvas_size=2560)
    block = T.dominant_block([b for b in T._boxes(det) if b["c"] >= min_conf])
    if not block:
        return None, []
    hs = np.median([b["h"] for b in block])
    x0 = int(max(0, min(b["x0"] for b in block) - margin * 2 * hs)); x1 = int(min(img.shape[1], max(b["x1"] for b in block) + margin * 2 * hs))
    y0 = int(max(0, min(b["y0"] for b in block) - margin * hs)); y1 = int(min(img.shape[0], max(b["y1"] for b in block) + margin * hs))
    return img[y0:y1, x0:x1], block


def tesseract_on_block(bgr, psm=7):
    crop, _ = block_crop(bgr)
    if crop is None or crop.size == 0:
        return ""
    s = 1200 / crop.shape[1]
    return tesseract(cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA), psm)


@lru_cache(maxsize=4)
def _trocr(model_id: str, processor_id: str):
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    if "Bangla" in model_id:  # this repo's processor config names a removed class; build the same parts by hand
        from transformers import AutoTokenizer, ViTImageProcessor
        proc = TrOCRProcessor(image_processor=ViTImageProcessor(size={"height": 224, "width": 224}, resample=3),
                              tokenizer=AutoTokenizer.from_pretrained(processor_id))
    else:
        proc = TrOCRProcessor.from_pretrained(processor_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return proc, model


def trocr_on_block(bgr, lang):
    """TrOCR reads the block crop as one line. The language is GIVEN (oracle): EN -> microsoft/trocr-base-printed,
    BN -> nightsagittariuswolf/SWIN_TrOCR_Bangla_model. This is easier than the other engines' task."""
    import torch
    crop, _ = block_crop(bgr)
    if crop is None or crop.size == 0:
        return ""
    if lang == "en":
        proc, model = _trocr("microsoft/trocr-base-printed", "microsoft/trocr-base-printed")
    else:
        proc, model = _trocr("nightsagittariuswolf/SWIN_TrOCR_Bangla_model", "nightsagittariuswolf/SWIN_trOCR_Bangla_processor")
    pix = proc(images=Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)), return_tensors="pt").pixel_values.to(model.device)
    with torch.inference_mode():
        ids = model.generate(pix, max_new_tokens=64)
    return " ".join(proc.batch_decode(ids, skip_special_tokens=True)[0].split())
