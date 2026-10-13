"""A fine-tuned letter recogniser for EasyOCR (same network, new weights), loaded into the glass's reader.

The weights come from ocr_bench/train_real_words.py: EasyOCR's Bangla+English recogniser fine-tuned on real word
photographs mixed with rendered words. Evidence and the rule by which a model was accepted are in config.py next
to OCR_RECOGNIZER_PATH.

install(reader) swaps the weights in; on any problem it leaves the released recogniser in place.
"""
from __future__ import annotations

import logging
import os

import torch

import config

logger = logging.getLogger("smart_glass.ocr")


def install(reader, path: str | None = None) -> str:
    path = path if path is not None else getattr(config, "OCR_RECOGNIZER_PATH", "")
    if not path or not os.path.isfile(path):
        return "released"
    try:
        sd = torch.load(path, map_location="cpu")
        if getattr(reader, "device", "cpu") == "cpu":
            from easyocr.model.model import Model          # generation-1 recogniser used by the Bangla model
            m = Model(input_channel=1, output_channel=512, hidden_size=512, num_class=sd["Prediction.weight"].shape[0])
            m.load_state_dict(sd)
            m.eval()
            # the same step EasyOCR applies on the CPU: 8-bit weights for the LSTM and linear layers
            m = torch.quantization.quantize_dynamic(m, dtype=torch.qint8)
            reader.recognizer = m
        else:
            net = reader.recognizer.module if hasattr(reader.recognizer, "module") else reader.recognizer
            net.load_state_dict(sd)
            net.eval()
        logger.info("Letter recogniser: fine-tuned weights (%s)", os.path.basename(path))
        return os.path.basename(path)
    except Exception as exc:
        logger.warning("Fine-tuned recogniser not used (%s); the released one stays", exc)
        return "released"
