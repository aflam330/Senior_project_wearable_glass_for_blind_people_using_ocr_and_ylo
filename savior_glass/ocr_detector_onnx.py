"""The text detector (CRAFT) of EasyOCR, run through ONNX Runtime on the CPU.

Measured on a laptop CPU limited to four threads (ocr_bench/detector_onnx.py, results/ocr_bench/detector_onnx.json,
2026-10-13): the detector takes 2.6 s instead of 5.6 s on a 1,200 x 900 picture, a 12-line page 5.6 s instead of
10.0 s, one picture 1.2 s instead of 2.3 s, with the same validation error (2.14 %; the outputs differ from
PyTorch by about 1e-7). An INT8 version is faster again (page 4.3 s) but raised validation error to 2.79 %
(+0.65 points, over the 0.5-point rule), so it is not the default. Not yet timed on the Raspberry Pi.

install(reader) replaces reader.detector; on any problem it leaves the PyTorch detector in place.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import torch

import config

logger = logging.getLogger("smart_glass.ocr")


class OrtDetector:
    """Stands in for the CRAFT module inside easyocr.detection.test_net: net(x) -> (y, feature)."""

    def __init__(self, path: str, threads: int = 0):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])

    def __call__(self, x):
        y = self.sess.run(None, {"x": x.detach().cpu().numpy().astype(np.float32)})[0]
        return torch.from_numpy(y), None

    def eval(self):
        return self


def install(reader) -> str:
    """Use the ONNX detector in this EasyOCR reader if config and files allow. Returns what is in use."""
    choice = getattr(config, "OCR_DETECTOR", "auto")
    if choice == "torch" or getattr(reader, "device", "cpu") != "cpu":
        return "pytorch"
    path = config.OCR_DETECTOR_ONNX_INT8 if choice == "onnx_int8" else config.OCR_DETECTOR_ONNX
    if choice == "onnx_int8" and not os.path.isfile(path):
        path = config.OCR_DETECTOR_ONNX                 # no 8-bit file: the full-precision ONNX detector is next best
    if not os.path.isfile(path):
        if choice != "auto":
            logger.warning("OCR_DETECTOR=%s but %s is missing; using PyTorch", choice, path)
        return "pytorch"
    try:
        det = OrtDetector(path)
        det(torch.zeros(1, 3, 64, 64))                 # one small pass: a broken file fails here, not on a capture
        reader.detector = det
        logger.info("Text detector: ONNX Runtime (%s)", os.path.basename(path))
        return os.path.basename(path)
    except Exception as exc:
        logger.warning("ONNX text detector not used (%s); using PyTorch", exc)
        return "pytorch"
