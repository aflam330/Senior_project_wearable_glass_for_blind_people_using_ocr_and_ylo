"""INT8 text detector: can a better calibration bring it inside the 0.5-point rule?

The first INT8 version (MinMax calibration on 64 images) was 4.3 times faster than PyTorch but raised validation
CER by 0.65 points (detector_onnx.py), so it is not the default. Variants tried here, all calibrated on the
"lexicon" and "select" phrase images only (never validation or test):

  minmax128      MinMax on 128 images
  percentile     Percentile calibration (99.999 %) on 128 images
  entropy        Entropy (KL) calibration on 128 images
  conv_only      MinMax on 128 images, only Conv layers quantised

Rule: the fastest variant whose validation CER is within 0.5 points of the full-precision detector's becomes the
default; if none is, full precision stays.

    python -m ocr_bench.detector_int8_variants          # accuracy on validation for each variant
    python -m ocr_bench.detector_int8_variants time     # timings of the variants that exist (run on a quiet CPU)
Output: results/ocr_bench/detector_int8_variants.json, models/craft_int8_<variant>.onnx
"""
from __future__ import annotations

import json
import sys
import time

import cv2
import numpy as np
import torch

from . import bench as B
from .detector_onnx import FP32, MODELS, THREADS, OrtDetector
from .page_bench import OUT, _ocr_mode, make_page, reader_fn

RES = OUT / "detector_int8_variants.json"
VARIANTS = ("minmax128", "percentile", "entropy", "conv_only")


def calib_inputs(n=128):
    from easyocr.imgproc import normalizeMeanVariance, resize_aspect_ratio
    items = [it for s in (0, 1, 2) for name in ("lexicon", "select") for it in B.build(name, s)][:n]
    for it in items:
        img = B.decode(it)
        h, w = img.shape[:2]
        img = cv2.resize(img, (1200, int(h * 1200 / w)), interpolation=cv2.INTER_CUBIC)
        g = cv2.cvtColor(cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2RGB)
        r, _, _ = resize_aspect_ratio(g, 2560, interpolation=cv2.INTER_LINEAR, mag_ratio=1.0)
        yield {"x": np.transpose(normalizeMeanVariance(r), (2, 0, 1))[None].astype(np.float32)}


def build(name):
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
    path = MODELS / f"craft_int8_{name}.onnx"
    if path.exists():
        return path

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = calib_inputs()

        def get_next(self):
            return next(self.it, None)

    kw = dict(quant_format=QuantFormat.QDQ, per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    if name == "percentile":
        kw.update(calibrate_method=CalibrationMethod.Percentile, extra_options={"CalibPercentile": 99.999})
    elif name == "entropy":
        kw.update(calibrate_method=CalibrationMethod.Entropy)
    elif name == "conv_only":
        kw.update(op_types_to_quantize=["Conv"])
    quantize_static(str(FP32), str(path), Reader(), **kw)
    return path


def accuracy():
    import config
    torch.set_num_threads(THREADS)
    m = _ocr_mode()
    read = reader_fn(m, 1200, 1, config.OCR_V2_DECODER)
    res = json.loads(RES.read_text(encoding="utf-8")) if RES.exists() else {"date": "2026-10-13", "min_conf": config.OCR_V2_MIN_CONF, "variants": {}}
    todo = [("full precision", FP32)] + [(v, None) for v in VARIANTS]
    for name, path in todo:
        if name in res["variants"]:
            continue
        try:
            path = path or build(name)
            m._reader.detector = OrtDetector(path)
            val = B.score(read, "val")
            res["variants"][name] = {"val_cer": round(val["overall"]["cer"], 4), "val_cer_bn": round(val["bn"]["cer"], 4),
                                     "val_cer_en": round(val["en"]["cer"], 4), "file_mb": round(path.stat().st_size / 1e6, 1)}
        except Exception as exc:                       # a calibration method that fails is recorded, not hidden
            res["variants"][name] = {"failed": f"{type(exc).__name__}: {str(exc)[:200]}"}
        RES.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(name, res["variants"][name], flush=True)
    base = res["variants"]["full precision"]["val_cer"]
    for v in res["variants"].values():
        if "val_cer" in v:
            v["within_0.5_points"] = bool(v["val_cer"] - base <= 0.005)
    RES.write_text(json.dumps(res, indent=1), encoding="utf-8")


def timing():
    import config
    torch.set_num_threads(THREADS)
    res = json.loads(RES.read_text(encoding="utf-8"))
    m = _ocr_mode()
    read = reader_fn(m, 1200, 1, config.OCR_V2_DECODER)
    x = torch.randn(1, 3, 896, 1216)
    pages = [make_page(s, w, h) for s in (0, 1, 2) for (w, h) in ((640, 480), (1280, 720))]
    for name, v in res["variants"].items():
        path = FP32 if name == "full precision" else MODELS / f"craft_int8_{name}.onnx"
        if "val_cer" not in v or not path.exists():
            continue
        net = OrtDetector(path)
        m._reader.detector = net
        net(x)
        t = []
        for _ in range(7):
            t0 = time.perf_counter(); net(x); t.append(time.perf_counter() - t0)
        read(pages[0][0])
        pt = []
        for img, _ in pages:
            t0 = time.perf_counter(); read(img); pt.append(time.perf_counter() - t0)
        v["detector_s_median"], v["page_s_median"] = round(float(np.median(t)), 3), round(float(np.median(pt)), 2)
        print(name, v, flush=True)
    RES.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    timing() if len(sys.argv) > 1 and sys.argv[1] == "time" else accuracy()
