"""Text detector (CRAFT) through ONNX Runtime instead of PyTorch: speed and accuracy on the CPU.

On the Pi 5 one text capture took 18.7 s and the time is in the text detector. The Taka detector became about twice
as fast there as an INT8 ONNX graph. This script measures the same idea for CRAFT on the laptop CPU limited to four
threads (the Pi has four cores). It cannot give a Pi number.

Variants: PyTorch (as now) / ONNX Runtime, full precision / ONNX Runtime, INT8 (static quantisation).
INT8 is calibrated on images of the "lexicon" and "select" phrase sets, never on validation or test phrases.

For each variant: detector time alone on a 1,200 x 900 input, time for a 12-line page through the glass pipeline,
and the validation error (48 phrases x 3 seeds x 3 conditions = 432 images). Rule, as for every speed setting of
this project: a faster variant is acceptable only if validation CER stays within 0.5 points of the current one.

    python -m ocr_bench.detector_onnx
Output: results/ocr_bench/detector_onnx.json, models/craft_fp32.onnx, models/craft_int8.onnx
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from . import bench as B
from .page_bench import OUT, _ocr_mode, make_page, reader_fn

MODELS = Path(__file__).resolve().parents[1] / "models"
FP32, INT8 = MODELS / "craft_fp32.onnx", MODELS / "craft_int8.onnx"
THREADS = 4


class OrtDetector:
    """Stands in for the CRAFT module inside easyocr.detection.test_net: net(x) -> (y, feature)."""

    def __init__(self, path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = THREADS
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])

    def __call__(self, x):
        y = self.sess.run(None, {"x": x.detach().cpu().numpy().astype(np.float32)})[0]
        return torch.from_numpy(y), None

    def eval(self):
        return self


def export(net):
    if not FP32.exists():
        x = torch.randn(1, 3, 608, 800)
        torch.onnx.export(net, x, str(FP32), input_names=["x"], output_names=["y", "feature"], opset_version=17,
                          dynamic_axes={"x": {0: "n", 2: "h", 3: "w"}, "y": {0: "n", 1: "h2", 2: "w2"}}, dynamo=False)
    if not INT8.exists():
        from easyocr.imgproc import normalizeMeanVariance, resize_aspect_ratio
        from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

        class Reader(CalibrationDataReader):
            def __init__(self):
                items = [it for s in (0, 1) for name in ("lexicon", "select") for it in B.build(name, s)]
                self.it = iter(items[:64])

            def get_next(self):
                it = next(self.it, None)
                if it is None:
                    return None
                img = B.decode(it)
                h, w = img.shape[:2]
                img = cv2.resize(img, (1200, int(h * 1200 / w)), interpolation=cv2.INTER_CUBIC)
                g = cv2.cvtColor(cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2RGB)
                r, _, _ = resize_aspect_ratio(g, 2560, interpolation=cv2.INTER_LINEAR, mag_ratio=1.0)
                return {"x": np.transpose(normalizeMeanVariance(r), (2, 0, 1))[None].astype(np.float32)}

        quantize_static(str(FP32), str(INT8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)


def main():
    torch.set_num_threads(THREADS)
    import config
    m = _ocr_mode()
    torch_net = m._reader.detector
    export(torch_net)
    variants = {"pytorch (current)": torch_net, "onnx runtime, full precision": OrtDetector(FP32),
                "onnx runtime, INT8": OrtDetector(INT8)}
    x = torch.randn(1, 3, 896, 1216)          # what the detector gets for a 1,200 x 900 picture (multiples of 32)
    pages = [make_page(s, w, h) for s in (0, 1, 2) for (w, h) in ((640, 480), (1280, 720))]
    read = reader_fn(m, 1200, 1, config.OCR_V2_DECODER)
    res = {"date": "2026-10-13", "threads": THREADS, "cpu": "laptop CPU limited to 4 threads; NOT a Raspberry Pi",
           "file_mb": {"fp32": round(FP32.stat().st_size / 1e6, 1), "int8": round(INT8.stat().st_size / 1e6, 1)}, "variants": {}}
    for name, net in variants.items():
        m._reader.detector = net
        with torch.no_grad():
            net(x)
            t = []
            for _ in range(7):
                t0 = time.perf_counter()
                net(x)
                t.append(time.perf_counter() - t0)
        read(pages[0][0])
        pt, pc = [], []
        for img, gt in pages:
            t0 = time.perf_counter()
            hyp = read(img)
            pt.append(time.perf_counter() - t0)
            pc.append(B.cer(gt, hyp))
        val = B.score(read, "val")
        res["variants"][name] = {"detector_s_median": round(float(np.median(t)), 3), "page_s_median": round(float(np.median(pt)), 2),
                                 "page_cer": round(float(np.mean(pc)), 4), "val_cer": round(val["overall"]["cer"], 4),
                                 "val_cer_bn": round(val["bn"]["cer"], 4), "val_cer_en": round(val["en"]["cer"], 4),
                                 "val_ms_median": round(val["latency_ms_median"], 0)}
        print(name, res["variants"][name], flush=True)
    m._reader.detector = torch_net
    base = res["variants"]["pytorch (current)"]["val_cer"]
    for name, v in res["variants"].items():
        v["within_0.5_points_of_current"] = bool(v["val_cer"] - base <= 0.005)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "detector_onnx.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
