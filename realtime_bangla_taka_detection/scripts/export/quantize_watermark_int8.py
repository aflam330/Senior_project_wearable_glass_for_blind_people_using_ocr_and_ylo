"""Static INT8 (QDQ, per-channel) for the watermark MobileNet, calibrated on TRAIN crops.

Dynamic INT8 changed 55 % of test decisions, and full static QDQ 47 %; only Conv weights are quantised here (results/watermark/mobilenet.json). Static quantisation is
calibrated on 200 TRAIN watermark crops of the serial-disjoint split; the first convolution and the
classifier stay in FP32 (as for the Taka detector, scripts/quantize_int8.py). Agreement with FP32 is
checked on the TEST crops. Output: models/watermark_mobilenet_int8.onnx (replaced) and an entry in
results/watermark/mobilenet.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
from roboeye.camva.notes import load_splits  # noqa: E402
from train_watermark_mobilenet import EV, WM  # noqa: E402

M = ROOT / "models"


def crops(split, limit=None):
    splits, _ = load_splits(ROOT / "results" / "serial_split")
    ok = {r["note_id"] for r in json.loads((WM / "features.json").read_text(encoding="utf-8")) if r["ok"]}
    ids = [n for n in splits[split] if n in ok][:limit]
    return [EV(Image.open(WM / "crops" / (n.replace(":", "_") + ".png")).convert("RGB")).unsqueeze(0).numpy() for n in ids]


class Reader(CalibrationDataReader):
    def __init__(self, xs):
        self.it = iter([{"image": x} for x in xs])

    def get_next(self):
        return next(self.it, None)


def main() -> None:
    fp32 = M / "watermark_mobilenet.onnx"
    graph = onnx.load(str(fp32)).graph
    convs = [n.name for n in graph.node if n.op_type == "Conv"]
    gemms = [n.name for n in graph.node if n.op_type in ("Gemm", "MatMul")]
    exclude = convs[:1] + gemms[-1:]
    out = M / "watermark_mobilenet_int8.onnx"
    quantize_static(str(fp32), str(out), Reader(crops("train", 200)), quant_format=QuantFormat.QDQ,
                    per_channel=True, reduce_range=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    nodes_to_exclude=exclude, op_types_to_quantize=["Conv"])  # hard-swish / SE stay FP32
    xs = crops("test")
    run = lambda path: np.concatenate([ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"image": x})[0] for x in xs])
    a, b = run(fp32), run(out)
    pa = np.exp(a - a.max(1, keepdims=True)); pa = pa[:, 1] / pa.sum(1)
    pb = np.exp(b - b.max(1, keepdims=True)); pb = pb[:, 1] / pb.sum(1)
    res = json.loads((WM / "mobilenet.json").read_text(encoding="utf-8"))
    res["export"]["watermark_mobilenet_int8.onnx (static, calibrated)"] = {
        "same_decision_as_fp32": float(((pa >= 0.5) == (pb >= 0.5)).mean()), "max_abs_prob_diff": float(np.abs(pa - pb).max()),
        "size_mb": round(out.stat().st_size / 1e6, 2), "excluded_nodes": exclude}
    res["export"].pop("watermark_mobilenet_int8.onnx", None)
    (WM / "mobilenet.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(res["export"])


if __name__ == "__main__":
    main()
