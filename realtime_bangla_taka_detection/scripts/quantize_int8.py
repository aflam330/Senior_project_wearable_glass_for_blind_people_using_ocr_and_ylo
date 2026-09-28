"""Static INT8 quantization of the YOLOv8 Taka detector (ONNX Runtime, QDQ format).

The backbone and neck are quantized with per-channel weights and activation
ranges calibrated on real validation images. The Detect head (``/model.22/``: box/class
convs, DFL, sigmoid, concat) always stays FP32: quantizing it collapses the class scores
to ~0.007 and the model detects nothing, which is what the Ultralytics
``format="onnx", int8=True`` export produced.

Which other layers stay FP32 is chosen on validation images that are disjoint from the
calibration images: the smallest candidate set whose top-1 accuracy is within
``--tolerance`` of the FP32 model wins. The test split is never used to choose. The
choice is written next to the output as ``<output>.selection.json``.

Usage:
  python scripts/quantize_int8.py
  python scripts/quantize_int8.py --model models/best.onnx --output models/best_int8.onnx --n-calib 200
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnx.version_converter
import yaml
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

ROOT = Path(__file__).resolve().parents[1]
HEAD_PREFIX = "/model.22/"


def letterbox(img: np.ndarray, size: int) -> np.ndarray:
    """Ultralytics-style letterbox: keep aspect, pad with 114, RGB, NCHW float in [0, 1]."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0


class ImageReader(CalibrationDataReader):
    def __init__(self, paths: list[Path], input_name: str, size: int):
        self.items = iter(paths)
        self.input_name = input_name
        self.size = size

    def get_next(self):
        for p in self.items:
            img = cv2.imread(str(p))
            if img is not None:
                return {self.input_name: letterbox(img, self.size)}
        return None


def default_calib_dir() -> Path:
    cfg = yaml.safe_load((ROOT / "data" / "data.yaml").read_text(encoding="utf-8"))
    return Path(cfg["path"]) / cfg["val"]


def top1_accuracy(model_path: str, images: list[Path]) -> float:
    """Share of images whose highest-confidence box has a class present in the YOLO label file."""
    from ultralytics import YOLO

    model = YOLO(model_path, task="detect")
    hit = n = 0
    for img in images:
        label = img.parent.parent / "labels" / (img.stem + ".txt")
        classes = {int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()} if label.is_file() else set()
        if not classes:
            continue
        r = model.predict(str(img), conf=0.25, imgsz=640, verbose=False)[0]
        n += 1
        if r.boxes is not None and len(r.boxes):
            hit += int(int(r.boxes.cls[r.boxes.conf.argmax()]) in classes)
    return hit / max(n, 1)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ROOT / "models" / "best.onnx"))
    p.add_argument("--output", default=str(ROOT / "models" / "best_int8.onnx"))
    p.add_argument("--calib-dir", default=None, help="validation images (default: data.yaml val split)")
    p.add_argument("--n-calib", type=int, default=200, help="images used to calibrate activation ranges")
    p.add_argument("--n-select", type=int, default=300,
                   help="other validation images used to choose which layers stay FP32 (disjoint from calibration)")
    p.add_argument("--tolerance", type=float, default=0.005,
                   help="accept the smallest FP32 set whose top-1 is within this of the FP32 model")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="minmax", choices=["minmax", "percentile", "entropy"],
                   help="activation-range calibration method (percentile/entropy need several GB of RAM)")
    p.add_argument("--keep-fp32", nargs="*", default=None,
                   help="layers to leave unquantized besides the head, e.g. model.0; skips the selection")
    args = p.parse_args(argv)

    calib_dir = Path(args.calib_dir) if args.calib_dir else default_calib_dir()
    paths = sorted(q for q in calib_dir.iterdir() if q.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise SystemExit(f"no calibration images in {calib_dir}")
    random.Random(args.seed).shuffle(paths)
    calib, select = paths[: args.n_calib], paths[args.n_calib: args.n_calib + args.n_select]

    model = onnx.load(args.model)
    inp = model.graph.input[0]
    size = int(inp.type.tensor_type.shape.dim[2].dim_value or 640)
    # Candidates, smallest FP32 part first. The test split is never used to choose.
    candidates = [args.keep_fp32] if args.keep_fp32 is not None else [
        [], ["model.0"], ["model.0", "model.15", "model.18", "model.21"]]
    print(f"model {args.model}: {len(model.graph.node)} nodes, imgsz {size}; calibration {len(calib)} images, "
          f"selection {len(select)} other images from {calib_dir}")

    report = {"calibration_images": len(calib), "selection_images": len(select), "tolerance": args.tolerance,
              "selected_on": "validation (disjoint from calibration)", "candidates": []}
    with tempfile.TemporaryDirectory() as tmp:
        src = args.model
        opset = next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
        if opset < 13:  # per-channel DequantizeLinear (axis attribute) needs opset >= 13
            src = str(Path(tmp) / "opset13.onnx")
            onnx.save(onnx.version_converter.convert_version(model, 13), src)
            print(f"converted opset {opset} -> 13 for per-channel quantization")
        pre = Path(tmp) / "pre.onnx"
        quant_pre_process(src, str(pre), skip_symbolic_shape=True)
        fp32_acc = top1_accuracy(args.model, select) if len(candidates) > 1 else None
        report["fp32_top1"] = fp32_acc
        chosen = None
        for i, extra in enumerate(candidates):
            keep = (HEAD_PREFIX, *(f"/{k.strip('/')}/" for k in extra))
            excluded = [n.name for n in model.graph.node if n.name.startswith(keep)]
            out = str(Path(tmp) / f"cand{i}.onnx")
            quantize_static(
                str(pre), out, ImageReader(calib, inp.name, size),
                quant_format=QuantFormat.QDQ, per_channel=True,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                calibrate_method={"minmax": CalibrationMethod.MinMax,
                                  "percentile": CalibrationMethod.Percentile,
                                  "entropy": CalibrationMethod.Entropy}[args.method],
                nodes_to_exclude=excluded,
            )
            if fp32_acc is None:
                chosen = (out, extra, None)
                break
            acc = top1_accuracy(out, select)
            report["candidates"].append({"fp32_layers": ["model.22 (head)", *extra], "top1": acc})
            print(f"  FP32 layers {['model.22', *extra]}: val top-1 {acc:.4f} (FP32 model {fp32_acc:.4f})", flush=True)
            if acc >= fp32_acc - args.tolerance:
                chosen = (out, extra, acc)
                break
        if chosen is None:  # none within tolerance: take the most accurate
            best = max(range(len(report["candidates"])), key=lambda j: report["candidates"][j]["top1"])
            chosen = (str(Path(tmp) / f"cand{best}.onnx"), candidates[best], report["candidates"][best]["top1"])
        Path(args.output).write_bytes(Path(chosen[0]).read_bytes())
    report["chosen_fp32_layers"] = ["model.22 (head)", *chosen[1]]
    Path(args.output).with_suffix(".selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"INT8 ONNX: {args.output}  (FP32 layers: {report['chosen_fp32_layers']})")


if __name__ == "__main__":
    main()
