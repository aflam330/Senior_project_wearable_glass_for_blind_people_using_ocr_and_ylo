"""Export YOLOv8s to TorchScript, ONNX, and INT8 ONNX for Raspberry Pi 5."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "best.pt"
DATA = ROOT / "data" / "data.yaml"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--skip-int8", action="store_true")
    args = p.parse_args()
    if not MODEL.is_file():
        raise SystemExit(f"missing {MODEL}")
    model = YOLO(str(MODEL))

    ts = model.export(format="torchscript", imgsz=args.imgsz)
    print(f"TorchScript: {ts}")

    onnx = model.export(format="onnx", imgsz=args.imgsz, simplify=True, opset=12)
    print(f"ONNX: {onnx}")

    if not args.skip_int8:
        try:
            int8 = model.export(
                format="onnx",
                imgsz=args.imgsz,
                int8=True,
                data=str(DATA),
                simplify=True,
            )
            print(f"INT8 ONNX: {int8}")
        except Exception as exc:
            print(f"INT8 ONNX export skipped ({exc})")

    # TFLite INT8 is typically ~6 MB for YOLOv8s (the 5.8 MB paper figure).
    try:
        tflite = model.export(
            format="tflite",
            imgsz=args.imgsz,
            int8=True,
            data=str(DATA),
        )
        print(f"INT8 TFLite: {tflite}")
    except Exception as exc:
        print(f"TFLite INT8 export skipped ({exc})")
        try:
            tflite = model.export(format="tflite", imgsz=args.imgsz)
            print(f"TFLite FP32: {tflite}")
        except Exception as exc2:
            print(f"TFLite export skipped ({exc2})")


if __name__ == "__main__":
    main()
