"""Latency on the laptop GPU (CUDA) vs CPU for the detector, PRMVT and the emotion network.

Inputs are real test images held in memory, so disk reads are not timed. GPU timings call
torch.cuda.synchronize() before the clock stops. Median, p95 and FPS over --iters runs
after --warmup runs. ONNX on the GPU needs the CUDA execution provider (onnxruntime-gpu);
without it that row is NOT_MEASURED. Writes results/speed/gpu_speed.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WS = ROOT.parent


def timeit(fn, warmup, iters, cuda):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    ms = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ms.append((time.perf_counter() - t) * 1000)
    ms.sort()
    med = statistics.median(ms)
    return {"median_ms": round(med, 2), "p95_ms": round(ms[int(0.95 * (len(ms) - 1))], 2), "fps": round(1000 / med, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    has_cuda = torch.cuda.is_available()
    out = {"gpu": torch.cuda.get_device_name(0) if has_cuda else None, "torch": torch.__version__, "rows": {}}
    rows = out["rows"]
    imgs = [cv2.imread(str(p)) for p in sorted((WS / "data set/currency_yolo_data/test/images").glob("*"))[:20]]

    # YOLO PyTorch
    from ultralytics import YOLO
    for dev in (["cuda", "cpu"] if has_cuda else ["cpu"]):
        m = YOLO(str(ROOT / "models/best.pt"))
        i = iter(range(10 ** 9))
        rows[f"yolo_pt_{dev}"] = timeit(lambda: m.predict(imgs[next(i) % len(imgs)], device=0 if dev == "cuda" else "cpu",
                                                         imgsz=640, verbose=False), args.warmup, args.iters, dev == "cuda")

    # ONNX FP32 / INT8 through onnxruntime (network only, letterboxed input)
    import onnxruntime as ort
    from scripts.quantize_int8 import letterbox
    x = letterbox(imgs[0], 640)
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()  # onnxruntime-gpu[cuda,cudnn] ships its own CUDA 13 / cuDNN 9 DLLs
    providers = ort.get_available_providers()
    out["onnxruntime_providers"] = providers
    for name in ("best.onnx", "best_int8.onnx"):
        for ep in ("CUDAExecutionProvider", "CPUExecutionProvider"):
            key = f"yolo_{'int8' if 'int8' in name else 'fp32'}_onnx_{'cuda' if ep.startswith('CUDA') else 'cpu'}"
            if ep not in providers:
                rows[key] = "NOT_MEASURED (onnxruntime-gpu not installed)"
                continue
            s = ort.InferenceSession(str(ROOT / "models" / name), providers=[ep])
            used = s.get_providers()[0]
            inp = s.get_inputs()[0].name
            r = timeit(lambda: s.run(None, {inp: x}), args.warmup, args.iters, False)
            rows[key] = {**r, "provider_used": used}

    # PRMVT (Q-DUIG prefix_ft) at 1 and 6 views, network only
    from roboeye.qduig.config_io import load_config
    from roboeye.qduig.engine import load_qduig
    d = ROOT / "results/qduig/prefix_ft/seed42"
    net = load_qduig(d / "checkpoint.pt", load_config(d / "config.yaml"))
    for dev in (["cuda", "cpu"] if has_cuda else ["cpu"]):
        net = net.to(dev).eval()
        for k in (1, 6):
            v = torch.randn(1, k, 3, 128, 128, device=dev)
            mk = torch.ones(1, k, dtype=torch.long, device=dev)
            with torch.inference_mode():
                rows[f"prmvt_{k}view_{dev}"] = timeit(lambda: net(v, mk), args.warmup, args.iters, dev == "cuda")

    # Emotion network (EfficientNet-V2-S, 224x224, with flip TTA as in the app)
    from torchvision import models
    bundle = torch.load(WS / "savior_glass/models/emotion_faces.pt", map_location="cpu", weights_only=False)
    emo = models.efficientnet_v2_s(weights=None)
    emo.classifier[-1] = torch.nn.Linear(emo.classifier[-1].in_features, 7)
    emo.load_state_dict(bundle["state_dict"])
    size = int(bundle.get("img_size", 224))
    for dev in (["cuda", "cpu"] if has_cuda else ["cpu"]):
        emo = emo.to(dev).eval()
        xb = torch.randn(2, 3, size, size, device=dev)
        with torch.inference_mode():
            rows[f"emotion_effnetv2s_{dev}"] = timeit(lambda: emo(xb), args.warmup, args.iters, dev == "cuda")

    for base in ("yolo_pt", "prmvt_1view", "prmvt_6view", "emotion_effnetv2s", "yolo_fp32_onnx", "yolo_int8_onnx"):
        g, c = rows.get(f"{base}_cuda"), rows.get(f"{base}_cpu")
        if isinstance(g, dict) and isinstance(c, dict):
            rows[f"{base}_speedup_gpu_over_cpu"] = round(c["median_ms"] / g["median_ms"], 1)
    dst = ROOT / "results/speed/gpu_speed.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
