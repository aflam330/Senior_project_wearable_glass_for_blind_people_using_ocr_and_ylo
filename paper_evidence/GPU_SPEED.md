# GPU speed (RTX 3050 Laptop GPU) vs CPU

Measured 2026-09-28 on the development laptop (NVIDIA GeForce RTX 3050 Laptop GPU 4 GB,
driver 617.14; AMD Ryzen 7 5800H CPU), PyTorch 2.14.0+cu126, ONNX Runtime 1.30 (GPU build).
Script: `realtime_bangla_taka_detection/scripts/eval/gpu_speed.py`; raw numbers:
`realtime_bangla_taka_detection/results/speed/gpu_speed.json`. 10 warm-up runs, median and
95th percentile over 50 runs; GPU timings synchronise the device before the clock stops.
Network time only (images already in memory), except the Ultralytics PyTorch row, which
includes Ultralytics pre- and post-processing.

| Model | GPU median (p95) | GPU FPS | CPU median (p95) | GPU speed-up |
|---|---:|---:|---:|---:|
| Taka YOLOv8s, PyTorch (Ultralytics predict) | 19.1 ms (19.7) | 52.2 | 105.2 ms (117.6) | 5.5x |
| Taka YOLOv8s, FP32 ONNX | 24.0 ms (24.5) | 41.7 | 92.5 ms (104.3) | 3.9x |
| Taka YOLOv8s, INT8 ONNX | 37.1 ms (37.7) | 27.0 | 60.8 ms (72.2) | 1.6x |
| PRMVT authenticator, 1 view | 15.5 ms (17.7) | 64.6 | 10.4 ms (11.9) | 0.7x |
| PRMVT authenticator, 6 views | 16.3 ms (23.4) | 61.3 | 21.3 ms (24.4) | 1.3x |
| Emotion EfficientNet-V2-S (face + flip) | 29.4 ms (41.3) | 34.0 | 83.0 ms (93.2) | 2.8x |

Findings:
- The detector is 5.5x faster on the GPU; all models run in real time on it.
- The INT8 model is the fastest on the CPU but slower than FP32 on the GPU: the CUDA
  execution provider does not accelerate this QDQ-quantised graph. INT8 is for CPU targets
  such as the Raspberry Pi.
- PRMVT is small; at one view the GPU call overhead makes it slower than the CPU.
- None of these numbers is a Raspberry Pi 5 measurement (see `PI5_RESULTS.md`).
