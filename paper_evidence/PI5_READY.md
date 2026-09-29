# Raspberry Pi 5: READY_FOR_DEVICE (2026-09-30)

No Pi 5 was connected, so no Pi number exists. Everything needed to measure it is on disk.

## Run

```bash
# on the Pi (Raspberry Pi OS 64-bit), from the repository root
bash savior_glass/scripts/deploy_pi5.sh              # installs dependencies and the systemd service
python savior_glass/scripts/benchmark_pi5.py         # latency, FPS, CPU, RAM, temperature, sustained run
```

Record the output in `PI5_RESULTS.md`. Its table lists median / P95 latency for 1–6 views, FPS, CPU, RAM, temperature and a 30-minute sustained run.

## What runs on the Pi

| Component | File | Size | Notes |
|---|---|---:|---|
| Taka detector, INT8 ONNX | `realtime_bangla_taka_detection/models/best_int8.onnx` | 18 MB | 200 / 200 correct on the check set (`Unused/README.md`) |
| PRMVT (safe jaal policy, 4 cut views) | `results/qduig/prefix_ft/seed42/checkpoint.pt` | 16.6 MB | `benchmark_pi5.py` times PRMVT at 1–6 views, which covers the 4-view check |
| Watermark window check (research) | ResNet-50 + logistic head (`scripts/eval/watermark_hybrid.py`) | ≈ 100 MB | Needs a back-lit photo ("hold the note up to the light"). Not yet part of the app. On the Pi, a MobileNet-sized backbone would be the practical choice; it has not been trained |
| Serial OCR | EasyOCR bn + en | large | Research only. Its false-match rate on real photos is in `SERIAL_WATERMARK_DETECTOR.md` |

## Pi-specific checks to add when running

1. Detector: time the INT8 ONNX path against the PyTorch path.
2. Safe jaal policy: end-to-end time from button press to speech for a 500 / 1,000 Taka note.
3. Thermal: the 30-minute sustained run with the case closed.
4. Battery: minutes of continuous currency mode per charge.
