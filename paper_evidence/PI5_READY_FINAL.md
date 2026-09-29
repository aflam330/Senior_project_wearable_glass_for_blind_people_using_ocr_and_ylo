# Raspberry Pi 5: READY_FOR_DEVICE, final (2026-09-30)

Run on the Pi (Raspberry Pi OS 64-bit), from the repository root:

```bash
bash savior_glass/scripts/deploy_pi5.sh
python savior_glass/scripts/benchmark_pi5.py
```

Record the results in `paper_evidence/PI5_RESULTS.md`: median / P95 latency, FPS, CPU, RAM, temperature, 30-minute sustained run.

## Models for the Pi

| Model | File | Size | Check done on the laptop |
|---|---|---:|---|
| Taka detector, INT8 ONNX | `realtime_bangla_taka_detection/models/best_int8.onnx` | 18 MB | 200 / 200 correct |
| PRMVT (4-view safe policy) | `realtime_bangla_taka_detection/results/qduig/prefix_ft/seed42/checkpoint.pt` | 16.6 MB | policy E: 0 / 88, 0 / 25 counterfeits passed |
| **Watermark MobileNetV2, INT8 ONNX** | `realtime_bangla_taka_detection/models/watermark_mobilenetv2_int8.onnx` | **2.6 MB** | 100 % the same decisions as FP32; app module 92.9 % on unseen-print test photos |
| Watermark MobileNetV3 (FP32 fallback) | `realtime_bangla_taka_detection/models/watermark_mobilenet.onnx` | 6.1 MB | its INT8 versions failed (3 attempts), see `Unused/README.md` |

## Add to the Pi run

1. **Watermark path latency:** `modes/watermark_check.py`, SIFT registration + INT8 model. Enable `WATERMARK_CHECK_ENABLED` for the benchmark only.
2. **Button-to-speech time** for a 500 / 1,000 Taka note with policy E.
3. **Battery minutes** in continuous currency mode.
4. **Glass-camera photos** of genuine and known counterfeit notes, held normally and held against light, grouped by physical note. This is the data that tests the watermark on the device and can let policy E confirm more notes.
