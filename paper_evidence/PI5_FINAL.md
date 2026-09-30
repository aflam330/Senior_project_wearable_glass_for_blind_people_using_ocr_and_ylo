# Raspberry Pi 5 protocol (2026-09-30)

Unchanged from `PI5_READY_FINAL.md`. Still READY_FOR_DEVICE. No Pi log is in the repository.

```bash
bash savior_glass/scripts/deploy_pi5.sh
python savior_glass/scripts/benchmark_pi5.py
```

Models to time, all already checked on the laptop:

| Model | File | Laptop check |
|---|---|---|
| Watermark MobileNetV2 INT8 | `models/watermark_mobilenetv2_int8.onnx` (2.6 MB) | same decisions as FP32 |
| Taka detector INT8 | `models/best_int8.onnx` | 200 / 200 |
| Prefix network used by policy E | `results/qduig/prefix_ft/seed42/checkpoint.pt` | 0 / 88 counterfeits passed |

Also time `savior_glass/modes/watermark_check.py` with `WATERMARK_CHECK_ENABLED` on, for the benchmark only. Write median, 95th percentile, CPU, RAM, temperature, and a 30-minute run into `PI5_RESULTS.md`. Battery minutes are a separate column; do not infer them from a laptop.
