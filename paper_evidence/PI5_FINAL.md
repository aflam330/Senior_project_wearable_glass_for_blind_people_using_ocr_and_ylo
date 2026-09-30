# Raspberry Pi 5 protocol (2026-10-01)

The device path is wired. Latency is still not measured: this machine is not a Pi 5.

On the Pi, from `savior_glass/`:

```bash
bash scripts/deploy_pi5.sh
python scripts/benchmark_pi5.py --iters 100 --sustained 30
```

`deploy_pi5.sh` installs `requirements.txt` (including `onnxruntime`) and runs `scripts/pi5_preflight.py`. The preflight checks files and runs one dummy input through each INT8 graph. It does not report a speed.

On a Pi 5 the app uses:

| Role | File | Why this one |
|---|---|---|
| Note detector | `realtime_bangla_taka_detection/models/best_int8.onnx` (18 MB) | Laptop check 200 / 200. Chosen automatically when `/proc/device-tree/model` contains Raspberry Pi 5 |
| Watermark | `models/watermark_mobilenetv2_int8.onnx` (2.6 MB) | Same decisions as FP32. The check turns on with the Pi and never says "counterfeit" |
| Safe genuine check | `results/qduig/prefix_ft/seed42/checkpoint.pt` (16.6 MB) | Policy E: 0 / 88 and 0 / 25 counterfeits passed |

The benchmark times `watermark_int8` as well as the detector and the other modes. Write median, 95th percentile, CPU, RAM, temperature, and the 30-minute run into `PI5_RESULTS.md`. Battery minutes stay a separate column. Do not copy a laptop time into that file.

A laptop can run `python scripts/pi5_preflight.py`. That only confirms the files and the graphs.
