# Pi 5 results

Pi 5 hardware was not attached. No latency, FPS, temperature, or memory number in this file was measured on a Pi.

The measurement script is `savior_glass/scripts/benchmark_pi5.py`. On a machine that is not a Pi it labels `raspberry_pi_5=false`, so a laptop run cannot be copied here as a Pi result. On the Pi it warms up, times currency detection, INT8 and FP32 ONNX, and PRMVT at 1 through 6 views, and can log a sustained run.

| quantity | value |
| --- | --- |
| median latency, 1–6 views | NOT_MEASURED |
| P95 latency | NOT_MEASURED |
| FPS | NOT_MEASURED |
| CPU, RAM, temperature | NOT_MEASURED |
| 30-minute sustained run | NOT_MEASURED |

Status: PROTOCOL_READY.

---

## How to measure (restored 2026-09-29)

On the Pi 5, inside the glass venv (after `savior_glass/install.sh`):

```bash
python3 scripts/benchmark_pi5.py                  # every mode, ~5 min
python3 scripts/benchmark_pi5.py --sustained 30   # + 30-minute mixed run: latency, temperature, throttling, memory per call (CSV)
```

Output: `savior_glass/results/pi5_benchmark_<host>_<time>.json`, with `raspberry_pi_5: true` only when `/proc/device-tree/model` reports a Pi 5. Copy `realtime_bangla_taka_detection/models/best_int8.onnx` (the fastest CPU format, see `GPU_SPEED.md`), `best.pt`, `best.onnx` and `results/qduig/prefix_ft/seed42/`. Laptop reference numbers (not Pi): Ryzen 7 5800H, Taka YOLO 109 ms PyTorch / 139 ms FP32 ONNX / 101 ms INT8 ONNX end to end.

---

## READY_FOR_DEVICE checklist (2026-09-29)

No Raspberry Pi 5 is attached. Every number above stays NOT_MEASURED until this checklist has been run on the device. A run on any other host is labelled `raspberry_pi_5: false` by the script and must not be copied here.

**Setup**
1. Pi 5 (record the RAM size), official 27 W USB-C supply, active cooler fitted (record which), Raspberry Pi OS 64-bit (record the version).
2. `savior_glass/install.sh` (or the container in `savior_glass/docker/`).
3. Copy the models from `realtime_bangla_taka_detection/models/`: `best.pt`, `best.onnx` and `best_int8.onnx`. The glass no longer runs PRMVT by default (`JAAL_VERDICT_ENABLED = False`), but the benchmark times it; for that, also copy `results/qduig/prefix_ft/seed42/checkpoint.pt`.

**Latency and throughput**, per module, with warm-up excluded:
`python3 scripts/benchmark_pi5.py --iters 100`. It records the median, P95, FPS and model load time for:
- currency mode
- the Taka YOLO as PT, ONNX and INT8 ONNX
- PRMVT at 1–6 views
- object mode
- OCR mode
- emotion

**Sustained run**, 30 minutes:
`python3 scripts/benchmark_pi5.py --iters 30 --sustained 30`. It records CPU temperature, `vcgencmd get_throttled` and memory per call to a CSV. A throttled value other than `0x0` must be reported.

**Battery life.** The script cannot measure this, because the Pi 5 has no battery gauge.
1. Put a USB-C power meter (record its model) between the battery pack and the Pi.
2. Charge the pack to 100 % and start `main.py` in object mode, which scans every 1.5 s. Press ACTION in currency mode once a minute, using a helper script or by hand.
3. Log the time and the meter's cumulative Wh every 5 minutes until the Pi shuts down.
4. Report runtime, mean W and Wh used, together with the pack's rated Wh.

**Acceptance.** These are not targets. Report whatever is measured.
- Every mode appears in the JSON with `raspberry_pi_5: true`.
- Median press-to-speech time is measured by `scripts/test_buttons_haptics.py` on real GPIO. It was 366 ms on the laptop with simulated GPIO.

Paste the resulting JSON path and the numbers into the table at the top of this file, and add claims to `CLAIM_REGISTRY.json`.

Status: **READY_FOR_DEVICE**.

**Status 2026-09-29: READY_FOR_DEVICE.** No Raspberry Pi 5 was connected during this pass. `savior_glass/scripts/benchmark_pi5.py` already times PRMVT at 1–6 views, which covers the 4-view safe jaal check now in the app (`JAAL_VERDICT_FIXED.md`). Run it on the Pi with `savior_glass/scripts/deploy_pi5.sh` and record the numbers above.
