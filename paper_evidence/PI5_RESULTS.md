# Pi 5 results

Measured 2026-10-01 on a Raspberry Pi 5 Model B Rev 1.0 (4-core aarch64, 8 GB RAM, Debian 13 trixie), `raspberry_pi_5: true`. Full output: `savior_glass/results/pi5_benchmark_memo_20261001T111325Z.json` (+ `.csv` for the sustained run). An interactive version of this table, with charts, is in `savior_glass/results/` as the source for the numbers below; see also `savior_glass/results/pi5_fixes_verification.json` for the post-fix re-measurement described at the end of this file.

The measurement script is `savior_glass/scripts/benchmark_pi5.py --iters 100 --sustained 30`. On a machine that is not a Pi it labels `raspberry_pi_5=false`, so a laptop run cannot be copied here as a Pi result.

| module | median | P95 | FPS |
| --- | ---: | ---: | ---: |
| currency_mode_full (detect+auth, end to end) | 326.62 ms | 560.03 ms | 3.06 |
| taka_yolo_pt (.pt, comparison only — live app uses int8) | 507.30 ms | 602.55 ms | 1.97 |
| taka_yolo_onnx_fp32 (comparison only) | 485.47 ms | 575.48 ms | 2.06 |
| taka_yolo_onnx_int8 (what currency_mode_full actually runs on Pi 5) | 254.44 ms | 339.38 ms | 3.93 |
| watermark_int8_sift (comparison only — live app uses the localizer) | 372.47 ms | 853.90 ms | 2.68 |
| watermark_int8_localizer (what the live app actually runs) | 50.06 ms | 62.10 ms | 19.98 |
| jaal_check_qduig | 36.10 ms | 49.68 ms | 27.70 |
| prmvt_1view … prmvt_6view | 36.98 – 90.16 ms | 56.66 – 106.01 ms | 27.04 – 11.09 |
| object_mode (yolov8n.pt) | 207.18 ms | 216.57 ms | 4.83 |
| ocr_mode (EasyOCR, beamsearch) | 18,725.73 ms | 20,403.89 ms | 0.05 |
| emotion | 48.42 ms | 52.03 ms | 20.65 |

CPU / RAM / temperature:
- RSS: 1,192 MB after the first model loads, peaking at 1,480 MB with everything resident; flat (no leak) across the 30-minute sustained run.
- CPU utilization: ~300–390% of 400% (4 cores), sampled manually with `ps`/`top` — `benchmark_pi5.py` does not instrument this itself.
- Temperature: idle 57.85 °C, end of the per-module pass 74.35 °C, sustained-run peak 77.1 °C. `vcgencmd get_throttled` stayed `0x0` for the whole run — no throttling.

30-minute sustained run: 1,229 calls across all 15 modules, round-robin, peak 77.1 °C, `throttled=0x0` throughout. Under mixed load, worst-case per-call latency ran up to ~4× the isolated median for some modules (e.g. currency_mode_full: 320 ms isolated median vs 1,206 ms worst case) — see the CSV for per-call detail.

`object_mode` required `models/yolov8n.pt` (not in this checkout originally; downloaded via `ultralytics` — see `pi5_fixes_verification.json`). Everything else ran from files already in the checkout.

**Five fixes tried against the first run, re-measured 2026-10-01** (`savior_glass/results/pi5_fixes_verification.json`):
1. OCR greedy decoder — regressed (20.6 s vs 18.7 s beamsearch); reverted. The ~19 s cost is EasyOCR's CPU text detector, not decoding.
2. Piper TTS — English: 4.89 s (espeak-ng) → 3.30 s (Piper), real. Bangla: blocked — the only prebuilt ARM64 Piper binary can't phonemize the only public Bangla voice; falls back to espeak-ng automatically (unchanged speed). Also fixed a latent bug where Piper failing silently produced no audio at all on this headless (no display) Pi, rather than falling back.
3. YOLOv8 COCO weights — downloaded; object_mode went from skipped to 207 ms median.
4. Watermark localizer over SIFT — already the live app's default; the SIFT number above is a forced comparison path, not real app behavior. No code change needed.
5. INT8 over FP32/.pt for currency detection — already the live app's default on Pi 5; confirmed via log. No code change needed.

Status: MEASURED.

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

Status: **MEASURED** (2026-10-01, superseding READY_FOR_DEVICE below).

**Status 2026-09-29: READY_FOR_DEVICE.** No Raspberry Pi 5 was connected during this pass. `savior_glass/scripts/benchmark_pi5.py` already times PRMVT at 1–6 views, which covers the 4-view safe jaal check now in the app (`JAAL_VERDICT_FIXED.md`). Run it on the Pi with `savior_glass/scripts/deploy_pi5.sh` and record the numbers above.

**Status 2026-10-01: MEASURED.** Run on a Raspberry Pi 5 Model B Rev 1.0, see the table and fixes summary above. `C_PI5` in `CLAIM_REGISTRY.json` is left `NOT_MEASURED`: it is scoped to `scripts/benchmark_edge_qduig.py` specifically, a different (qduig-edge-only) script this pass did not run.
