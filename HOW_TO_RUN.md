# How to run this project

Clone this branch. The Pi setup and the measured models are here, not on the default branch.

```bash
git clone -b research-final-2026-09-30 https://github.com/aflam330/Senior_project_wearable_glass_for_blind_people_using_ocr_and_ylo.git
cd Senior_project_wearable_glass_for_blind_people_using_ocr_and_ylo
```

GitHub page: https://github.com/aflam330/Senior_project_wearable_glass_for_blind_people_using_ocr_and_ylo/tree/research-final-2026-09-30

## Where each part lives

| Path | What it is | Do you run it on the Pi? |
|---|---|---|
| `savior_glass/` | The glass app: camera, buttons, speech, OCR, objects, currency | Yes. This is the program. |
| `savior_glass/main.py` | Pi entry point | Yes |
| `savior_glass/test_windows.py` | Same modes on a laptop, keyboard instead of buttons | No. Laptop only. |
| `savior_glass/config.py` | Pins, model paths, watermark on/off | Edit only if a pin or path changes |
| `savior_glass/modes/` | OCR, object, currency, watermark, online Claude | Used by `main.py` |
| `realtime_bangla_taka_detection/` | Research code and the model files the glass loads | The app reads models from here. You do not retrain on the Pi. |
| `realtime_bangla_taka_detection/models/best_int8.onnx` | Note detector used on the Pi (18 MB) | Loaded automatically on a Pi 5 |
| `realtime_bangla_taka_detection/models/best.pt` | Same detector, PyTorch file used on a laptop | Not the Pi default |
| `realtime_bangla_taka_detection/models/watermark_mobilenetv2_int8.onnx` | Back-lit watermark model (2.6 MB) | On when the app sees a Pi 5 |
| `realtime_bangla_taka_detection/results/qduig/prefix_ft/seed42/checkpoint.pt` | Safe genuine check | Yes, for 500 and 1000 Taka |
| `data set/.../500 Taka_0001.jpg` and `1000  Taka_001.jpg` | Front templates used to find the watermark window | Yes. These two photos are in the repo. The rest of `data set/` is not. |
| `paper_evidence/` | Written results. Not a program. | No |
| `docs/` | Paper notes | No |
| `Unused/` | Old copies. Do not run these. | No |

On a Pi 5 the currency mode uses the INT8 detector and turns the watermark check on. On a laptop it uses `best.pt` and leaves the watermark check off. The app never says "counterfeit". For 500 and 1000 Taka it says the note is likely genuine, or it asks for a hand check.

## Test on a Windows laptop

From `savior_glass/`:

```powershell
pip install -r requirements.txt
python test_windows.py
```

Or double-click `run_windows_test.bat`.

Click the camera window first. Then:

| Key | Action |
|---|---|
| M | Cycle mode: text, object, currency, online Claude |
| A | Capture. Text mode stores the OCR text. Object and currency speak immediately. |
| R | Speak the last OCR text |
| C | Send the frame to Claude (needs internet and `ANTHROPIC_API_KEY`) |
| + / - | Volume |
| Q | Quit |

This laptop run is a function test. It is not a Raspberry Pi speed measurement.

Check that the model files are present, still from `savior_glass/`:

```powershell
python scripts/pi5_preflight.py
```

A pass here means the files open. It does not print a Pi latency.

## Set up the Raspberry Pi 5

Use Raspberry Pi OS 64-bit. Copy the clone onto the Pi, or clone there:

```bash
git clone -b research-final-2026-09-30 https://github.com/aflam330/Senior_project_wearable_glass_for_blind_people_using_ocr_and_ylo.git
cd Senior_project_wearable_glass_for_blind_people_using_ocr_and_ylo/savior_glass
bash scripts/deploy_pi5.sh
```

`deploy_pi5.sh` creates `.venv` inside `savior_glass/`, installs `requirements.txt` (PyTorch, OpenCV, ONNX Runtime, and the rest), and runs the preflight. The first install takes a while.

If the preflight prints `ok` for the watermark model, the INT8 detector, the safe-check checkpoint, and both templates, start the glass:

```bash
source .venv/bin/activate
python main.py
```

Buttons, BCM numbering, from `config.py`:

| BCM pin | Button |
|---|---|
| 17 | Next mode |
| 27 | Capture / announce |
| 24 | Read the last OCR text |
| 22 | Volume up |
| 23 | Volume down |

Speech uses `espeak-ng` for Bangla. The fuller `install.sh` also installs system packages, Piper, and a boot service. Use that only if you want the app to start on boot. `deploy_pi5.sh` is enough to run and test from this clone.

For a watermark reading, hold a 500 or 1000 Taka note up to a light so the portrait window shows, then press capture.

## Measure speed on the Pi

Still inside `savior_glass/` with the venv active:

```bash
python scripts/benchmark_pi5.py --iters 100 --sustained 30
```

This times currency, the INT8 detector, the watermark check, object detection, OCR, and the safe check. It also runs for 30 minutes and records temperature. The JSON file is:

`savior_glass/results/pi5_benchmark_<hostname>_<time>.json`

Copy the median, 95th percentile, CPU, RAM, and temperature into `paper_evidence/PI5_RESULTS.md` after the run. Do not type a laptop time into that file. Battery minutes are a separate measurement; do not guess them from the benchmark.

## Order to follow

1. Clone branch `research-final-2026-09-30`.
2. On the laptop, run `python test_windows.py` and try text, object, and currency.
3. On the laptop, run `python scripts/pi5_preflight.py` and confirm every line says `ok`.
4. On the Pi, run `bash scripts/deploy_pi5.sh`.
5. On the Pi, run `python main.py` and press the buttons.
6. On the Pi, run the benchmark and keep the JSON.

Pi latency is not in the repository yet. It exists only after step 6.
