# Raspberry Pi 5: hand-off of 29 October 2026

What was run on the glass's Pi on 29 October, what is still open, and what needs fixing. The measured numbers and graphs
are in [REPORT.md](REPORT.md) (built from `raw/` by `scripts/make_report.py`). This file adds what the data cannot say: the
team's answers and the problems found while setting up.

## Your list, item by item

| # | Item | Status | Result |
|---|---|---|---|
| a | `benchmark_pi5.py` | Done, 4 runs, none throttled | Report run: `raw/pi5_benchmark_memo_20261029T110832Z.json`. Also `savior_glass/results/pi5_benchmark_memo_20261029T10*.json` |
| b | `benchmark_pi5.py --sustained 30` | **Not run** (cancelled) | The only 30-minute data is still from 1 October (77.1 °C, 1.48 GB) |
| c | `pi_check_all.sh` | **Not run** (cancelled) | Its OCR response test ran with the default settings: `raw/ocr_response.json`. The microphone and sound-device lists are in `raw/hardware_20261029.txt`. The four detector and read-width settings, and Piper's speech start time, were not measured |
| d | `measure_mode_switch.py` | Done | `raw/mode_switch_currency.json`: first switch into currency mode 3.06 s, first check 664 ms, later checks 482 ms |
| e | `power_log.py` | **Not run** (cancelled) | When it runs, use `--bank-wh 74` (the bank is 20,000 mAh) |

Also done: `extras_bench.py` (emotion with a face, both object detectors, espeak-ng speech, the camera), a temperature,
clock, CPU and memory timeline across the whole run, and two sessions with the real app and camera (`raw/field/`).

## Hardware (from the Pi, `raw/hardware_20261029.txt`)

| Part | What the Pi reports |
|---|---|
| Board | Raspberry Pi 5 Model B, 8 GB; Debian 13.5 (trixie), kernel 6.18.33 |
| Camera | Logitech C270 USB webcam (`/dev/video0`), not a ribbon camera. 640x480 at 7.6 frames/s; 1.3 s from opening to the first frame |
| Microphone | The C270's built-in microphone, the only capture device. Voice commands use it |
| Sound out | Bluetooth. ALSA lists only the two HDMI outputs and there is no USB sound card. Paired: AWEI Y382 (connected during the app test), soundcore R60i NC, Freego1 Air |
| Buttons (BCM) | 17 MODE, 27 ACTION, 24 READ, 22 VOL+, 23 VOL− (`config.py`) |
| Vibration motor | BCM 13 (`HAPTIC_PIN`) |
| Cooling | A fan and a heat sink were added after 13 October (team). The fan ran at up to 10,100 rpm in the run |
| Power bank | 20,000 mAh, about 74 Wh at 3.7 V (team) |

## Your seven questions

1. **Headphone connection:** not answered in words. The Pi shows Bluetooth only: the AWEI Y382 was the sound output during the app test.
2. **Vibration motor driver:** open. The answer, a picture or link, did not come through.
3. **UV lamp power:** the team is noting it by hand.
4. **Thermometer reading:** the team is noting it by hand.
5. **Latency check:** no stopwatch test. The app's own field log measures from button press to spoken answer; see the table below.
6. **Cooling since 13 October:** a fan and a heat sink were added. That fits the cooler run (peak 79.3 °C, flag `0x0`, against 88.7 °C and throttling on 13 October).
7. **Real button test:** still open. In both app sessions every press came from the preview window's keyboard or from a voice command. No physical GPIO press and no vibration event were logged.

## The two app sessions (real camera, after the benchmark)

`raw/field/20261029_172213_memo` (12 min) and `raw/field/20261029_184909_memo` (6 min). The second ran from the
Senior_project clone on the USB drive.

| What | Times | Median, press to spoken answer |
|---|---:|---:|
| Currency check | 10 | 654 ms |
| Object scan and announcement | 8 | 811 ms |
| Text reading, one capture | 8 | 15.6 s (13.4 s to 86.3 s) |
| Guided watermark check | 6 | 10.3 s, every one a timeout |

- **Watermark check:** all 6 guided checks timed out. Of 111 frames, 65 were rejected as blurry and 46 as showing no note, so the glass never reached "likely genuine". The webcam gave 7.6 frames/s in the benchmark. Most rejected frames were blurry, which needs a closer look with this camera.
- **Currency:** the note was named only when the detector was at least 80% sure (`CURRENCY_ANNOUNCE_CONF` 0.8), which happened in 2 of 10 presses. The other 8 asked the user to hold the note closer, or found no note.
- **Text reading:** real pages read with clear mistakes, unlike the rendered test pictures, which had no character errors. Two English captures in the first session took about 86 s each, with the CPU busy throughout. The log does not show why; a page with many lines is one possible cause.
- **Objects:** the scans announced a person and a mobile phone. Voice commands worked: "glass scan" made a capture and "glass object" switched mode.
- **Load on the Pi:** the app used up to 2.97 GB of memory and reached up to 78.2 °C.

## Problems found, to fix in the repositories

1. **Senior_project is missing three model files:** `savior_glass/models/currency_mobilenet.pt`, `yolov8s.pt` and `yolov8n.pt`. Its `.gitignore` has `models/*.pt`, so a fresh clone has no currency classifier and no object detector files. They were copied from SARA by hand on the Pi. Push them with `git add -f`, or say in the guide where they come from.
2. **Out-of-date instructions:** `HOW_TO_RUN.md` in Senior_project still points to the `research-final-2026-09-30` branch and says the Pi uses `best_int8.onnx`. `deploy_pi5.sh` prints the same old line. Since `3f90251` the app loads `best_handheld_int8.onnx`.
3. **`benchmark_pi5.py` times only the earlier Taka detector on its own** (`best_int8.onnx`, `best.onnx`, `best.pt`). The detector the app uses is timed only inside the end-to-end currency row (441 ms). A `best_handheld_int8.onnx` row would close that gap.
4. **README 98.2%:** the README gives 98.2% for the counterfeit check with one view. The evaluation files of the checkpoint the Pi runs give 97.1% (`realtime_bangla_taka_detection/results/qduig/prefix_ft/seed42/test_views_20260928`).
5. **Emotion model:** the README names YuNet + EfficientNet-V2-S, but the Pi answers with the `rgb_mobilenet_blend` classifier. The RAF-DB confusion matrix in `savior_glass/results/emotion_rafdb.json` is for EfficientNet-V2-S, so it does not describe the glass.
6. **The test datasets are not on the Pi** (RAF-DB, JaalTaka, COCO val2017). `benchmark_pi5.py` falls back to a blank frame for emotion, a repeated note photo for the multi-view check, and note photos for objects. `extras_bench.py` times emotion with a real face (675 ms).

## Running the glass from the USB drive

The SD card is nearly full, so the clones live on the USB drive (exFAT). The Python environment stays on the SD card
(`~/.venvs/savior_glass`), because exFAT cannot hold its links. The GPIO library's pipe files go to the SD card through `LG_WD`.

```bash
cd "/media/memo/7250-9755/Senior_project_wearable_glass_for_blind_people_using_ocr_and_ylo/savior_glass"
source ~/.venvs/savior_glass/bin/activate
export LG_WD=~/.lgpio
python main.py
```
