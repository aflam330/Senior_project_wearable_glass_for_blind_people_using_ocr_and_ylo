"""Builds benchmark_report/2026-11-23_ai_hat/REPORT.md and its graphs from the raw results in that folder and the
23 November CPU run (benchmark_report/2026-11-23/raw/ocr_response_onnx_int8_1200.json).

python make_ai_hat_report.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "2026-11-23_ai_hat"
RAW = OUT / "raw"
(OUT / "graphs").mkdir(exist_ok=True)

cpu = {(r["case"], r["path"]): r for r in json.load(open(ROOT / "2026-11-23/raw/ocr_response_onnx_int8_1200.json"))["results"]}
hai = {(r["case"], r["path"]): r for r in json.load(open(RAW / "ocr_response_hailo.json"))["results"]}
signs = json.load(open(RAW / "ocr_signs.json"))
glass = json.load(open(RAW / "ocr_glass_set.json"))
obj = json.load(open(RAW / "object_detect.json"))


def s(ms):
    return "–" if ms is None else f"{ms / 1000:.2f} s"


def err(r):
    return "–" if r["cer"] is None else f"{r['cer']:.1f}%"


def pct(a, b):
    return f"{100 * (b - a) / a:+.0f}%"


rows = [k for k in cpu if k[1] == "app" and k in hai]

# graph 1: whole capture, CPU against AI HAT
fig, ax = plt.subplots(figsize=(9, 4.2))
labels = [cpu[k]["label"].replace(" (640x480)", "") for k in rows]
y = range(len(rows))
ax.barh([i + 0.2 for i in y], [cpu[k]["total_ms_median"] / 1000 for k in rows], 0.4, label="CPU finds text (Pi default, 8-bit CRAFT)", color="#9aa5b1")
ax.barh([i - 0.2 for i in y], [hai[k]["total_ms_median"] / 1000 for k in rows], 0.4, label="AI HAT finds text (OCR_DETECTOR=hailo)", color="#2a7de1")
ax.set_yticks(list(y), labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("seconds from ACTION to the full text (median)")
ax.legend(fontsize=8, loc="lower right")
ax.set_title("Text mode on the Raspberry Pi 5: one capture", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "graphs/01_ocr_capture_time.png", dpi=130)

# graph 2: the step that moved, and the object detector
fig, ax = plt.subplots(figsize=(7, 2.8))
names = ["Find text (one capture)", "Object detector (one picture)"]
c = [cpu[rows[0]]["detect_ms_median"], obj["cpu_ms_median"]]
h = [hai[rows[0]]["detect_ms_median"], obj["hailo_ms_median"]]
ax.barh([0.2, 1.2], c, 0.4, label="CPU", color="#9aa5b1")
ax.barh([-0.2, 0.8], h, 0.4, label="AI HAT", color="#2a7de1")
for yy, v in zip([0.2, 1.2, -0.2, 0.8], c + h):
    ax.text(v + 30, yy, f"{v:,.0f} ms", va="center", fontsize=8)
ax.set_yticks([0, 1], names, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("milliseconds (median)")
ax.set_xlim(0, max(c) * 1.25)
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "graphs/02_chip_steps.png", dpi=130)

S, G = signs["summary"], glass["summary"]
md = [f"""# SARA on the Raspberry Pi AI HAT (Hailo-8): what moved to the chip and what it gained

Run on 23 and 24 November 2026 on the glass's own board (Raspberry Pi 5, 4 cores) with the Raspberry Pi AI HAT+
(Hailo-8, 26 TOPS, HailoRT 4.23.0). Code: branch `pi5-benchmark-2026-11-23` on `112c574` plus the AI HAT changes listed
at the end. Everything is **opt-in**: without `AI_HAT=1` the app runs exactly as before on the CPU.

## Summary

| Feature | What runs on the AI HAT | Before (CPU) | With the AI HAT | Accuracy |
|---|---|---:|---:|---|
| Objects | the whole YOLOv8s detector | {obj['cpu_ms_median']:.0f} ms a picture | **{obj['hailo_ms_median']:.0f} ms** | same objects on {obj['same']} of {obj['n']} pictures |
| Text (signs, labels) | finding the text | {S['cpu']['time_s_median']:.1f} s a sign | **{S['hailo']['time_s_median']:.1f} s** | character error {S['cpu']['cer_mean']:.1f} % → **{S['hailo']['cer_mean']:.1f} %** (60 signs) |
| Text (pages) | finding the text | {s(cpu[('page_12_lines_640','app')]['first_line_ms_median'])} to the first line spoken | **{s(hai[('page_12_lines_640','app')]['first_line_ms_median'])}** | 0.0 % on the rendered pages (0.1 % at 1280x720); on real book-page photos see below |
| Currency | nothing yet | – | – | needs a model compile (below) |
| Emotion | nothing yet | – | – | needs a model compile (below) |

## Objects

`OBJECT_BACKEND=hailo` (or `AI_HAT=1`): Hailo's own compile of the same COCO YOLOv8s that object mode uses
(`/usr/share/hailo-models/yolov8s_h8.hef`), box filtering on the chip. {obj['n']} pictures (2 Ultralytics test pictures,
the 23 November note pictures and live frames from the glass webcam):

| | CPU (`yolov8s.pt`) | AI HAT |
|---|---:|---:|
| Detector, median | {obj['cpu_ms_median']:.0f} ms | **{obj['hailo_ms_median']:.1f} ms** |
| Detector, 95th percentile | {obj['cpu_ms_p95']:.0f} ms | {obj['hailo_ms_p95']:.1f} ms |
| Same object list as the CPU | – | {obj['same']} of {obj['n']} (other runs 18 of 20: objects at the confidence threshold in fresh webcam frames) |

With `AI_HAT` off, object mode gives the same boxes and announcement as the committed code on {obj['default_same']} of {obj['n']} pictures.
In a live app session (23 November) object scans took a median 14 ms on the chip against 772 ms on the CPU.

## Text reading (OCR)

`OCR_DETECTOR=hailo` (or `AI_HAT=1`): the step that **finds** the text runs on the chip with Hailo's stock compile of
the PaddleOCR PP-OCRv5 mobile text detector (`models/hailo/paddle_ocr_v5_mobile_detection.hef`; it finds text in any
language). SARA's own Bangla + English letter recogniser still **reads** the text on the CPU, and everything after it is
unchanged. The detector's boxes go through EasyOCR's own grouping, so the recogniser gets boxes of the usual kind.

![Text mode, one capture](graphs/01_ocr_capture_time.png)

Same captures and script as the 23 November run (`ocr_response_bench.py`, app path with line-by-line speech on pages):

| Capture | CPU total | AI HAT total | change | finding text CPU → AI HAT | reading letters CPU → AI HAT | first line spoken CPU → AI HAT | character errors CPU / AI HAT |
|---|---:|---:|---:|---:|---:|---:|---:|"""]
for k in rows:
    a, b = cpu[k], hai[k]
    md.append(f"| {a['label']} | {s(a['total_ms_median'])} | **{s(b['total_ms_median'])}** | {pct(a['total_ms_median'], b['total_ms_median'])} | "
              f"{s(a['detect_ms_median'])} → {s(b['detect_ms_median'])} | {s(a['recognize_ms_median'])} → {s(b['recognize_ms_median'])} | "
              f"{s(a['first_line_ms_median'])} → {s(b['first_line_ms_median'])} | "
              f"{err(a)} / {err(b)} |")
md.append(f"""
Finding the text drops from 2.3–2.8 s to **0.37 s** on every capture. Reading the letters (CPU) is unchanged in kind but
not in amount: the chip detector draws fewer, longer boxes, which takes the recogniser a little longer on some captures
(the 12-line page at 1280x720 ended 0.4 s slower overall).

### Accuracy on signs and labels (60 pictures)

`hailo_ocr_sign_eval.py`: 60 pictures at 640x480 made with a fixed seed: 1–4 lines of common Bangla and English words,
nine Bangla fonts, dark-on-light and light-on-dark, tilted up to 4°, blur, uneven light and noise
(`raw/sign_eval_pictures/`). Same app code, only the detector switched:

| | CPU finds text | AI HAT finds text |
|---|---:|---:|
| Mean character error | {S['cpu']['cer_mean']:.2f} % | **{S['hailo']['cer_mean']:.2f} %** |
| Read without any error | {S['cpu']['exact']} / 60 | **{S['hailo']['exact']} / 60** |
| Median time per sign | {S['cpu']['time_s_median']:.2f} s | **{S['hailo']['time_s_median']:.2f} s** |
| Better than the other | {signs['cpu_better']} signs | {signs['hailo_better']} signs |

### Real glass-camera photos of book pages

`hailo_ocr_glass_eval.py` on `~/Desktop/glass_ocr/glass_set` ({glass['shots']} photos of {glass['pages']} printed pages, typed reference text),
at the camera width text mode gets ({glass['camera_width']} px):

| | CPU | AI HAT |
|---|---:|---:|
| Finding text, all {glass['shots']} photos, median (95th pct.) | {G['cpu']['find_ms_all_shots_median']:,.0f} ms ({G['cpu']['find_ms_all_shots_p95']:,.0f}) | **{G['hailo']['find_ms_all_shots_median']:,.0f} ms** ({G['hailo']['find_ms_all_shots_p95']:,.0f}) |
| Full read of a page, median ({glass['pages']} pages) | {G['cpu']['total_s_median']:.0f} s | **{G['hailo']['total_s_median']:.0f} s** |
| Mean character error ({glass['pages']} pages) | {G['cpu']['cer_mean']:.1f} % | {G['hailo']['cer_mean']:.1f} % |

At 640 px these dense book pages are beyond SARA's reader with either detector (most pages near 100 % error for both), so this
set does not separate the two on accuracy. On the one page both could read (page 1) the CPU detector did better
({glass['pages_read'][0]['cpu']['cer']:.1f} % against {glass['pages_read'][0]['hailo']['cer']:.1f} %).

## What is not on the chip yet, and why

A model runs on the Hailo-8 only after it is compiled into a `.hef` file by Hailo's Dataflow Compiler, which runs only
on an x86-64 Linux PC (not on the Pi) and is downloaded from the Hailo Developer Zone with a free account. Hailo has
compiled the standard models (that is how objects and text-finding run on the chip today); SARA's own trained models
have no compile yet:

| Feature | Model(s) | CPU time on the Pi (23 Nov) |
|---|---|---:|
| Currency | hand-held Taka detector (YOLOv8s, own classes) | 280 ms |
| Currency | counterfeit safe-check (EfficientNet) | 198 ms a crop |
| Emotion | expression classifiers (MobileNet + FER+) | ~500 ms with a face |
| Text | Bangla + English letter recogniser (CRNN with LSTM) | 0.6–12.5 s a capture |

Once compiled on any x86 PC (once per model), these `.hef` files are copied to the Pi and run on the AI HAT like the two above.

## How to use it

```
cd savior_glass && source ~/.venvs/savior_glass/bin/activate
AI_HAT=1 python main.py          # objects + text-finding on the AI HAT
python main.py                   # as before, everything on the CPU
```
Each feature can also be set alone (`OBJECT_BACKEND=hailo`, `OCR_DETECTOR=hailo`). If the chip or a `.hef` file is
missing, or the chip fails while running, that feature goes back to its CPU model by itself. The app's tests give the
same results with and without `AI_HAT=1` (11 pass, 3 skip for data that is not on the Pi, as on 23 November).

## Changes

- `savior_glass/hailo_device.py` (new): one shared handle on the chip with HailoRT's scheduler, so several models can use it.
- `savior_glass/hailo_yolo.py` (new): YOLOv8s on the chip for object mode.
- `savior_glass/hailo_text_detector.py` (new): PP-OCRv5 text detector on the chip, installed in front of EasyOCR's `detect`.
- `savior_glass/config.py`: `AI_HAT`, `OBJECT_BACKEND`, `HAILO_YOLO_HEF`, `HAILO_OCR_DET_HEF`, `OCR_HAILO_MODE`, `OCR_HAILO_CPU_FALLBACK`; defaults unchanged.
- `savior_glass/modes/object_mode.py`, `modes/ocr_mode.py`, `ocr_detector_onnx.py`, `live_diag.py`: use the chip when asked, CPU otherwise.
- `savior_glass/models/hailo/paddle_ocr_v5_mobile_detection.hef`: Hailo Model Zoo v2.19 stock file (sha256 `1daaca8700b49a48…`).
- `benchmark_report/scripts/hailo_object_eval.py`, `hailo_ocr_sign_eval.py`, `hailo_ocr_glass_eval.py`, `make_ai_hat_report.py` (new).
""")
(OUT / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
print("wrote", OUT / "REPORT.md")
