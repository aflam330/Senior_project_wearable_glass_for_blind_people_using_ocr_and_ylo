# Raspberry Pi 5, second run (13 October 2026) and what was changed because of it

Source: `benchmark_report/REPORT.md` with its raw files and scripts (run on the glass's own board, starting 23:32 local time; pushed from the Pi to the SARA repository, commit `3f5c6fe`). This note does not repeat the report; it records the findings that led to changes and what is still unmeasured.

## The run in one table

| What | Median | First run, 1 October |
|---|---:|---:|
| Currency check, end to end | 399 ms | 327 ms |
| Taka detector, INT8 ONNX | 306 ms | 254 ms |
| Watermark check, learned localizer | 64 ms | 50 ms |
| Object scan as the app runs it (YOLOv8s) | 562 ms | not run |
| Object detector YOLOv8n alone | 227 ms | 207 ms |
| Emotion, face in view | 680 ms | not run (48 ms was a frame without a face) |
| Text, 3-line sign, as the app ran it | 34.0 s | not run |
| Text, 3-line sign, text found once | 18.3 s | 18.7 s |
| Text, 12-line page: first line / whole page | 16.4 s / 36.5 s | not run |
| Text, no text in view | 27.5 s | not run |

**The board was hot and throttled**: 73 °C before the first call, 88.7 °C peak, throttled or clock-capped in 76 % of the samples of the text test, clock down to 1.5 GHz. No under-voltage. Every number is for that state.

## Changes made on 14 October because of this run

| Finding in the report | Change | Checked how | Measured on the Pi? |
|---|---|---|---|
| A short sign ran the text detector twice (34.0 s against 18.3 s) | The detector runs once per capture; the line-by-line path and the normal path use the same boxes | `scripts/test_ocr_single_pass.py`: identical text to EasyOCR's `readtext` on 16 benchmark images; one detector call for a sign, a blank picture and a page | **No** |
| A blank picture took 27.5 s and returned 48 made-up letters | No text boxes → "no text found" at once; the handwritten-letter reader is not tried on such a picture | Same script: blank picture answers "no text found", stored text empty | **No** |
| One detector pass takes 11–18 s | On the Pi the 8-bit detector is the default. Stated exception to the 0.5-point rule: +0.57 points of rendered validation CER (3.15 % against 2.58 %) | Laptop: page read correctly through the 8-bit path | **No**: its speed on the Pi is unknown |
| Object scan 562 ms with YOLOv8s, 227 ms with YOLOv8n | `YOLO_MODEL=yolov8n.pt` chooses the faster model; the default stays YOLOv8s (mAP@0.5 0.551 against 0.453) | — | — |

Expected on the Pi from these changes, to be confirmed and not to be quoted: a short sign near the "text found once" time instead of twice that; a blank picture in one detector pass instead of two.

## Not changed

- **Heat.** Software cannot cool the board. The run followed two sessions with the app, with VS Code and a preview window open. A repeat on a cool board, without VS Code, would separate heat from code.
- **Emotion at 680 ms.** That is the real cost with a face in view; it is fast enough for a button press.
- **Enlarging the 640 × 480 frame to 1,200 pixels before detection.** The report shows it makes a camera frame cost more than a 1280 × 720 one. Reading at the camera's own size was measured on 11 October: faster, but +0.74 points of validation error, and rejected.

## To run on the Pi next

```
cd ~/"SARA UPDATE"/SARA && git pull
cd savior_glass && . ~/.venvs/savior_glass/bin/activate
python scripts/test_ocr_single_pass.py
python ../benchmark_report/scripts/ocr_response_bench.py ../benchmark_report/raw/ocr_response_int8.json      # 8-bit detector (new default on the Pi)
OCR_DETECTOR=onnx python ../benchmark_report/scripts/ocr_response_bench.py ../benchmark_report/raw/ocr_response_fp32.json   # full precision, for comparison
```

Let the board cool first and close VS Code.
