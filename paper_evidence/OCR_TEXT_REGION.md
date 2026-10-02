# Text region detection before OCR

Measured 2026-10-02. The images are the same synthetic 640×160 renders as `savior_glass/scripts/eval_ocr_offline.py` (Nirmala, wild samples rotated up to 6 degrees). No currency photos.

## What was compared

EasyOCR’s detector on these runs is CRAFT (`detect_network` default). Every full-frame row below is CRAFT plus the usual recognizer.

An Otsu text crop was the region method that actually ran: invert the ink, take the bounding box, pad 8 pixels, then run the same OCR on the crop.

YOLOv8 is installed (`ultralytics`). No text-detector weights were in the repo. The download URL in the task prompt was a placeholder (`https://github.com/.../text_detector.pt`), so no YOLO text model was fetched and no YOLO number exists.

EAST was not run. There is no EAST model in the repo.

DBNet18 is EasyOCR’s other detector. The weights downloaded. A second attempt put Ninja on `PATH` (`Scripts\ninja.EXE`). The CUDA extension still did not compile: `cl.exe` was not found, and `CUDA_HOME` is not set. The forward pass raised `deform_conv_cuda.*.so is not imported successfully`. DBNet has no character-error number.

## Selection

Candidates were scored on a new validation list: 12 Bangla phrases and 12 English phrases that are not in the 24-phrase set and not in the 80-image phrase list. Three image seeds, 9000, 9100, and 9200. 144 images. GPU (RTX 3050 laptop). The rule was fixed first: lowest raw character error, then Bangla character error, then seconds.

The crop lost to the full frame on that list. It was not the kept pipeline.

| Method | Images | Raw CER | Bangla CER | English CER | Seconds |
|---|---:|---:|---:|---:|---:|
| Full frame, CRAFT, slope 0.1 | 144 | 0.1417 | 0.2168 | 0.0666 | 31.45 |
| Otsu crop, then the same OCR | 144 | 0.1624 | 0.2244 | 0.1004 | 34.28 |

The renders are already a tight text strip. Cropping them raised the error. The kept change is not a crop. It is `slope_ths=0.2` on the full frame, chosen on this same validation list (see `OCR_PARAM_TUNING.md`).

## Held-out, read once, after the choice

These two sets were not used to pick the crop or the slope. Device is the same GPU. Times are laptop times, not Pi times. The full-frame baseline on the 24-phrase set matches the earlier CPU file `ocr_select_canvas2560.json` (raw CER 0.132872), so the accuracy gap is not a GPU-versus-CPU artifact.

| Set | Pipeline | Raw CER | Bangla CER | English CER | WER | Seconds |
|---|---|---:|---:|---:|---:|---:|
| 24 phrases, 80 images | Full frame, slope 0.1 | 0.1329 | 0.2449 | 0.0208 | 0.3667 | 16.97 |
| 24 phrases, 80 images | Full frame, slope 0.2 | 0.1162 | 0.2324 | 0.0000 | 0.3542 | 16.88 |
| 80-image phrase list | Full frame, slope 0.1 | 0.0990 | 0.1746 | 0.0233 | 0.2333 | 18.43 |
| 80-image phrase list | Full frame, slope 0.2 | 0.0831 | 0.1428 | 0.0233 | 0.2188 | 18.22 |

Source: `savior_glass/results/ocr_improve_summary.json`.

The target (raw error under 10%, Bangla under 20%) is met on the 80-image list and is not met on the 24-phrase list. The 24-phrase list is the one that was at 13.3% and 24.5%. It is now 11.6% and 23.2%.

<!-- ocr-bench-2026-10-02 -->
## Benchmark pass with correctly shaped Bangla (2026-10-02, afternoon)

**Why the earlier numbers in this file are not comparable.** The earlier renders used Pillow's basic text layout. On this Windows machine Pillow has no Raqm, so Bangla was drawn wrongly: pre-base vowel signs after the consonant, conjuncts broken (for example "দেশ" drawn as "দশে"). Real print never looks like that. This pass shapes text with HarfBuzz (`savior_glass/ocr_bench/shaped_text.py`), the same shaping a printer or browser does.

**Benchmark** (`savior_glass/ocr_bench/bench.py`):
- **Phrase sets:**
  - val, 24 Bangla + 24 English new phrases, used for every choice;
  - test, 48 other new phrases, read once per finished method;
  - lexicon, the old 24 phrases;
  - select, the 24 phrases used once for the canvas choice.
- **Fonts:** val is Nirmala UI and Noto Serif Bengali / Georgia and Verdana. Test is Nirmala UI, Noto Sans Bengali and Tiro Bangla / Arial, Times and Calibri. Training fonts are never scored.
- **Images:** every phrase appears three ways: a clean page, a distorted page (rotation, blur, low resolution), and a sign placed into a real COCO photo at 640 × 480, the glass frame size.
- **Seeds:** three image seeds (0, 1, 2), so 432 images per set.
- **Metric:** CER and WER after Unicode NFC, on a laptop GPU. The final pipeline's CPU latency is reported separately.

### Method

1. EasyOCR reads the whole frame with CRAFT.
2. Boxes below a confidence are dropped.
3. **The dominant block is kept:** start from the most confident tall box, then add boxes of similar height that touch it. This is the sign or label the user faces; stray text in the background is dropped.
4. Boxes are joined line by line.

The "crop and read again" variant cuts the block out, upscales it and runs OCR a second time.

**Not run:**
- **DBNet18:** EasyOCR DBNet18 needs its deformable-conv C++ extension compiled; no C++ compiler on this laptop.
- **YOLOv8 and EAST text detectors:** no text-detection weights are available offline (the prompt's YOLO link was a placeholder).

### Validation (3 seeds, every choice made here)

| Method | CER | Bangla CER | English CER | WER | Clean | Distorted | Photo scene | CER per seed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Old offline path (EasyOCR, boxes in its own order) | 17.8 % | 18.3 % | 17.3 % | 22.8 % | 2.0 % | 6.5 % | 45.0 % | 10.6 / 19.6 / 23.2 |
| + boxes ordered into lines | 15.1 % | 16.3 % | 13.9 % | 19.7 % | 2.0 % | 1.3 % | 41.9 % | 8.4 / 16.3 / 20.6 |
| + confidence ≥ 0.1 | 6.3 % | 5.0 % | 7.6 % | 12.8 % | 2.0 % | 1.7 % | 15.3 % | 6.7 / 6.9 / 5.4 |
| + confidence ≥ 0.3 | 5.2 % | 4.6 % | 5.8 % | 10.1 % | 3.6 % | 1.7 % | 10.4 % | 5.7 / 5.1 / 4.9 |
| + confidence ≥ 0.5 | 13.4 % | 19.2 % | 7.6 % | 16.5 % | 11.2 % | 9.2 % | 19.8 % | 12.1 / 10.7 / 17.4 |
| Dominant text block, conf ≥ 0.1, one pass (kept) | 3.4 % | 2.4 % | 4.5 % | 8.0 % | 2.0 % | 1.9 % | 6.4 % | 4.8 / 3.5 / 2.1 |
| Dominant text block, conf ≥ 0.1, crop and read again | 4.7 % | 4.4 % | 5.0 % | 10.7 % | 2.8 % | 3.4 % | 7.8 % | 6.6 / 4.0 / 3.5 |
| Dominant text block, conf ≥ 0.3, one pass | 4.3 % | 3.8 % | 4.9 % | 8.4 % | 3.6 % | 1.9 % | 7.5 % | 5.2 / 4.7 / 3.1 |
| Dominant text block, conf ≥ 0.3, crop and read again | 5.0 % | 5.5 % | 4.6 % | 10.5 % | 4.6 % | 3.4 % | 7.1 % | 5.4 / 5.5 / 4.2 |

**Error analysis behind it.** On the old path, photo-scene errors were mostly junk read from the background ("WEUCo4C 10 THE রান্নাঘর") and words in the wrong order ("Clinic Eye"). The sign text itself was usually right.

### Test, read once

| Method | CER | Bangla CER | English CER | WER | Clean | Distorted | Photo scene | CER per seed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Old offline path | 11.6 % | 11.8 % | 11.4 % | 18.1 % | 2.4 % | 8.5 % | 24.0 % | 9.4 / 11.0 / 14.4 |
| App OCR mode as shipped | 9.3 % | 11.9 % | 6.6 % | 15.8 % | 5.8 % | 11.2 % | 10.7 % | 10.8 / 7.0 / 9.9 |
| + Task 1 text region | 2.4 % | 3.6 % | 1.3 % | 8.3 % | 1.1 % | 2.8 % | 3.4 % | 2.0 / 2.5 / 2.8 |

The text region alone takes test CER from 9.3 % (app) to 2.4 %. The photo-scene condition falls from 10.7 % to 3.4 %.

### Speed on CPU (the Pi's path)

`results/ocr_bench/cpu_latency.json`: laptop CPU, EasyOCR quantised CPU recognizer, 60 test images. This is not a Pi number.

| Pipeline | Median | 95th percentile | CER on these 60 images |
|---|---:|---:|---:|
| App OCR mode as shipped | 1923 ms | 5194 ms | 13.2 % |
| Final pipeline (v2, now in the app) | 1851 ms | 5201 ms | 1.8 % |

The region step costs nothing measurable, because it reuses the boxes EasyOCR already returns.
