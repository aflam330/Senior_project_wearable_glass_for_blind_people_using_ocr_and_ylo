# Bangla-specific OCR engines

Compared with EasyOCR on the validation phrase list only (144 images, seeds 9000 / 9100 / 9200). The 24-phrase set and the 80-image set were not opened for these engines. None of them replaced `slope_ths=0.2`.

EasyOCR on that same list, full glass preprocess, slope 0.2: raw CER 0.0926, Bangla 0.1713, English 0.0139, about 31 seconds.

## PaddleOCR

Not run. `pip install paddlepaddle` on this Python 3.14 install returned `No matching distribution found for paddlepaddle`. No Bangla CER.

## Tesseract 5

Not run. `pytesseract` 0.3.13 is installed. The `tesseract` binary is not on `PATH`, and `ben.traineddata` was not installed. The task’s `apt` and `/usr/share/tesseract-ocr/5/tessdata/` steps are for Linux. This machine is Windows. No Bangla CER.

## TrOCR

`microsoft/trocr-base-printed` ran on GPU. It is an English printed-text model. Transformers 5.18 reported `encoder.pooler.dense.weight` and `bias` as missing and newly initialized. Generation still ran. 144 images took 38.6 seconds.

| Slice | Character error (case-sensitive, same metric as EasyOCR) |
|---|---:|
| All 144 | 1.095 |
| Bangla, 72 | 1.088 |
| English, 72 | 1.102 |

Bangla lines came back as Latin garbage: `আসন নম্বর` → `***`, `দরজা বন্ধ` → `MASI 468`, `জানালা খুলুন` → `GIRINT SDN`.

The English number is high because the metric is case-sensitive and TrOCR emits capitals, often with the space removed or a trailing `~~`. On the same 72 English images, case-folded character error is 0.420. Examples from seed 9000: `Seat 4` → `SEAT4` (case-sensitive 0.667, case-folded 0.167), `Door Closed` → `DOOR CLOSED` (0.727 and 0.000), `Window Open` → `WINDOW OPEN ~~` (1.000 and 0.273). EasyOCR’s English error on this list is 0.0139 with case preserved. TrOCR is not the better English reader here, and it is not a Bangla reader.

No Bangla TrOCR checkpoint was trained. See `OCR_FINETUNE.md`.

## What is kept

EasyOCR, CRAFT, beam search, canvas 2560, `slope_ths=0.2`. Held-out numbers for that pipeline are in `OCR_PARAM_TUNING.md`.

| Set, read once | Raw CER | Bangla CER |
|---|---:|---:|
| 24 phrases (was 0.1329 / 0.2449) | 0.1162 | 0.2324 |
| 80-image list (was 0.0990 / 0.1746) | 0.0831 | 0.1428 |

The target, raw under 0.10 and Bangla under 0.20, holds on the 80-image list. It does not hold on the 24-phrase list.

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

### Engines

- **Tesseract 5.5.2:** tesserocr wheel, `tessdata_best` ben+eng. The UB-Mannheim installer needs administrator rights; the silent install was cancelled.
- **PaddleOCR 3.7:** on Python 3.10, because there is no paddle wheel for 3.14. It has no Bangla model: `lang='bn'` and `'bengali'` give "No models are available". The oneDNN path crashes on Windows, so it runs with `enable_mkldnn=False`.
- **TrOCR:** reads the Task 1 text block with the language given (oracle). English uses `microsoft/trocr-base-printed`. Bangla uses the community `nightsagittariuswolf/SWIN_TrOCR_Bangla_model`; its processor config names a removed class, so it was rebuilt from its parts.

### Validation (3 seeds)

| Method | CER | Bangla CER | English CER | WER | Clean | Distorted | Photo scene | CER per seed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| EasyOCR + Task 1 text region | 3.4 % | 2.4 % | 4.5 % | 8.0 % | 2.0 % | 1.9 % | 6.4 % | 4.8 / 3.5 / 2.1 |
| tesseract frame psm3 | 41.8 % | 48.9 % | 34.7 % | 49.8 % | 11.4 % | 20.3 % | 93.8 % | 41.6 / 41.0 / 42.9 |
| tesseract frame psm6 | 84.9 % | 81.8 % | 87.9 % | 150.1 % | 8.1 % | 10.7 % | 235.9 % | 86.1 / 77.5 / 91.0 |
| tesseract frame psm11 | 105.5 % | 108.6 % | 102.4 % | 196.4 % | 9.4 % | 12.6 % | 294.5 % | 117.6 / 104.6 / 94.3 |
| tesseract block psm6 | 26.3 % | 30.4 % | 22.3 % | 49.4 % | 10.1 % | 11.9 % | 57.0 % | 33.0 / 28.0 / 18.0 |
| tesseract block psm7 | 23.4 % | 25.9 % | 20.9 % | 39.8 % | 9.6 % | 11.9 % | 48.8 % | 25.1 / 26.2 / 18.9 |
| trocr block oracle lang | 78.0 % | 79.6 % | 76.5 % | 95.5 % | 74.9 % | 77.8 % | 81.4 % | 79.4 / 79.2 / 75.5 |
| PaddleOCR PP-OCRv5 (English items only) | 43.4 % | — | 43.4 % | 45.9 % | 0.0 % | 0.0 % | 130.3 % | 91.8 / 23.0 / 15.5 |

**Notes.**
- The Microsoft TrOCR answers in capitals (it was trained on receipts), so its CER is case-penalised.
- The community Bangla TrOCR returns unrelated words, for example "রান্নাঘর" → "রামানগঞ্জ".
- PaddleOCR reads every piece of text in a scene, so its photo-scene CER exceeds 100 %.

### Test, read once

| Method | CER | Bangla CER | English CER | WER | Clean | Distorted | Photo scene | CER per seed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| EasyOCR final pipeline | 2.4 % | 3.2 % | 1.6 % | 7.9 % | 2.4 % | 2.6 % | 2.2 % | 3.3 / 1.4 / 2.4 |
| Tesseract 5 (tesseract block psm7) | 16.6 % | 18.5 % | 14.7 % | 34.4 % | 7.4 % | 9.8 % | 32.5 % | 19.5 / 13.3 / 16.9 |
| PaddleOCR (English items only) | 11.0 % | — | 11.0 % | 9.5 % | 0.0 % | 0.0 % | 33.1 % | 9.1 / 16.0 / 8.1 |

EasyOCR with the text region stays far ahead. Tesseract is the best other engine for Bangla, at about 7× the error.
