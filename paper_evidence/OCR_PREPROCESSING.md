# OCR preprocessing

Same validation list as `OCR_TEXT_REGION.md`: 144 images, seeds 9000 / 9100 / 9200, GPU. The glass preprocess (scale width into 1200–1600, grayscale, bilateral filter, CLAHE) is already inside the baseline. Binarization was not added to the glass. The earlier code comment that binarization hurt EasyOCR is consistent with the Sauvola and Niblack rows below.

The cited pipeline stays full glass preprocess plus `slope_ths=0.2`. Rows marked sensitivity were scored after the held-out sets had already been read, so they were not allowed to replace that choice.

| Method | On top of | Raw CER | Bangla CER | English CER | Seconds | Eligible |
|---|---|---:|---:|---:|---:|---|
| Glass preprocess (baseline) | slope 0.1 | 0.1417 | 0.2168 | 0.0666 | 31.45 | yes |
| Sauvola, window 25, k 0.34 | slope 0.1 | 0.1233 | 0.2284 | 0.0182 | 32.25 | yes |
| Unsharp mask after CLAHE | slope 0.1 | 0.1556 | 0.2291 | 0.0821 | 31.94 | yes |
| Deskew, `minAreaRect`, only if 0.4–12 degrees | slope 0.1 | 0.1365 | 0.1722 | 0.1007 | 32.88 | yes |
| Niblack, window 25, k −0.2 | slope 0.2 | 0.1362 | 0.2537 | 0.0187 | 31.68 | no, later |
| Perspective warp of the ink rectangle | slope 0.2 | 0.3780 | 0.3966 | 0.3595 | 31.39 | no, later |
| Grayscale resize only, no bilateral, no CLAHE | slope 0.2 | 0.1118 | 0.1892 | 0.0343 | 31.82 | no, later |
| CLAHE only, no bilateral | slope 0.2 | 0.1090 | 0.1872 | 0.0307 | 31.41 | no, later |
| Glass preprocess | slope 0.2 | 0.0926 | 0.1713 | 0.0139 | 33.04 | yes, kept |

Sauvola lowered English error and raised Bangla error. It was not kept. Deskew helped Bangla and hurt English. Unsharp, Niblack, and the perspective warp were worse. Dropping bilateral or CLAHE, while keeping slope 0.2, was worse than keeping both (0.109–0.112 versus 0.0926).

`contrast_ths=0.05` and `adjust_contrast=0.7` produced the same text as the baseline on all 144 images (CER 0.1417). That knob did not move this render set.

Word error was not recomputed for the later rows. For the eligible rows it is in `ocr_improve_val.json`.

No held-out read was spent on Sauvola, deskew, unsharp, Niblack, or the perspective warp. The one held-out read is the kept pipeline in `OCR_TEXT_REGION.md`.

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

All variants run on top of the Task 1 text region.

### Validation (3 seeds)

| Method | CER | Bangla CER | English CER | WER | Clean | Distorted | Photo scene | CER per seed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Glass (upscale, bilateral, CLAHE) | 3.4 % | 2.4 % | 4.5 % | 8.0 % | 2.0 % | 1.9 % | 6.4 % | 4.8 / 3.5 / 2.1 |
| Grey only | 4.8 % | 3.5 % | 6.1 % | 8.9 % | 3.2 % | 1.1 % | 10.1 % | 5.3 / 4.5 / 4.6 |
| Colour | 4.3 % | 3.1 % | 5.5 % | 8.3 % | 3.2 % | 1.1 % | 8.5 % | 4.1 / 4.8 / 3.9 |
| No upscale | 4.0 % | 4.0 % | 4.1 % | 7.1 % | 4.0 % | 0.8 % | 7.2 % | 4.8 / 3.3 / 4.0 |
| CLAHE only (kept) | 3.4 % | 3.2 % | 3.5 % | 7.6 % | 2.9 % | 0.8 % | 6.4 % | 3.7 / 4.1 / 2.4 |
| Bilateral only | 4.1 % | 3.2 % | 5.0 % | 8.1 % | 1.7 % | 1.6 % | 8.9 % | 3.6 / 4.4 / 4.4 |
| Unsharp mask | 5.9 % | 4.5 % | 7.2 % | 13.4 % | 5.3 % | 3.3 % | 8.9 % | 6.2 / 5.2 / 6.2 |
| Glass + unsharp | 4.0 % | 2.8 % | 5.2 % | 8.8 % | 2.7 % | 1.5 % | 7.8 % | 4.1 / 3.8 / 4.0 |
| Sauvola binarisation | 5.2 % | 5.2 % | 5.1 % | 12.3 % | 3.6 % | 4.3 % | 7.5 % | 4.6 / 4.9 / 6.0 |
| Niblack binarisation | 12.5 % | 12.2 % | 12.7 % | 31.5 % | 5.8 % | 6.5 % | 25.1 % | 13.1 / 11.7 / 12.6 |
| Deskew | 4.8 % | 4.0 % | 5.5 % | 8.5 % | 3.0 % | 1.5 % | 9.8 % | 4.6 / 4.4 / 5.4 |
| Perspective correction | 8.3 % | 9.5 % | 7.2 % | 13.0 % | 4.9 % | 1.1 % | 19.0 % | 8.3 / 9.2 / 7.5 |
| Deskew + CLAHE | 3.6 % | 2.9 % | 4.3 % | 8.4 % | 3.5 % | 1.8 % | 5.5 % | 3.9 / 2.7 / 4.2 |

- **Kept:** CLAHE only had the lowest validation CER (3.37 %), by 0.08 points over the glass preprocessing. That gap is within seed noise.
- **What hurt:** binarisation (Sauvola, Niblack) and the perspective warp hurt. Deskew alone hurt.

### Test, read once

| Method | CER | Bangla CER | English CER | WER | Clean | Distorted | Photo scene | CER per seed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Task 1 text region (glass preprocessing) | 2.4 % | 3.6 % | 1.3 % | 8.3 % | 1.1 % | 2.8 % | 3.4 % | 2.0 / 2.5 / 2.8 |
| + Task 2 CLAHE only | 3.3 % | 4.2 % | 2.4 % | 11.7 % | 2.6 % | 3.0 % | 4.2 % | 2.9 / 3.9 / 3.1 |

On test the CLAHE-only step is worse than the glass preprocessing (3.3 % against 2.4 %). The validation choice did not carry over. The Task 4 parameters, chosen on top of it, bring test CER back to 2.4 % (see `OCR_PARAM_TUNING.md`).
