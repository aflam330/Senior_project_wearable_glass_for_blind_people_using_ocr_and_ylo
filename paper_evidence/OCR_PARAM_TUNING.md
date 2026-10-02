# EasyOCR parameter tuning

Validation only for the choice. 144 images, three seeds, GPU. Rule fixed before the run: lowest raw character error, then Bangla character error, then seconds. The 24-phrase set and the 80-image set were read once after `slope_ths=0.2` won. File: `savior_glass/results/ocr_improve_summary.json`.

The glass now passes `OCR_SLOPE_THS` (default 0.2) into `readtext`. Canvas size stays 2560. Decoder stays beam search, width 5.

## Why slope moves the error

`group_text_box` treats a box as a horizontal line only when both edge slopes are below `slope_ths`. The default is 0.1, which is about tan(6°). The wild renders rotate by up to 6°. Those lines were sent to the free-box path instead of the line merger. `slope_ths=0.2` keeps them in the merger. That is the whole change.

## Eligible candidates (scored before any held-out read)

| Setting | Raw CER | Bangla CER | English CER | WER | Seconds |
|---|---:|---:|---:|---:|---:|
| slope 0.1, beam search, mag 1, canvas 2560 | 0.1417 | 0.2168 | 0.0666 | 0.3368 | 31.45 |
| canvas 640 | 0.1409 | 0.2016 | 0.0802 | 0.3507 | 17.68 |
| decoder greedy | 0.1442 | 0.2218 | 0.0666 | 0.3472 | 32.01 |
| mag_ratio 1.5 | 0.1389 | 0.2223 | 0.0556 | 0.3403 | 56.58 |
| mag_ratio 2.0 | 0.1324 | 0.2092 | 0.0556 | 0.3403 | 100.12 |
| text_threshold 0.5, low_text 0.3, link_threshold 0.3 | 0.1312 | 0.2069 | 0.0556 | 0.3368 | 32.25 |
| slope_ths 0.2 | 0.0926 | 0.1713 | 0.0139 | 0.3056 | 33.04 |
| contrast_ths 0.05, adjust_contrast 0.7 | 0.1417 | 0.2168 | 0.0666 | 0.3368 | 33.43 |

Per seed, slope 0.2 beat slope 0.1 on every seed (48 images each):

| Seed | Baseline CER | Slope 0.2 CER | Baseline Bangla | Slope 0.2 Bangla |
|---:|---:|---:|---:|---:|
| 9000 | 0.1282 | 0.0841 | 0.2194 | 0.1683 |
| 9100 | 0.1703 | 0.1066 | 0.2194 | 0.1715 |
| 9200 | 0.1266 | 0.0871 | 0.2116 | 0.1741 |

## Not available for Bangla

EasyOCR `recognition_models` has Bengali only under gen1 (`bengali_g1`). Gen2 (the transformer recognizer) has no Bengali entry. `detect_network='craft'` is the default and is what the table used. DBNet18 downloaded and then failed in the forward pass (deformable convolution did not compile). It has no CER.

## Sensitivity, not eligible

Scored after the held-out read, so not used to change the glass:

| slope_ths | Raw CER | Bangla CER | English CER |
|---:|---:|---:|---:|
| 0.15 | 0.0995 | 0.1713 | 0.0278 |
| 0.20 (cited) | 0.0926 | 0.1713 | 0.0139 |
| 0.30 | 0.0857 | 0.1713 | 0.0000 |

Bangla error is the same at 0.15, 0.20, and 0.30. The extra overall gain at 0.30 is English on this validation list. It was not given a held-out read.

## Held-out, one read

| Set | slope 0.1 raw / Bangla | slope 0.2 raw / Bangla |
|---|---|---|
| 24 phrases, 80 images | 0.1329 / 0.2449 | 0.1162 / 0.2324 |
| 80-image phrase list | 0.0990 / 0.1746 | 0.0831 / 0.1428 |

Latency on that GPU is about 0.21 seconds per image for both slope values (80 images in 16.9–18.4 seconds). The earlier canvas-2560 timing of about 120 seconds for 80 images was a CPU run. This file does not replace that Pi-relevant CPU timing.

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

### Search (validation only)

- **Search space:**
  - text_threshold, low_text, link_threshold;
  - mag_ratio 1.0 / 1.5 / 2.0;
  - contrast_ths, adjust_contrast;
  - decoder (greedy / beam search), slope_ths;
  - canvas 1280 / 2560, minimum box confidence.
- **Procedure:** 30 random configurations plus the defaults were scored on val seed 0 (seed-0 CER ranged from 2.7 to 10.2 %). The 4 best and the defaults were re-scored on all three val seeds.
- **Fixed parts:** CRAFT is the only detector that runs here. EasyOCR has one Bangla recognizer (a CRNN, `bengali.pth`); there is no Bangla 'transformer' recognizer to try.

| Method | CER | Bangla CER | English CER | WER | Clean | Distorted | Photo scene | CER per seed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| config 00 (defaults) | 3.4 % | 3.2 % | 3.5 % | 7.6 % | 2.9 % | 0.8 % | 6.4 % | 3.7 / 4.1 / 2.4 |
| config 06 | 4.0 % | 3.5 % | 4.5 % | 8.4 % | 2.9 % | 0.7 % | 8.3 % | 3.0 / 4.2 / 4.8 |
| config 12 (kept) | 2.8 % | 2.9 % | 2.7 % | 6.1 % | 2.5 % | 0.4 % | 5.6 % | 2.7 / 3.0 / 2.7 |
| config 16 | 3.8 % | 3.0 % | 4.5 % | 8.0 % | 2.9 % | 0.7 % | 7.6 % | 2.7 / 4.2 / 4.3 |
| config 27 | 2.8 % | 2.9 % | 2.8 % | 6.1 % | 2.4 % | 0.5 % | 5.6 % | 2.7 / 3.1 / 2.7 |

**Kept:** `{"text_threshold": 0.8, "low_text": 0.3, "link_threshold": 0.4, "mag_ratio": 1.0, "contrast_ths": 0.1, "adjust_contrast": 0.7, "decoder": "beamsearch", "slope_ths": 0.4, "canvas_size": 2560, "min_conf": 0.2}` — validation CER 2.83 % against 3.37 % with the defaults.

### Test, read once

| Method | CER | Bangla CER | English CER | WER | Clean | Distorted | Photo scene | CER per seed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Task 2 (CLAHE, Task 1 settings) | 3.3 % | 4.2 % | 2.4 % | 11.7 % | 2.6 % | 3.0 % | 4.2 % | 2.9 / 3.9 / 3.1 |
| + Task 4 parameters | 2.4 % | 3.2 % | 1.6 % | 7.9 % | 2.4 % | 2.6 % | 2.2 % | 3.3 / 1.4 / 2.4 |
