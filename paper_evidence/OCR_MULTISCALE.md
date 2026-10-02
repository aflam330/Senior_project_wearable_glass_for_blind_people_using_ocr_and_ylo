# Multi-scale OCR

Validation list, 144 images, three seeds, before the held-out read. At each image the recognizer ran at scale 1.0 and scale 1.5. The hypothesis with the higher mean detection confidence was kept. Ties do not prefer the longer string.

| Method | Raw CER | Bangla CER | English CER | WER | Seconds |
|---|---:|---:|---:|---:|---:|
| Scale 1.0 only (baseline, slope 0.1) | 0.1417 | 0.2168 | 0.0666 | 0.3368 | 31.45 |
| Scales 1.0 and 1.5, higher confidence | 0.1479 | 0.2168 | 0.0790 | 0.3438 | 66.52 |

Bangla error did not change. English error went up. The run took about twice as long. It was not kept.

`mag_ratio` 1.5 and 2.0 are a different knob (detector magnification, not two full reads). They are in `OCR_PARAM_TUNING.md`. Both were slightly better than the baseline and worse than `slope_ths=0.2`, and both were slower (56.6 s and 100.1 s versus 33.0 s on the same 144 images).

No majority vote was scored. With two scales, a vote does not break ties, and a third scale was not run after the confidence rule had already lost on validation.

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

The preprocessed image was read at 1.0×, 1.5× and 2.0× with the Task 4 settings. Two ways of choosing between the readings were tried:
- **most confident:** the reading with the highest character-weighted box confidence;
- **vote:** the medoid, i.e. the reading with the smallest total edit distance to the other two.

### Validation (3 seeds)

| Method | CER | Bangla CER | English CER | WER | Clean | Distorted | Photo scene | CER per seed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| scale1.0 | 2.8 % | 2.9 % | 2.7 % | 6.1 % | 2.5 % | 0.4 % | 5.6 % | 2.7 / 3.0 / 2.7 |
| scale1.5 | 3.6 % | 2.8 % | 4.4 % | 7.7 % | 2.2 % | 0.7 % | 8.0 % | 3.0 / 4.0 / 3.9 |
| scale2.0 | 3.5 % | 2.6 % | 4.4 % | 7.5 % | 1.5 % | 0.7 % | 8.3 % | 4.0 / 3.1 / 3.4 |
| most_confident | 3.0 % | 2.1 % | 3.8 % | 6.5 % | 2.1 % | 0.5 % | 6.2 % | 2.5 / 2.5 / 4.0 |
| vote_medoid | 2.9 % | 2.6 % | 3.2 % | 6.2 % | 1.9 % | 0.4 % | 6.2 % | 2.6 / 2.7 / 3.3 |

**Not kept.** The single scale is best (2.83 %). Larger scales lower Bangla CER a little but raise English and photo-scene CER, and every extra scale costs a full OCR pass. Nothing new was read on test (the single scale is the Task 4 pipeline).
