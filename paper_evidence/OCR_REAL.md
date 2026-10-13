# Text reading on real photographs (2026-10-13)

Source: `savior_glass/ocr_bench/real_words.py` → `savior_glass/results/ocr_bench/real_words_mlt19.json`.

## Data

Word images from the ICDAR 2019 MLT scene-text set (street signs, shop fronts, notices), in the Hugging Face copy `sai-santhosh/mlt19-train-ocr-data`. Each image has its text typed out. The pool holds 3,935 readable Bangla words and 58,840 readable Latin-script words (English, French, German, Italian). Words its annotators marked unreadable ("###") are left out, as the benchmark does.

These are the first real photographs the text reader has been scored on. None was used for training or tuning.

## Protocol (fixed before the run)

- 500 Bangla and 500 Latin-script words, drawn at random with seed 0.
- Each word image is scaled so that its longer side is 400 pixels and placed in the middle of a 640 × 480 grey frame, the way the glass would see a sign.
- The frame goes through the pipeline that is in the glass (1,200 px width, greedy decoder, main-text-block rule, clean-up rules).
- Scored with the benchmark's CER and WER, case-sensitive.

## Result

| Words | n | CER | 95 % interval | Word exactly right | Nothing read |
|---|---:|---:|---:|---:|---:|
| Bangla, all | 500 | 41.7 % | 38.0–45.5 | 36.8 % | 24.6 % |
| Bangla, upright | 458 | 37.8 % | 34.0–41.9 | 39.7 % | 21.6 % |
| Bangla, rotated by 90° | 42 | 83.9 % | 73.2–93.7 | 4.8 % | 57.1 % |
| Latin script, all | 500 | 64.9 % | 60.9–68.8 | 20.4 % | 48.4 % |
| Latin script, upright | 422 | 58.5 % | 54.2–62.9 | 24.2 % | 40.1 % |
| Latin script, upright and at least 32 px high | 282 | 46.8 % | 41.4–52.4 | 32.6 % | 27.7 % |
| Latin script, rotated by 90° | 78 | 99.4 % | 98.1–100 | 0 % | 93.6 % |

For comparison, the same pipeline scores 2.4 % CER (Bangla 3.2 %, English 1.6 %) on the project's rendered benchmark.

## Reading the result

- **The benchmark figures do not carry over to real photographs.** The gap is a factor of ten or more.
- **Much of the error is silence.** On a quarter of the Bangla words and half of the Latin-script words the glass read nothing. The text-finding settings (text threshold 0.8, low-text 0.3) and the confidence filter were chosen on rendered text.
- **Rotated words are not read.** No setting of the glass handles text turned by 90°.
- **Small words fail.** 140 of the 422 upright Latin-script words are under 32 pixels high in the original photograph; enlarging them does not bring back detail.
- **Latin script here is not only English.** The recogniser has English and Bangla models; French, German and Italian words with accents cannot be fully right.
- An outside reference: the BanglaWild benchmark (arXiv 2608.03884) reports 48.4 % CER for plain EasyOCR on real Bangla scene photographs. Different data, so only the order of magnitude is comparable.

## Limits of this test

- Single words, not signs with several lines. The main-text-block rule has nothing to choose between.
- The words come from many countries; Bangla scene text from India (label `bn_IN`), not Bangladesh.
- A word image placed on a grey frame is easier than a word in its scene in one way (no clutter) and harder in another (no context).
- One seed, one sample of 500 per script.

## Settings study on real words

Source: `savior_glass/ocr_bench/real_words_tune.py` → `results/ocr_bench/real_words_tune.json`.

Protocol, fixed before the run: four candidate settings; a real validation sample of 150 Bangla and 150 Latin-script words (seed 1) that does not overlap the 1,000 test words; a candidate is admissible only if its CER on the rendered benchmark's validation set stays within 0.5 points of the current setting; the admissible candidate with the lowest real-validation CER wins and is run once on the 1,000 test words.

| Candidate (text / low-text / link threshold, minimum confidence) | Real validation CER (mean of both scripts) | Nothing read, Bangla / Latin | Rendered validation CER | Admissible |
|---|---:|---:|---:|---|
| A current (0.8 / 0.3 / 0.4, 0.2) | 51.1 % | 21 % / 51 % | 2.14 % | yes |
| B EasyOCR defaults (0.7 / 0.4 / 0.4, 0.2) | 56.7 % | 25 % / 58 % | 3.39 % | no |
| **C current thresholds, confidence 0.1** | **47.8 %** | 10 % / 43 % | 2.58 % | yes |
| D permissive (0.6 / 0.3 / 0.4, 0.1) | 47.9 % | 9 % / 42 % | 2.95 % | no |

C wins. On the 1,000 real test words, read once with C:

| Words | CER before | CER with C | Nothing read before | Nothing read with C | Word exactly right before / with C |
|---|---:|---:|---:|---:|---:|
| Bangla | 41.7 % | 39.2 % | 24.6 % | 17.0 % | 36.8 % / 37.4 % |
| Latin script | 64.9 % | 62.5 % | 48.4 % | 38.0 % | 20.4 % / 20.4 % |

- The gain is real but small: about 2.5 points of CER, and 8 to 10 points fewer words left unread. The words that are now read are often read wrongly, so exact matches barely move.
- The price is on rendered text: validation CER 2.14 % → 2.58 %.
- `OCR_V2_MIN_CONF` is now 0.1 in the glass.
- EasyOCR's own detector defaults (B) were worse on both kinds of data, so the detector thresholds chosen earlier were not the cause.
- **Settings are exhausted.** The remaining error is in the recogniser on real text. The next step is training data from real photographs, not another threshold.
