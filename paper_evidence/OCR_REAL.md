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

## Next

A settings study on real words, with its own real validation sample that does not overlap these 1,000 test words, is described in the section below if it has been run.
