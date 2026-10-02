# Fine-tuning the OCR recognizer

No recognizer was fine-tuned, and none was trained from scratch. There is no character-error number for a fine-tuned model.

The repo does not contain a Bangla line-image training set that is separate from the offline renderer. The 80-image phrases and the 24-phrase list are the held-out renders. Training a CRNN or TrOCR on images drawn by `eval_ocr_offline.py` would use the same font, the same canvas, and the same noise process as the test. That would not be an independent training set, so it was not done.

EasyOCR’s Bengali recognizer is the shipped gen1 model (`bengali_g1`). It was not updated.

What did change is a decode-time threshold, `slope_ths=0.2`, chosen on a different phrase list. That result is in `OCR_PARAM_TUNING.md`. It is not a fine-tune.

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

### Training (`savior_glass/ocr_bench/train_task7.py`)

- **Model:** EasyOCR's Bangla + English recognizer (CRNN, CTC, 54 M parameters), fine-tuned from its released weights.
- **Training text:**
  - words from the general 50k frequency lists (46422 Bangla, 46487 English), with every word of every benchmark phrase removed (350 forms);
  - random numbers in both scripts.
- **Rendering:** shaped with HarfBuzz in six Bangla and four Latin training fonts that are never scored, with blur, low resolution, rotation, noise and JPEG damage.
- **Optimisation:** 2000 steps, batch 16, AdamW 3e-5, mixed precision.
- **Checkpoint choice:** greedy CER on line crops of the val phrases in val fonts, every 250 steps. Three seeds.

| Seed | Best step | Val crop CER before | Val crop CER after |
|---|---:|---:|---:|
| 42 | 500 | 0.55 % | 0.12 % |
| 43 | 1500 | 0.55 % | 0.32 % |
| 44 | 1500 | 0.55 % | 0.20 % |

### End to end on validation (the decision)

| Method | CER | Bangla CER | English CER | WER | Clean | Distorted | Photo scene | CER per seed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Final pipeline, original recognizer | 2.3 % | 1.9 % | 2.7 % | 5.1 % | 2.3 % | 0.4 % | 4.2 % | 2.0 / 2.2 / 2.6 |
| Final pipeline, fine-tuned seed 42 | 2.5 % | 1.5 % | 3.6 % | 5.1 % | 1.0 % | 1.1 % | 5.5 % | 3.5 / 1.7 / 2.3 |
| Final pipeline, fine-tuned seed 43 | 2.7 % | 1.1 % | 4.4 % | 4.6 % | 1.9 % | 1.2 % | 5.0 % | 3.0 / 2.5 / 2.7 |
| Final pipeline, fine-tuned seed 44 | 2.6 % | 1.1 % | 4.1 % | 3.8 % | 1.3 % | 1.0 % | 5.4 % | 3.4 / 1.8 / 2.6 |

**Not adopted.** Fine-tuning lowers Bangla CER on validation but raises English CER, so overall validation CER is higher than with the original recognizer. Training on synthetic text alone makes the model partly forget English. The test read below is reported for completeness and was not used for the decision.

### Test, read once per seed

| Method | CER | Bangla CER | English CER | WER | Clean | Distorted | Photo scene | CER per seed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Final pipeline, original recognizer | 2.4 % | 3.2 % | 1.6 % | 7.9 % | 2.4 % | 2.6 % | 2.2 % | 3.3 / 1.4 / 2.4 |
| fine-tuned seed 42 | 1.8 % | 2.6 % | 1.0 % | 5.3 % | 1.4 % | 2.5 % | 1.5 % | 2.9 / 0.8 / 1.6 |
| fine-tuned seed 43 | 1.8 % | 2.5 % | 1.1 % | 3.9 % | 1.2 % | 2.8 % | 1.4 % | 1.9 / 1.3 / 2.2 |
| fine-tuned seed 44 | 2.1 % | 2.9 % | 1.2 % | 4.8 % | 1.2 % | 2.2 % | 2.8 % | 2.8 / 0.7 / 2.7 |

**Test disagrees with validation.** On test the fine-tuned seeds score 1.8 %, 1.8 %, 2.1 % against 2.4 % for the original recognizer, lower in every seed. That direction is the opposite of validation. The decision stays with validation: switching now would be choosing on the test set. Fine-tuning is the most promising next step, on real glass photos and with English kept in the training mix so it is not forgotten. A fresh test set is needed to show that it helps.
