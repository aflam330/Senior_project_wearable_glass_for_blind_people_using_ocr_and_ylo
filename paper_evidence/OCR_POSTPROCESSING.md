# OCR post-processing

The cited read is still EasyOCR with `slope_ths=0.2` and no extra corrector. Lexicon repair was already in the glass. It is reported here. It was not used to choose the pipeline.

## Lexicon on the sets that were read once

`repair_ocr_text` NFC-normalizes, then replaces a short line by a lexicon phrase when the character error is at most 0.34, else replaces a word when the error is at most 0.40 and the edit distance is at most 2. The 80-image phrases are in `assets/ocr_lexicon.txt`. The 24-phrase list is not.

| Set | Pipeline | Raw CER | Lexicon CER | Raw Bangla | Lexicon Bangla |
|---|---|---:|---:|---:|---:|
| 24 phrases | slope 0.1 | 0.1329 | 0.1422 | 0.2449 | 0.2387 |
| 24 phrases | slope 0.2 | 0.1162 | 0.1256 | 0.2324 | 0.2262 |
| 80-image list | slope 0.1 | 0.0990 | 0.0267 | 0.1746 | 0.0300 |
| 80-image list | slope 0.2 | 0.0831 | 0.0117 | 0.1428 | 0.0000 |

On the 24-phrase list the lexicon raises overall error (it rewrites unseen lines toward phrases that are not the truth). On the 80-image list it looks very good because those phrases are in the word list. That repaired number is not an unseen-text result.

The validation phrases are also absent from the lexicon. On the slope-0.2 read of those 144 images, word-level repair still moved Bangla CER from 0.1713 to 0.1537 and left English at 0.0139 (overall 0.0926 to 0.0838). Some validation words happen to sit near words that are in the list. That score was not used to pick the pipeline, and the repair was not turned on or off because of it. The 24-phrase result above is the one that matches the “not in the word list” claim: there, lexicon repair makes the overall error worse.

## Character rule, validation folds only

On the slope-0.2 hypotheses, a substitution was kept when it occurred at least 8 times in two seeds and the reverse substitution was less than a quarter of that. The third seed was scored. All three folds learned one rule, `ো` → `া`.

| Held seed | Raw CER | After the rule |
|---:|---:|---:|
| 9000 | 0.0841 | 0.0763 |
| 9100 | 0.1066 | 0.0988 |
| 9200 | 0.0871 | 0.0792 |

Mean held-seed CER 0.0926 → 0.0848. The rule was not applied to the 24-phrase set or the 80-image set, and it is not in the glass. A global rewrite of `ো` to `া` will also change a correct `ো` (the validation list itself contains `ধোঁয়া`). The net drop is real on this list and is not safe as a product rule.

## Not installed

- `bnlp` / `bnlp-toolkit`: the install failed while building `gensim` (`Microsoft Visual C++ 14.0 or greater is required`). No spell-check CER.
- `kenlm`: the wheel build failed with the same missing Visual C++ toolset. No language-model CER.
- No confusion-matrix corrector was fit on the 24-phrase set or the 80-image set.

Source files: `savior_glass/results/ocr_improve_summary.json`, `savior_glass/results/ocr_improve_extra.json`.

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

All methods were applied to the same saved validation readings of the Task 4 pipeline (no new OCR).

- **rules:** NFC; removes invisible characters and stray symbols at word edges; puts digits in the script of the words around them.
- **dictionary:** SymSpell over general 50k-word frequency lists (Bangla and English, hermitdave/FrequencyWords, MIT), with a frequency margin.
- **byt5:** `Stup702/ByT5-Bengali-OCR-Correction`.
- **lexicon:** the app's old repair against its 24-phrase list.

**Not available:**
- **KenLM:** does not build on this laptop; there is no C++ compiler. The frequency lists act as a unigram model inside the dictionary method.
- **bnlp:** installed on Python 3.10, but it has no spell checker.
- **`bangla-spell-checker`:** not on PyPI.

### Validation (3 seeds)

| Method | CER | Bangla CER | English CER | WER | Exact | Time per line |
|---|---:|---:|---:|---:|---:|---:|
| none | 2.8 % | 2.9 % | 2.7 % | 6.1 % | 91.4 % | 0.00 ms |
| rules | 2.3 % | 1.9 % | 2.7 % | 5.1 % | 92.4 % | 0.00 ms |
| dictionary | 3.3 % | 4.0 % | 2.7 % | 8.4 % | 87.5 % | 0.13 ms |
| rules+dictionary | 2.8 % | 2.9 % | 2.6 % | 6.6 % | 90.0 % | 0.14 ms |
| byt5 | 105.4 % | 207.1 % | 3.7 % | 106.8 % | 45.8 % | 419.03 ms |
| rules+byt5 | 104.9 % | 206.2 % | 3.7 % | 106.5 % | 46.1 % | 425.44 ms |
| lexicon | 4.6 % | 3.5 % | 5.7 % | 12.7 % | 79.6 % | 2.74 ms |

**Kept: rules.**
- **Dictionary:** the most frequent nearby word is often the wrong word for a real Bangla sign, so it hurts Bangla.
- **ByT5:** it rewrites and repeats text ("রান্নাঘর" → "রান্নাঘর প্রতিষ্ঠিত হয়") and takes 6–11 s per line on CPU.
- **Lexicon repair:** it pulls new phrases toward its 24 old ones.

### Test, read once

| Method | CER | Bangla CER | English CER | WER | Clean | Distorted | Photo scene | CER per seed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Task 4 pipeline, no post-processing | 2.4 % | 3.2 % | 1.6 % | 7.9 % | 2.4 % | 2.6 % | 2.2 % | 3.3 / 1.4 / 2.4 |
| + rules (final) | 2.4 % | 3.2 % | 1.6 % | 7.9 % | 2.4 % | 2.6 % | 2.2 % | 3.3 / 1.4 / 2.4 |

On test the rules change 42 of 432 readings, but almost all of these are Unicode normalisation, which the metric already applies. The few real edits cost the same number of characters ("টাক| জম দিন" → "টাক জম দিন"). So test CER is unchanged; the spoken text is cleaner.
