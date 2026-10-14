# Text reading on real photographs: retrained recogniser, and the read-width study (14 October 2026)

Result files: `savior_glass/results/ocr_bench/real_ft_eval.json`, `real_ft_seed42/43/44.json` (training), `width_study.json`, `blend_ship_check.json`.
Scripts: `ocr_bench/train_real_words.py`, `ocr_bench/real_ft_eval.py`, `ocr_bench/width_study.py`, `ocr_bench/blend_ship_check.py`.
Every candidate was read through the glass's own reading function (`OCRMode._read_v2`, CPU, 8-bit LSTM as on the Pi). Rules were written into the scripts before each run. The test sets were read once, after the choice.

Data: real word photographs from ICDAR 2019 MLT, split into training words, a validation sample (150 Bangla + 150 Latin-script) and the 1,000 test words used since 13 October; and the rendered benchmark (validation and test, 432 images each).

## 1. The recogniser: retrained, then blended with the released one

EasyOCR's Bangla+English recogniser was fine-tuned on real training words mixed with rendered words, three times (seeds 42, 43, 44).

| Recogniser (validation) | Real words, mean CER | Bangla | Latin | Rendered CER | Within 0.5 points? |
|---|---:|---:|---:|---:|---|
| Released (EasyOCR) | 47.8 % | 34.2 % | 61.3 % | 2.58 % | reference |
| Fine-tuned, seed 42 / 43 / 44 | 34.2 / 34.7 / 35.3 % | 22.0 / 21.5 / 23.6 % | 46.3 / 47.9 / 47.1 % | 3.82 / 3.66 / 3.34 % | **no** (English text got worse: 5.84 / 5.28 / 4.91 %) |
| Blend 0.5, seed 42 / 43 / 44 | 37.2 / 36.9 / **36.7** % | 24.4 / 25.4 / 24.3 % | 50.0 / 48.4 / 49.0 % | 2.33 / 2.44 / 2.63 % | **yes, all three** |

Fully retrained models learn the real words but forget part of the clean printed text. Blending the weights half and half with the released ones (θ = 0.5·released + 0.5·fine-tuned; the rule for this was added on 14 October, after seed 42 failed and before any other model was scored) keeps most of the gain and loses nothing on printed text. By the rule, the admissible model with the lowest real-validation error was chosen: **blend 0.5, seed 44**.

**Rotation retry** (turn the picture 90° each way when nothing is read), scored with that recogniser: Latin-script words left unread fell from 32.7 % to 28.7 %, but the mean CER did not change (36.65 % both). The rule asked for a lower CER, so it stays **off**.

**Test, read once** (1,000 real test words; rendered test set):

| | Before (13 October) | Now | Change |
|---|---:|---:|---:|
| Real words, Bangla CER | 39.2 % | **28.7 %** | −10.5 points |
| Real words, Latin-script CER | 62.5 % | **50.6 %** | −11.9 points |
| Words with nothing read, Bangla / Latin | 17 % / 38 % | 12.4 % / 32.6 % | |
| Words read exactly, Bangla / Latin | — | 47.8 % / 30.6 % | |
| Rendered test CER (Bangla / English) | 2.46 % (3.25 / 1.67) | **2.27 %** (3.03 / 1.51) | −0.19 points |

The gain on real photographs held on the test words, and printed text is read slightly better as well. The blended recogniser is now the glass's default (`models/ocr_recognizer_real_blend/`, `OCR_RECOGNIZER_PATH`). It has the same network as the released one, so it costs no time.

Shipping: the weights are 215 MB in 32-bit numbers, over GitHub's 100 MB limit. They are stored in half precision in two parts (56 and 52 MB) and turned back into 32-bit numbers when loaded; the CPU path then makes the LSTM and linear layers 8-bit as before. `blend_ship_check.json`: the shipped copy reads the validation sets as the evaluated one did (real validation 36.5 % against 36.7 %, rendered 2.54 % against 2.63 %).

**The Pi's configuration** (8-bit text detector) with this recogniser, validation: rendered 2.86 %, real words 40.1 %. Against the reference of 2.58 % that is +0.28 points, inside the 0.5-point rule, so the 8-bit detector on the Pi is no longer an exception to the rule (with the released recogniser it was +0.57).

## 2. Read width: how far a camera picture is enlarged before reading

The glass enlarges a 640-pixel camera picture to 1,200 pixels before the text detector, whose time grows with the number of pixels. Widths 1,040, 960 and 880 were scored with both detectors (this study used the released recogniser).

| Detector, width (validation) | Real words, mean CER | Rendered CER | Admissible by the rule |
|---|---:|---:|---|
| Full precision, 1,200 | 47.8 % | 2.58 % | reference |
| Full precision, 1,040 / 960 / 880 | 48.3 / 45.9 / 46.4 % | 2.59 / 3.09 / 2.21 % | no / no (by 0.01) / yes |
| 8-bit, 1,200 | 51.6 % | 3.15 % | reference |
| 8-bit, 1,040 / 960 / 880 | 48.4 / 48.0 / 47.4 % | 3.10 / 2.99 / 2.98 % | yes / yes / yes |

The rule picked 880 for both detectors. **The single test read did not confirm it**: at 880 the rendered test CER was 3.15 % (full precision) and 3.12 % (8-bit), against 2.46 % at 1,200 with full precision, with Bangla at 4.9 % and 4.6 %. The validation figures also move up and down with width in no order (880 scored better than 960 and 1,040), so the differences between widths on 432 validation images are about as large as the noise.

**Decision: the read width stays at 1,200.** This departs from the rule written before the run, in the cautious direction: a setting that the test shows to read printed Bangla worse is not made the default because a noisy validation set favoured it. `OCR_READ_WIDTH=880` remains available. Its speed on the Pi is measured by `scripts/pi_check_all.sh`; on this busy laptop a rendered image took 2.0 s at 880 against 2.6 s at 1,200 (full precision), which decides nothing.

## Not measured

- Any of this on the Raspberry Pi, and through the glass camera.
- The blended recogniser together with a smaller read width.
- Real signs photographed in Bangladesh: ICDAR MLT words are cropped single words from many countries.
