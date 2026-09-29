# Deployed jaal verdict (2026-09-29)

## Status now: safe policy E is on for 500 and 1,000 Taka

The glass never says "জাল টাকা" (counterfeit). For a detected 500 or 1,000 Taka note, it says "সম্ভবত আসল" (likely genuine) only when p(genuine) > τ = 0.9995918869972229. Otherwise it says "আসল কিনা হাতে যাচাই করুন" (check by hand). Other denominations keep "জাল যাচাই করা হয়নি" (check not done): JaalTaka has only 500 and 1,000 Taka notes.

**Pipeline.** Taka YOLO box → crop resized to 640 px wide and turned landscape → four windows cut at the positions where JaalTaka views 1–4 lie on a note → PRMVT (`results/qduig/prefix_ft/seed42/checkpoint.pt`) on those four views. The code is `savior_glass/modes/currency_mode.py` (`_safe_check`); the settings are in `savior_glass/config.py` (`JAAL_SAFE_*`, `JAAL_VIEW_WINDOWS`).

### Headline

| Measure | Result | 95 % Wilson CI | Data |
|---|---:|---|---|
| Genuine notes told "counterfeit" | **0** (the policy cannot say it) | — | all sets |
| Counterfeit passed as "likely genuine", JaalTaka test, real views 1–4 | **0 / 88** | 0 – 4.2 % | 88 physical notes |
| Counterfeit passed, whole-note photos, app code path | **0 / 20** checked | 0 – 16.1 % | 4 physical-note groups |
| Counterfeit passed, whole-note photos, experiment script | **0 / 25** | 0 – 13.3 % | same groups |
| Genuine confirmed as "likely genuine", JaalTaka test | 69 / 120 (57.5 %) | | in-domain |
| Genuine confirmed, whole notes, app path | cf-set 192 / 1,138 (16.9 %); Bangla Money 1 / 277 (0.4 %); BanglaTaka 6 / 296 (2.0 %) | | |

**Margin.** The highest whole-note counterfeit score is p = 0.99567, from one 1,000 BDT burst (`cf1000_counterfeit_burst000`), against τ = 0.99959. That is 0.004 in probability and 2.4 in log-odds (5.44 against 7.80). The augmented copies peak at 0.98137. "0 passed" is a narrow result on whole notes, not a wide one.

**Reading.** The policy is safe on everything measured, and meets the < 5 % target for counterfeit passed in-domain (upper bound 4.2 %). On whole-note photographs it is safe but seldom useful: nearly every note is sent to "check by hand". The whole-note safety evidence rests on about four physical counterfeit notes (group-level 95 % upper bound about 49 %). It is supporting evidence, not proof.

### How it was chosen (no test data used)

1. **View windows**, measured on 200 JaalTaka TRAIN notes by SIFT + RANSAC registration to a whole-note template (`scripts/eval/jaal_view_geometry.py` → `results/jaal_whole/view_geometry.json`). Median x ranges: view 1 [0, 0.476], view 2 [0.284, 0.826], view 3 [0.572, 1.0], view 4 [0.509, 1.0], full height. Every interquartile range is within ±0.03.
2. **Rules** written before any whole-note score was read (`results/jaal_whole/PREREGISTERED_OPERATING_POINTS.md`). τ is the highest p(genuine) of any JaalTaka VAL counterfeit note, and "likely genuine" requires p > τ. The checker is the one confirming the most VAL genuine notes at its τ; that chose S4 (PRMVT, views 1–4): 66 / 120 VAL genuine confirmed, 0 / 88 VAL counterfeit passed.
3. **Scored once**: JaalTaka TEST and the whole-note photographs (`scripts/eval/jaal_policy.py` → `results/jaal_whole/policy.json`), then through the app (`scripts/eval/eval_jaal_safe_app.py` → `results/jaal_whole/app_check.json`).

### The five approaches that were tried

Whole-note photographs, 500 / 1,000 BDT (`scripts/eval/jaal_whole_note.py` → `results/jaal_whole/summary.json`). Rates are genuine photos called counterfeit at p = 0.5. AUC is on the counterfeit set's original photographs (1,194 genuine, 25 counterfeit).

| Approach | Checker | cf-set genuine | Bangla Money | BanglaTaka | AUC | Outcome |
|---|---|---:|---:|---:|---:|---|
| (deployed before) | S0: whole crop as 1 view | 27.1 % | 45.1 % | 61.5 % | 0.640 | replaced |
| A (as specified): synthetic whole notes on COCO, fine-tune PRMVT | fine-tuned PRMVT, 4 cut views | 3.5 % | 34.2 % | 45.0 % | 0.689 | passes 14 / 20 real counterfeits at its validation threshold; not deployed |
| A (re-scoped): build JaalTaka-like input from whole notes | S3: 3 cut views | 1.1 % | 20.5 % | 30.8 % | 0.819 | better ranking; too many false "counterfeit" to speak |
| A | S4: 4 cut views | 5.6 % | 43.4 % | 60.2 % | 0.829 | used inside E |
| D: feature-level (frozen ResNet-50 probe on cut views) | R2 | 1.6 % | 12.5 % | 16.5 % | 0.966 | best ranking; at 0.5 it passes 59 % of counterfeit images |
| B: train on the whole-note counterfeit set (leave one counterfeit group out) | B | 0.1 % | 26.4 % | 86.7 % | 0.886 | fits its own set, fails on independent genuine notes |
| C: two-model agreement | — | | | | | not needed: E passes no counterfeit with one model, and every added model only adds a way to pass one |
| **E: never say "counterfeit"; "likely genuine" above τ** | **S4, τ = 0.99959** | **0 %** by construction | **0 %** | **0 %** | — | **shipped** |

**Approach A as specified, run 2026-09-29 (evening).**
- **Data.** All 1,390 JaalTaka notes were rebuilt as whole notes by registering views 1–4 to a template (SIFT + RANSAC); 1,383 were kept. Gaps were filled by inpainting from the note's own pixels (`scripts/eval/synth_whole_notes.py`).
- **Training.** PRMVT was fine-tuned on composites of the training notes: random COCO background, perspective, lighting, blur, JPEG and occluding patch, then the glass's four cut views (`scripts/train/train_whole_note_synth.py`). Validation accuracy on synthetic composites rose from 0.7356 to 0.8269.
- **Rules.** Fixed beforehand in `results/jaal_whole/approach_a/PREREGISTERED.md`. Results are in `results/jaal_whole/approach_a/eval.json`.

| | Approach A | Deployed S4 |
|---|---:|---:|
| Synthetic TEST accuracy / AUC (205 notes) | 0.873 / 0.933 | 0.790 / 0.875 |
| Synthetic TEST counterfeit passed | 6 / 87 at τ_A = 0.683 | 0 / 87 at τ = 0.99959 |
| Real whole-note AUC (counterfeit-set originals) | **0.689** | **0.810** |
| Real counterfeit originals passed as "likely genuine" | **14 / 20** | **0 / 20** |
| Independent genuine confirmed (Bangla Money + BanglaTaka) | 729 | 19 |

**Decision: Approach A is not deployed**, under the pre-registered rule, because it passes real counterfeits. It learns the synthetic whole notes better, but real photographs worse. The synthetic-to-real gap is the same kind of gap that broke the original close-up checker. It confirms many more genuine notes only because its threshold, fitted on synthetic data, is far lower.

**Why "A" was first re-scoped.** Generating counterfeit whole-note images from JaalTaka close-ups was tried as stitching views 1–4 (OpenCV panorama). Only some notes stitched cleanly: the left edge was often lost, and one note failed completely. Such images would carry stitching artifacts that no camera photo has. Instead, the deploy input is turned into JaalTaka's shape (cut views), which needs no synthetic images.

**Why a whole-note classifier cannot be certified on the available data.** The counterfeit set has 87 counterfeit images: 60 are augmented copies, and the originals come from about four physical notes. A logistic regression on file metadata only (log width, log height, aspect, PNG flag, bytes per pixel), trained leaving one counterfeit group out, catches 76 % of counterfeit images and flags 1.5 % of genuine ones. The set is separable without looking at the note. The 500 BDT counterfeit series shares its serial number, ছক ৩২৭৪৬৫৮, with JaalTaka counterfeit notes, so it may be the same print batch.

**Why the looser threshold is not used.** At the 99th percentile of VAL counterfeit scores (1 / 88 VAL counterfeit passed), S4 passes 16 of 25 whole-note counterfeit photographs. A threshold calibrated on close-ups does not transfer to whole notes.

### What would make the feature more useful

Grouped whole-note photographs of known counterfeit notes, taken with the glass camera. They would allow a threshold fitted on whole notes and a test that is not four notes deep. The pipeline and scripts above run unchanged on such data.

### Checks after the change

- `savior_glass/scripts/test_buttons_haptics.py` recognises both new sentences (one detect pulse each).
- The app-level run spoke no "জাল টাকা" in 1,889 photos (`app_check.json`, `spoken_jaal: 0`).

---

# Earlier record: deployed verdict measured, then switched off (2026-09-29, first pass)

## What the glass did

`savior_glass/modes/currency_mode.py` ran the Taka YOLO detector, cropped the top box, and passed that crop to PRMVT (`results/qduig/prefix_ft/seed42/checkpoint.pt`) as **one view**. It said "আসল" (genuine) if p(genuine) ≥ 0.5, otherwise "জাল টাকা" (counterfeit). There was no abstain.

PRMVT was trained and tested only on JaalTaka close-up views: 128×128 crops of note regions such as the portrait and the hologram strip. The app gave it whole-note crops from a camera frame, and that input had never been evaluated.

## Measurement

Script: `realtime_bangla_taka_detection/scripts/eval/eval_jaal_deployed.py`. It runs the app's own `CurrencyMode.detect_live` with the verdict forced on.

Raw data:
- `realtime_bangla_taka_detection/results/jaal_deployed/rows.json` (one row per image)
- `summary.json`

No image was used to train or select PRMVT.

| Set | Truth | Images | Judged | Wrong verdict | 95 % Wilson CI |
|---|---|---:|---:|---:|---|
| Bangla Money (Kaggle), 8 denominations | genuine | 1,536 | 1,464 | **699 (47.7 %)** | 45.2–50.3 % |
| BanglaTaka raw, 50/class, seed 42 | genuine | 450 | 441 | **261 (59.2 %)** | 54.5–63.7 % |
| Bangladeshi Counterfeit Currency Image Dataset, 500/1000 BDT | genuine | 1,199 | 1,187 | **242 (20.4 %)** | 18.2–22.8 % |
| same | counterfeit | 87 | 75 | **30 (40.0 %)** | 29.7–51.3 % |

"Judged" excludes images where no note was found or p was not finite.

**Ranking signal.** On the counterfeit dataset, which has both classes, the ROC-AUC of p(genuine) is 0.751 (n = 1,269).

**Probability spread.** Median p(genuine) for genuine notes is 0.527 on Bangla Money, 0.383 on BanglaTaka and 0.955 on the counterfeit set. The spread across sets is wider than the gap between classes, so no single threshold separates them.

**Counterfeit images.** The 32 counterfeit 500 BDT images are `augmented_*.png`: the dataset's own augmented copies of a few partial notes. So the 75 judged counterfeit images are not 75 independent notes, and the 40 % figure has a wider real uncertainty than its Wilson interval shows.

**Photo grouping.** Photos are not grouped by physical note. The counterfeit set contains burst shots, and it has only 87 counterfeit images. Its rates therefore describe images, not independent notes.

## Diagnosis

The failure comes from an input mismatch, not a bug:

- JaalTaka views show one region of a note at close range.
- The glass shows the whole note, small, against a background.

The verdict tracks the photo collection (median p from 0.38 to 0.96 on genuine notes) more than authenticity. The dataset-level shift is larger than the class signal.

## Fix applied

1. **`savior_glass/config.py`.** Adds `JAAL_VERDICT_ENABLED = False`.
2. **`savior_glass/modes/currency_mode.py`.** When the verdict is off:
   - the authenticator is not loaded or run
   - `hit["auth"]` stays `"unknown"`
   - the spoken sentence ends with "জাল যাচাই করা হয়নি" ("counterfeit check not done"), so silence cannot be read as "genuine"
   - the haptic plays the plain detect pulse

   The module docstring now describes the YOLO stage 0.
3. **`savior_glass/test_windows.py`.** English fallback "Authenticity not checked".
4. **`savior_glass/scripts/test_buttons_haptics.py`.** Recognises the new message. Rerun result: all checks pass. Press-to-speech is 366 ms (989 ms with the authenticator; laptop, simulated GPIO) (`savior_glass/results/buttons_haptics_test.json`).

## Target status

The target was a wrong-verdict rate below 5 % on genuine photos. **Not met by any verdict-speaking configuration measured.** With the verdict off, the glass gives no wrong verdicts because it gives none.

## What would re-enable it

A checker trained on whole-note crops, validated on photographs grouped by physical note, with a rejection threshold chosen on validation.

The only whole-note counterfeit data on disk is 87 images of 500/1000 BDT with no note IDs. That is too small to train and test such a checker without leakage. **Collecting grouped whole-note photos of known counterfeit notes is the prerequisite.**
