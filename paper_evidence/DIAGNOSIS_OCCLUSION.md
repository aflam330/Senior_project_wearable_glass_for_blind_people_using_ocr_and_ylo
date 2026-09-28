# Diagnosis: occlusion 0.55

Target is 6-view test accuracy 0.90. The best measured score is 0.875.

## What was loaded

Checkpoint `results/qduig/occlusion_ft/seed42/checkpoint.pt`. Test n=208. The occlusion draw is `np.random.seed(1000 + start + i)` once per note, then a black box of area fraction 0.55 on every view, then resize to 128.

## Measurements

| setting | 6-view test accuracy | source |
| --- | ---: | --- |
| black box | 0.6682692307692307 | `results/robustness/occlusion_ft_seed42.json` |
| median fill | 0.875 | `results/robustness/occlusion_repairs_seed42.json` |
| TELEA inpaint on this checkpoint | 0.6057692307692307 | same file |
| full frame after median fill | 0.875 | `results/robustness/occlusion_tta_seed42.json` |
| mean of five crops | 0.8653846153846154 | same file |
| most confident crop | 0.8557692307692307 | same file |

Validation before any new update: clean 0.9375, median-fill occlusion 0.8413461538461539. Validation has 120 genuine notes out of 208.

## Loss and predictions

A full pass at learning rate 0.0003, a full pass at 0.00001, and a full pass with batch-norm frozen all finished at validation accuracy 0.5769230769230769 on both clean and occluded images. That is exactly 120/208. The model predicted the genuine class for every note.

The 40-note run shows the mechanism. Losses on the first nine batches fell from 0.7579418420791626 to 0.23342555264631906. The tenth batch produced a non-finite gradient and wrote non-finite weights into the ViT (`encoder.vit.cls_token`, patch embedding, and attention projections). After that, logits are replaced by zeros and every note is called genuine. Skipping that batch leaves clean validation at 0.9423076923076923 and occluded validation unchanged at 0.8413461538461539. The checkpoint was not kept. It is byte-identical to the occlusion fine-tune.

Gradient norms per epoch, feature drift, and a full batch-norm statistic dump were not logged by the earlier trainer. The tenth-batch gradient was measured directly: it was not finite. A 20-step probe before that batch moved weights by at most 5.017966032028198e-05 and left batch-norm running statistics unchanged.

## Root cause

This is information loss, not a class-balance bug in the labels. Area 0.55 removes the same fraction of ink from every view. Median fill restores a flat color, not the print, and still gains 0.20673076923076928 over the black box because a black rectangle is a stronger out-of-distribution cue than a flat color. Classical inpainting and crops do not recover the missing print. Continuing Adam on that input for a full pass hits a non-finite ViT step and the predictor collapses to the majority class.

A learned reconstructor and an uncertainty rejection rule were not measured in the runs above.

---

## Update 2026-09-28: root cause

Measured on the **validation** split, 208 notes, 55% black box per view, the same seeding rule as the test protocol. Script: `realtime_bangla_taka_detection/scripts/eval/diagnose_occlusion.py`; raw: `results/robustness/occlusion_diagnosis_val.json`.

| Model / condition | Accuracy | Genuine notes called genuine | Counterfeit notes called counterfeit | Mean P(genuine) |
|---|---:|---:|---:|---:|
| PRMVT clean, 6 views | 98.6% | 99% | 98% | 0.58 |
| PRMVT 55% black box, 6 views | 52.4% | **23%** | 93% | 0.25 |
| PRMVT 55% black box, 1 view | 50.0% | 16% | 97% | 0.20 |
| PRMVT 55% box + median fill, 6 views | 54.3% | 21% | 100% | 0.20 |
| Baseline 55% black box, 6 views | 85.1% | 88% | 82% | 0.53 |

1. **The failure is one-sided: occluded genuine notes are called counterfeit.** PRMVT had learned that a note with a region that looks wrong or missing is a fake, and a black box looks like a missing security feature. Counterfeit notes are still caught.
2. **Visibility does not explain which notes fail.** Wrong and right notes kept the same visible area (45.0% vs 44.9% per view).
3. **Median fill does not help** (54.3%). Filling the box with the note's median colour still reads as "missing feature".
4. **Why earlier fixes failed:**
   - `occlusion_ft`: 2 epochs at lr 3×10⁻⁴, occlusion in half the batches.
   - `occlusion_matched`: stopped after 40 training notes, and its gradients went non-finite. That was the NaN-entropy bug fixed on 2026-09-28 (see `WEAK_RESULTS_FIX.md`).
   - In every run the ImageNet MobileNet stayed frozen. It supplies 576 of the 704 feature dimensions per view and sees the black box first.

**Fix implied:** train on occluded genuine *and* counterfeit notes long enough that "covered" stops meaning "counterfeit". Let the last CNN blocks adapt, at a low learning rate with BatchNorm statistics frozen. See `OCCLUSION_FINETUNE.md`.
