# Print-Disjoint Counterfeit Detection: A Dataset Flaw, a Watermark Solution, and a Safety Guarantee

*Workshop paper draft (4–8 pages), 2026-09-30. Every number comes from a saved result file cited in the text. Authors to be added.*

## Abstract

Counterfeit banknotes are produced in print runs: many notes share one printing plate and one serial number. We show that the public Bangladeshi counterfeit benchmark, JaalTaka, has this structure: 279 of its 322 readable counterfeit 500 Taka notes carry the same serial. A random note-disjoint split therefore places notes of the same print in both training and test.

- **The flaw.** A serial-number lookup alone scores 90.2 % on the standard test split. On a print-disjoint split, where no test counterfeit shares a print with training, accuracy falls from about 98 % to 89.9 % (prefix-trained multi-view network, three seeds), 92.0 / 89.9 % (fine-tuned ResNet-50, 1 / 6 views) and 85.1 % (frozen ResNet-50).
- **A watermark solution.** The watermark window is visible when a note is held against light; counterfeits carry only a blank, faint or printed imitation. We register the back-lit photo to a note template, crop the window, and score it with a 2.6 MB MobileNetV2. Combined with the network, it reaches **95.0 % on unseen prints** (six views, all three seeds; exact McNemar p ≤ 0.013 against the network), 5.1 points above a fine-tuned ResNet-50, with 3.3 % of genuine notes flagged and 7 of 101 counterfeits missed (vs 13–32 for the fine-tuned ResNet-50).
- **A safety guarantee.** On an assistive glass for blind users, the check never says "counterfeit": it says "likely genuine" above a validation-fixed threshold, otherwise "check by hand". An order-statistic argument bounds the pass rate of new counterfeits by 1/(n+1) in-domain; 0 of 88 test counterfeits and 0 of 25 real whole-note counterfeit photos were passed.

## 1. Introduction

Machine-learning counterfeit detectors are usually evaluated with random splits of a single dataset. Counterfeits, unlike genuine notes, come in batches from one plate. If notes of one batch appear in both training and test, a model can succeed by recognising the batch (its serial, its print defects), not by recognising what makes a note counterfeit. This is a group-leakage problem well known in other domains, where improper splitting inflates accuracy by 5–30 % (Scientific Data 2022).

We make three contributions.
1. **A measured instance of print leakage** in the only public Bangladeshi counterfeit benchmark, with a corrected print-disjoint protocol and an effective-sample-size result: the evidence scales with the number of prints, not notes (Theorem 16).
2. **A watermark-window detector** that transfers to unseen prints, where full-note classifiers do not.
3. **A deployable safety policy** with a finite-sample guarantee, running on an offline smart glass.

## 2. Related work

- **Bangladeshi counterfeit detection.** CNN classifiers evaluated on single random splits (modified AlexNet + SVM; reported 85–90 %). JaalTaka provides 1,390 notes × 6 region photos.
- **Transmitted light.** Transmitted-light imaging is used with dedicated infrared-transmission sensors (Sensors, multinational fitness classification). Smartphone-based fake detection has been proposed for visually impaired users.
- **Leakage.** Group-disjoint evaluation is recommended when samples share an identity (arXiv 2401.13796).
- **Our difference.** We isolate the watermark window as a separate registered input and evaluate on unseen prints.

Sources are listed in `LITERATURE_SEARCH.md`.

## 3. Method

**Print-disjoint split.** Counterfeit serials are read by OCR (EasyOCR, Bangla + English) from reconstructed whole notes. Counterfeits are grouped by the first six serial digits. The two largest prints (3274658 and 184383x) go to training; the remaining prints go whole to validation or test. The result is 946 / 222 / 222 notes, with 101 test counterfeits from prints never seen in training (effective number of prints 23.6).

**Network.** A multi-view network (MobileNetV3-Small + TinyViT encoders, attention-style fusion, shared batch-norm) trained on random view prefixes. This removes the collapse of fixed-view training at fewer views (71.2 % → 98.2 % at one view on the standard split).

**Watermark window.**
1. JaalTaka view 6 is photographed against light. It is registered to a whole-note front template (SIFT + RANSAC; 1,261 of 1,390 notes).
2. The window (0.70–0.93 × 0.33–0.85 of the note) is cropped below the serial, so no digits enter.
3. A MobileNetV2, fine-tuned on training-note windows, scores it. The model was chosen over MobileNetV3 by validation AUC (0.9957 vs 0.9943).

**Combination.** A logistic regression on the validation split combines the network logit, the watermark logit and a missing-window flag.

## 4. Experiments

- **Protocol.** Test data never trained or tuned anything. Seeds 42, 43, 44 for the network. Exact McNemar tests on the same notes.

**Baselines:**
- frozen ResNet-50 features + logistic regression (the strongest of five ImageNet backbones on the standard split);
- the network alone;
- serial blacklist.

## 5. Results

**Table 1.** Print-disjoint test, 222 notes (101 counterfeit from unseen prints). Sources: `results/watermark/hybrid_final_seeds.json`, `serial_split_eval.json`, `mobilenetv2.json`.

| Detector | 1 view | 6 views | Genuine called counterfeit (6 views) | Counterfeits missed (6 views) |
|---|---:|---:|---|---|
| Serial blacklist | catches 0 counterfeits | — | — | 101 / 101 |
| ResNet-50 probe | 85.1 % | 83.8 % | 1 / 121 | 35 / 101 |
| ResNet-50 fine-tuned (3 seeds) | 92.0 ± 2.3 % | 89.9 ± 4.4 % | 1 / 121 | 13–32 / 101 |
| Prefix network (3 seeds) | 89.9 ± 0.7 % | 89.3 ± 1.3 % | 1–3 / 121 | 19–26 / 101 |
| Watermark window alone (197 registered) | 92.9 % | — | 2 / 98 | 12 / 99 |
| **Network + watermark (3 seeds)** | **94.4 ± 0.5 %** | **95.0 ± 0.0 %** | **4 / 121 (3.3 %)** | **7 / 101** |

**Against a fine-tuned ResNet-50** the hybrid gains +2.4 / +5.1 points (1 / 6 views), significant in 3 of 6 per-seed comparisons (six views: p = 0.000027, 0.035, 0.51). Against the frozen ResNet-50, p ≤ 0.0003 on every seed (`BEAT_RESNET50_FINAL.md`).

**The dataset flaw, measured three ways:**
- the serial lookup's 90.2 % on the standard split;
- the note-disjoint vs print-disjoint drop (98.2 → 89.9 % for the network);
- five-fold cross-validation over all 1,390 notes: 98.3 % with note folds vs 95.8 % with serial-grouped folds at one view.

**Is the network reading serials?** Masking the serial region at test time lowers accuracy only from 97.1 % to 92.3 %, with genuine and counterfeit notes affected alike. So the standard-split accuracy is not mainly serial memorisation. The flaw is print similarity more broadly.

**Figures.**
- Fig. 19: standard vs print-disjoint.
- Fig. 16: unseen prints.
- Fig. 21: misses and false alarms.
- Fig. 24: serial leakage.
- Fig. 17: watermark windows.

## 6. Safety

**Policy.** For 500 and 1,000 Taka the glass says "সম্ভবত আসল" (likely genuine) only if p > τ, where τ is the highest score of any validation counterfeit. Otherwise it says "check by hand". It never says "counterfeit".

**Guarantee.** For exchangeable counterfeit scores, the pass probability is 1/(n+1), and P(pass rate > ε) = (1 − ε)^n. With n = 88: 1.1 % expected, and ≤ 5 % with 98.9 % confidence (Theorem 8). Under a deployment shift the bound grows by the total-variation distance (Theorem 9).

**Measured.**
- 0 / 88 JaalTaka test counterfeits passed.
- 0 / 25 real whole-note counterfeit photos passed.
- "Counterfeit" spoken 0 times in 1,889 runs of the app code.
- The policy confirms few whole-note genuine notes (0.4–2 %). It is safe, but rarely confirms a note.

## 7. Discussion

- **Print-disjoint evaluation should be the default for counterfeit benchmarks.** It changes the ranking: on the standard split the frozen probe ties the network; on unseen prints the network leads by about 5 points, and the watermark by about 10.
- **Physical security features generalise across prints better than whole-note appearance.** The watermark is a physical feature that a counterfeiter's plate imitates poorly.

## 8. Limitations

- **Small print count.** About 24 effective prints in the print-disjoint test (print-level uncertainty ±0.28 at 95 % by Theorem 16). Absolute numbers are estimates; paired comparisons are more reliable.
- **Watermark needs back-lighting.** The glass must ask the user to hold the note to light, which is not yet tested with the glass camera.
- **Pen-mark confound.** Circulated genuine notes can carry pen marks in the window.
- **Serial OCR errors** can mis-group a few notes.
- **One authentication dataset.**

## 9. Conclusion

A counterfeit detector should be judged on counterfeit prints it has never seen. On JaalTaka, doing so removes about 8–14 points of apparent accuracy. A watermark-window check recovers most of it. A policy that never asserts "counterfeit" makes the result safe to deploy on an assistive device.
