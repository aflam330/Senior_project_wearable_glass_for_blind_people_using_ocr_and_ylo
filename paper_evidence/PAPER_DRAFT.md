# Prefix-Robust Multi-View Learning: Fixing View-Count Distribution Shift for Currency Authentication

Draft assembled 2026-09-26 03:28 UTC from artifacts. A number that is not in a cited file is not in this draft.

## 1. Introduction

A classifier trained to see all six views of a JaalTaka note is not a classifier for one view. On the seed-42 test split, the earlier fixed-count Q-DUIG checkpoint scores 0.9663 with six views and 0.5865 with one view (`FINAL_RESULTS.md`). That drop is view-count distribution shift: training puts all mass on six views, and test prefixes are shorter. Prefix-robust training samples those shorter prefixes. The seed-42 prefix-robust checkpoint (PRMVT) scores 0.9712 at one view and 0.9760 at six views.

Fifteen other training variants were run on the same split. NDAL v2 is the highest one-view score in the saved v2 files. SFPL stays below the CNN+ViT baseline from two views upward. OGPD v2 was worse than OGPD v1, so v1 is the number that stands. Pi 5 and Android measurements are NOT_MEASURED.

## 2. Related work

Labels and citations are in `NOVELTY_DECLARATION.md`. PRMVT extends missing-view training (RMAE; dual-masked VAEs, IJCAI 2025; RML, ICCV 2025). It does not claim to be the first variable-view model. NDAL adapts focal loss (Lin et al., ICCV 2017). PRAVT reverses view order; it is not PGD (RDML, IJCAI 2025). Phone-camera banknote authentication under visible light is prior work (Sensors 2019). Scopus, IEEE Xplore, ACM DL, and Web of Science were not searched as separate databases.

## 3. Method

The shared encoder is a frozen MobileNetV3-Small plus TinyViT. Images are 128 pixels on a side. The label is genuine or counterfeit. The decision threshold is 0.5.

PRMVT is the Q-DUIG prefix protocol: six epochs of mixed view dropout (`configs/proposed_prefix.yaml`), then three epochs of fine-tuning (`configs/proposed_prefix_ft.yaml`). The checkpoint is the best mean validation accuracy over one to six views.

The other algorithms are heads or losses on that encoder, documented in `NOVELTY_DECLARATION.md`. CVS is a leave-one-view logit KL, not a fitted causal graph. SFPL is FedAvg inside one process. SAVS is one sharpness-aware step. SFAQ has no hologram or thread labels.

## 4. Theory

Definitions and proofs are in `THEORETICAL_ANALYSIS.md`. The support proposition says that a training distribution with positive probability on every view count does not have the fixed-N support mismatch. It is not a convergence rate. The PAC-Bayes display is the McAllester inequality. For a Gaussian posterior on the saved prefix-robust weights, the smallest penalty on the fixed prior grid is 57.5893, so the bound is vacuous (`results/theory/pacbayes_prmvt_seed42.json`).

## 5. Experiments

JaalTaka has 1390 notes (802 genuine, 588 counterfeit), six views, and a note-disjoint split of 974 / 208 / 208 at seed 42. Leakage between splits is zero (`results/camva/splits/split_metadata.json`). The test split is not used for training, checkpoint selection, or threshold selection.

Seeds 43 and 44 repeat the same configs for PRMVT, NDAL, PRAVT, VAT, UGF, and CVS. Early stopping can stop a seed before the maximum epoch. That is the config, and it is reported when the logs differ.

## 6. Results

Seed 42, test, n=208. Sources are the `test_metrics.json` paths in `scripts/eval/write_astar_paper.py` and `results/qduig/prefix_ft/seed42/test_views/views_1_to_6.json`.

| algorithm | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.7356 | 0.8702 | 0.9135 | 0.9183 | 0.8990 | 0.9183 |
| prmvt | 0.9712 | 0.9760 | 0.9760 | 0.9760 | 0.9808 | 0.9760 |
| ndal | 0.9808 | 0.9712 | 0.9760 | 0.9663 | 0.9567 | 0.9663 |
| pravt | 0.9760 | 0.9519 | 0.9663 | 0.9567 | 0.9663 | 0.9615 |
| vat | 0.9712 | 0.9712 | 0.9760 | 0.9663 | 0.9712 | 0.9663 |
| ugf | 0.9471 | 0.9519 | 0.9567 | 0.9519 | 0.9375 | 0.9375 |
| cvs | 0.9327 | 0.9567 | 0.9663 | 0.9712 | 0.9712 | 0.9712 |
| apc | 0.9615 | 0.9808 | 0.9856 | 0.9712 | 0.9615 | 0.9663 |
| cris | 0.9519 | 0.9279 | 0.9231 | 0.9279 | 0.9038 | 0.9183 |
| savs | 0.9375 | 0.9279 | 0.9327 | 0.9327 | 0.9279 | 0.9327 |
| mavt | 0.9183 | 0.9519 | 0.9663 | 0.9567 | 0.9519 | 0.9567 |
| vcie | 0.8798 | 0.8750 | 0.9087 | 0.8990 | 0.9038 | 0.9135 |
| mtpt | 0.9519 | 0.9231 | 0.9375 | 0.9327 | 0.9231 | 0.9135 |
| sfpl | 0.7404 | 0.8510 | 0.8606 | 0.8462 | 0.8462 | 0.8317 |
| ogpd | 0.9231 | 0.9231 | 0.9279 | 0.9135 | 0.9087 | 0.9183 |
| sfaq | 0.9471 | 0.9423 | 0.9663 | 0.9375 | 0.9375 | 0.9423 |
| igcr | 0.9471 | 0.9423 | 0.9519 | 0.9375 | 0.9279 | 0.9327 |

OGPD in that table is v1. The v2 OGPD file is lower (0.7596 at one view through 0.7933 at six views) and is not the standing OGPD result.

Multi-seed mean, standard deviation, and the normal approximation interval:

# Multi-seed results

Seeds 42, 43, and 44. A missing seed is NOT_MEASURED. The interval is a normal approximation from the seed-level accuracies, not a paired bootstrap of notes.

| algorithm | views | n seeds | mean | sd | approx 95% CI |
|---|---:|---:|---:|---:|---|
| prmvt | 1 | 3 | 0.9647 | 0.0155 | [0.9473, 0.9822] |
| prmvt | 2 | 3 | 0.9824 | 0.0073 | [0.9741, 0.9907] |
| prmvt | 3 | 3 | 0.9792 | 0.0056 | [0.9729, 0.9854] |
| prmvt | 4 | 3 | 0.9744 | 0.0073 | [0.9660, 0.9827] |
| prmvt | 5 | 3 | 0.9840 | 0.0056 | [0.9777, 0.9903] |
| prmvt | 6 | 3 | 0.9792 | 0.0100 | [0.9678, 0.9905] |
| ndal | 1 | 3 | 0.9696 | 0.0100 | [0.9582, 0.9809] |
| ndal | 2 | 3 | 0.9647 | 0.0111 | [0.9522, 0.9773] |
| ndal | 3 | 3 | 0.9728 | 0.0100 | [0.9614, 0.9841] |
| ndal | 4 | 3 | 0.9567 | 0.0127 | [0.9423, 0.9711] |
| ndal | 5 | 3 | 0.9503 | 0.0111 | [0.9378, 0.9629] |
| ndal | 6 | 3 | 0.9583 | 0.0139 | [0.9426, 0.9740] |
| pravt | 1 | 3 | 0.9615 | 0.0127 | [0.9471, 0.9759] |
| pravt | 2 | 3 | 0.9615 | 0.0083 | [0.9521, 0.9710] |
| pravt | 3 | 3 | 0.9744 | 0.0073 | [0.9660, 0.9827] |
| pravt | 4 | 3 | 0.9631 | 0.0073 | [0.9548, 0.9715] |
| pravt | 5 | 3 | 0.9679 | 0.0121 | [0.9543, 0.9816] |
| pravt | 6 | 3 | 0.9647 | 0.0147 | [0.9481, 0.9814] |
| vat | 1 | 3 | 0.9728 | 0.0073 | [0.9644, 0.9811] |
| vat | 2 | 3 | 0.9728 | 0.0073 | [0.9644, 0.9811] |
| vat | 3 | 3 | 0.9792 | 0.0056 | [0.9729, 0.9854] |
| vat | 4 | 3 | 0.9679 | 0.0028 | [0.9648, 0.9711] |
| vat | 5 | 3 | 0.9679 | 0.0056 | [0.9617, 0.9742] |
| vat | 6 | 3 | 0.9631 | 0.0028 | [0.9600, 0.9663] |
| ugf | 1 | 3 | 0.9167 | 0.0486 | [0.8616, 0.9717] |
| ugf | 2 | 3 | 0.9455 | 0.0111 | [0.9329, 0.9581] |
| ugf | 3 | 3 | 0.9567 | 0.0000 | [0.9567, 0.9567] |
| ugf | 4 | 3 | 0.9535 | 0.0073 | [0.9452, 0.9618] |
| ugf | 5 | 3 | 0.9471 | 0.0096 | [0.9362, 0.9580] |
| ugf | 6 | 3 | 0.9503 | 0.0182 | [0.9297, 0.9709] |
| cvs | 1 | 3 | 0.9535 | 0.0194 | [0.9315, 0.9755] |
| cvs | 2 | 3 | 0.9599 | 0.0056 | [0.9537, 0.9662] |
| cvs | 3 | 3 | 0.9760 | 0.0096 | [0.9651, 0.9868] |
| cvs | 4 | 3 | 0.9712 | 0.0096 | [0.9603, 0.9820] |
| cvs | 5 | 3 | 0.9744 | 0.0056 | [0.9681, 0.9806] |
| cvs | 6 | 3 | 0.9679 | 0.0056 | [0.9617, 0.9742] |

The interval above uses the seed-level accuracies. Per-note paired bootstrap intervals are in `STATISTICAL_ANALYSIS.md`.



Paired tests against the CNN+ViT baseline on the same 208 notes:

# Statistical analysis

Cohen's h compares each available seed-42 accuracy with the CNN+ViT baseline at the same view count.
Wilcoxon and McNemar on three seed-level numbers are not reported: n=3 is too small for a rank test to mean anything.
Per-note McNemar for PRMVT versus the baseline at 6 views remains the seed-42 result already in `FINAL_RESULTS.md` (n01=3, n10=13, p=0.0244).
Bonferroni uses one family: 6 algorithms × 6 view counts = 36 Cohen's h comparisons. Those comparisons are descriptive; they are not 36 independent hypothesis tests with a pre-registered null.

| algorithm | views | seed42 | baseline | Cohen h |
|---|---:|---:|---:|---:|
| prmvt | 1 | 0.9712 | 0.7356 | 0.7389 |
| prmvt | 2 | 0.9760 | 0.8702 | 0.4258 |
| prmvt | 3 | 0.9760 | 0.9135 | 0.2858 |
| prmvt | 4 | 0.9760 | 0.9183 | 0.2685 |
| prmvt | 5 | 0.9808 | 0.8990 | 0.3685 |
| prmvt | 6 | 0.9760 | 0.9183 | 0.2685 |
| ndal | 1 | 0.9808 | 0.7356 | 0.8020 |
| ndal | 2 | 0.9712 | 0.8702 | 0.3958 |
| ndal | 3 | 0.9760 | 0.9135 | 0.2858 |
| ndal | 4 | 0.9663 | 0.9183 | 0.2109 |
| ndal | 5 | 0.9567 | 0.8990 | 0.2276 |
| ndal | 6 | 0.9663 | 0.9183 | 0.2109 |
| pravt | 1 | 0.9760 | 0.7356 | 0.7689 |
| pravt | 2 | 0.9519 | 0.8702 | 0.2950 |
| pravt | 3 | 0.9663 | 0.9135 | 0.2282 |
| pravt | 4 | 0.9567 | 0.9183 | 0.1608 |
| pravt | 5 | 0.9663 | 0.8990 | 0.2777 |
| pravt | 6 | 0.9615 | 0.9183 | 0.1851 |
| vat | 1 | 0.9712 | 0.7356 | 0.7389 |
| vat | 2 | 0.9712 | 0.8702 | 0.3958 |
| vat | 3 | 0.9760 | 0.9135 | 0.2858 |
| vat | 4 | 0.9663 | 0.9183 | 0.2109 |
| vat | 5 | 0.9712 | 0.8990 | 0.3054 |
| vat | 6 | 0.9663 | 0.9183 | 0.2109 |
| ugf | 1 | 0.9471 | 0.7356 | 0.6161 |
| ugf | 2 | 0.9519 | 0.8702 | 0.2950 |
| ugf | 3 | 0.9567 | 0.9135 | 0.1781 |
| ugf | 4 | 0.9519 | 0.9183 | 0.1377 |
| ugf | 5 | 0.9375 | 0.8990 | 0.1413 |
| ugf | 6 | 0.9375 | 0.9183 | 0.0745 |
| cvs | 1 | 0.9327 | 0.7356 | 0.5553 |
| cvs | 2 | 0.9567 | 0.8702 | 0.3181 |
| cvs | 3 | 0.9663 | 0.9135 | 0.2282 |
| cvs | 4 | 0.9712 | 0.9183 | 0.2385 |
| cvs | 5 | 0.9712 | 0.8990 | 0.3054 |
| cvs | 6 | 0.9712 | 0.9183 | 0.2385 |

Paired tests use the same test notes. `a` is the CNN+ViT baseline and `b` is the named model. McNemar uses the continuity correction. The paired bootstrap resamples notes 1000 times with seed 42. Wilcoxon is on the paired 0/1 correctness vectors and drops ties.

| algorithm | views | n | McNemar p | Bonferroni p | bootstrap 95% CI of accuracy difference | Wilcoxon p |
|---|---:|---:|---:|---:|---|---:|
| prmvt | 1 | 208 | 4.30e-11 | 1.55e-09 | [0.1779, 0.2981] | 1.75e-08 |
| prmvt | 2 | 208 | 1.81e-05 | 0.0007 | [0.0625, 0.1538] | 2.35e-05 |
| prmvt | 3 | 208 | 0.0019 | 0.0700 | [0.0288, 0.1010] | 0.0090 |
| prmvt | 4 | 208 | 0.0060 | 0.2145 | [0.0240, 0.0963] | 0.0340 |
| prmvt | 5 | 208 | 0.0002 | 0.0087 | [0.0433, 0.1203] | 0.0002 |
| prmvt | 6 | 208 | 0.0033 | 0.1182 | [0.0240, 0.0962] | 0.0157 |
| ndal | 1 | 208 | 6.51e-12 | 2.34e-10 | [0.1875, 0.3077] | 4.49e-09 |
| ndal | 2 | 208 | 6.33e-05 | 0.0023 | [0.0529, 0.1490] | 0.0003 |
| ndal | 3 | 208 | 0.0059 | 0.2126 | [0.0240, 0.1011] | 0.0038 |
| ndal | 4 | 208 | 0.0162 | 0.5816 | [0.0144, 0.0865] | 0.0219 |
| ndal | 5 | 208 | 0.0190 | 0.6846 | [0.0144, 0.1011] | 0.0156 |
| ndal | 6 | 208 | 0.0162 | 0.5816 | [0.0144, 0.0817] | 0.1094 |
| pravt | 1 | 208 | 2.59e-11 | 9.33e-10 | [0.1779, 0.3029] | 2.35e-08 |
| pravt | 2 | 208 | 0.0002 | 0.0087 | [0.0433, 0.1202] | 0.0005 |
| pravt | 3 | 208 | 0.0055 | 0.1996 | [0.0192, 0.0865] | 0.0192 |
| pravt | 4 | 208 | 0.0269 | 0.9668 | [0.0096, 0.0721] | 0.0593 |
| pravt | 5 | 208 | 0.0012 | 0.0415 | [0.0288, 0.1106] | 0.0052 |
| pravt | 6 | 208 | 0.0159 | 0.5710 | [0.0144, 0.0769] | 0.0505 |
| vat | 1 | 208 | 1.80e-11 | 6.48e-10 | [0.1779, 0.2981] | 9.66e-09 |
| vat | 2 | 208 | 0.0001 | 0.0043 | [0.0529, 0.1538] | 5.43e-05 |
| vat | 3 | 208 | 0.0036 | 0.1299 | [0.0240, 0.1010] | 0.0065 |
| vat | 4 | 208 | 0.0162 | 0.5816 | [0.0144, 0.0865] | 0.0962 |
| vat | 5 | 208 | 0.0013 | 0.0475 | [0.0337, 0.1154] | 0.0033 |
| vat | 6 | 208 | 0.0162 | 0.5816 | [0.0144, 0.0865] | 0.1094 |
| ugf | 1 | 208 | 5.42e-10 | 1.95e-08 | [0.1538, 0.2692] | 4.27e-07 |
| ugf | 2 | 208 | 0.0002 | 0.0087 | [0.0433, 0.1250] | 0.0005 |
| ugf | 3 | 208 | 0.0265 | 0.9540 | [0.0143, 0.0769] | 0.0277 |
| ugf | 4 | 208 | 0.0455 | 1.0000 | [0.0048, 0.0625] | 0.1097 |
| ugf | 5 | 208 | 0.0269 | 0.9668 | [0.0096, 0.0721] | 0.0745 |
| ugf | 6 | 208 | 0.3428 | 1.0000 | [-0.0096, 0.0481] | 0.6465 |
| cvs | 1 | 208 | 2.13e-08 | 7.67e-07 | [0.1346, 0.2596] | 7.82e-07 |
| cvs | 2 | 208 | 0.0003 | 0.0104 | [0.0480, 0.1346] | 0.0022 |
| cvs | 3 | 208 | 0.0098 | 0.3536 | [0.0192, 0.0913] | 0.0022 |
| cvs | 4 | 208 | 0.0026 | 0.0925 | [0.0240, 0.0865] | 0.0033 |
| cvs | 5 | 208 | 0.0003 | 0.0108 | [0.0385, 0.1106] | 0.0007 |
| cvs | 6 | 208 | 0.0026 | 0.0925 | [0.0240, 0.0865] | 0.0033 |

Bonferroni family size is the number of McNemar rows that had paired predictions: 36.



## 7. Ablation

Baseline one-to-six view accuracy is the CNN+ViT row above. It was not retrained. The other rows are Q-DUIG component configs at seed 42.

| config | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| quality | 0.5625 | 0.5529 | 0.7500 | 0.8606 | 0.8654 | 0.9519 |
| uncertainty | 0.6971 | 0.7019 | 0.7115 | 0.7356 | 0.7452 | 0.8173 |
| diversity | 0.7019 | 0.8654 | 0.9183 | 0.9135 | 0.9135 | 0.8942 |
| qd | 0.6587 | 0.4615 | 0.6010 | 0.7019 | 0.6971 | 0.8269 |
| ud | 0.7596 | 0.8942 | 0.9038 | 0.9231 | 0.9231 | 0.9567 |
| full | 0.6298 | 0.6394 | 0.6490 | 0.7212 | 0.7067 | 0.7404 |
| no_cost | 0.6298 | 0.6394 | 0.6490 | 0.7212 | 0.7067 | 0.7404 |
| no_calibration | 0.6298 | 0.6394 | 0.6490 | 0.7212 | 0.7067 | 0.7404 |
| no_redundancy | 0.5721 | 0.6635 | 0.7692 | 0.8413 | 0.8606 | 0.9519 |
| no_infogain | 0.5625 | 0.7212 | 0.7644 | 0.8798 | 0.8510 | 0.9231 |

These rows are the 4-epoch component configs, not the 6-epoch plus 3-epoch prefix-robust checkpoint. The prefix-protocol ablation, including a PCR-IG-only row, is in `ABLATION_RESULTS.md`. On that protocol, `-cost` and `-calibration` keep the saved PRMVT weights: the mask-count cost has no parameter gradient, and a post-hoc calibrator does not change the uncalibrated 0.5-threshold accuracy. The distinct cost measurement for the saved checkpoint is the acquisition policy in `results/qduig/prefix_ft/seed42/cost_policy_test.json`. A matched retraining where the cost term includes predictive entropy is in `ABLATION_RESULTS.md`: 1-view accuracy is 0.9519 with that term and 0.9615 without it. The distinct calibration measurement for the saved checkpoint is the 6-view HER accuracy 0.9711538461538461 against the uncalibrated 0.9759615384615384, plus ECE in `CALIBRATION_RESULTS.md`.

## 8. Robustness

# Robustness, seed 42, test split

Drop is clean accuracy minus corrupted accuracy at the same view count. Thirteen corruptions were measured, including the twelve named in the plan plus sensor noise.
Source: `realtime_bangla_taka_detection/results/robustness/top_seed42.json`.

| corruption | severity | views | PRMVT | drop | NDAL | drop | baseline | drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gaussian_blur | 3.0 | 1 | 0.9760 | -0.0048 | 0.9808 | 0.0000 | 0.7308 | 0.0048 |
| gaussian_blur | 3.0 | 2 | 0.9808 | 0.0000 | 0.9712 | 0.0000 | 0.8654 | 0.0048 |
| gaussian_blur | 3.0 | 3 | 0.9712 | 0.0048 | 0.9808 | -0.0048 | 0.9087 | 0.0048 |
| gaussian_blur | 3.0 | 4 | 0.9760 | 0.0000 | 0.9808 | -0.0144 | 0.9183 | 0.0048 |
| gaussian_blur | 3.0 | 5 | 0.9760 | 0.0096 | 0.9615 | -0.0048 | 0.8990 | 0.0000 |
| gaussian_blur | 3.0 | 6 | 0.9760 | 0.0000 | 0.9663 | 0.0000 | 0.9135 | 0.0048 |
| motion_blur | 5.0 | 1 | 0.9712 | 0.0000 | 0.9808 | 0.0000 | 0.7404 | -0.0048 |
| motion_blur | 5.0 | 2 | 0.9712 | 0.0096 | 0.9712 | 0.0000 | 0.8702 | 0.0000 |
| motion_blur | 5.0 | 3 | 0.9760 | 0.0000 | 0.9712 | 0.0048 | 0.9038 | 0.0096 |
| motion_blur | 5.0 | 4 | 0.9760 | 0.0000 | 0.9808 | -0.0144 | 0.9231 | 0.0000 |
| motion_blur | 5.0 | 5 | 0.9808 | 0.0048 | 0.9663 | -0.0096 | 0.9038 | -0.0048 |
| motion_blur | 5.0 | 6 | 0.9760 | 0.0000 | 0.9663 | 0.0000 | 0.9135 | 0.0048 |
| low_light | 0.35 | 1 | 0.6202 | 0.3510 | 0.7740 | 0.2067 | 0.6010 | 0.1346 |
| low_light | 0.35 | 2 | 0.5962 | 0.3846 | 0.6971 | 0.2740 | 0.6490 | 0.2212 |
| low_light | 0.35 | 3 | 0.6058 | 0.3702 | 0.7067 | 0.2692 | 0.6587 | 0.2548 |
| low_light | 0.35 | 4 | 0.6058 | 0.3702 | 0.7404 | 0.2260 | 0.6587 | 0.2644 |
| low_light | 0.35 | 5 | 0.6106 | 0.3750 | 0.7452 | 0.2115 | 0.6442 | 0.2548 |
| low_light | 0.35 | 6 | 0.5962 | 0.3798 | 0.7115 | 0.2548 | 0.6202 | 0.2981 |
| brightness | 1.6 | 1 | 0.8221 | 0.1490 | 0.7885 | 0.1923 | 0.7740 | -0.0385 |
| brightness | 1.6 | 2 | 0.7596 | 0.2212 | 0.6490 | 0.3221 | 0.6442 | 0.2260 |
| brightness | 1.6 | 3 | 0.8077 | 0.1683 | 0.6346 | 0.3413 | 0.6490 | 0.2644 |
| brightness | 1.6 | 4 | 0.6635 | 0.3125 | 0.6683 | 0.2981 | 0.6635 | 0.2596 |
| brightness | 1.6 | 5 | 0.7115 | 0.2740 | 0.6346 | 0.3221 | 0.6202 | 0.2788 |
| brightness | 1.6 | 6 | 0.7212 | 0.2548 | 0.6875 | 0.2788 | 0.6587 | 0.2596 |
| contrast | 1.8 | 1 | 0.9615 | 0.0096 | 0.9519 | 0.0288 | 0.6875 | 0.0481 |
| contrast | 1.8 | 2 | 0.9471 | 0.0337 | 0.9423 | 0.0288 | 0.8798 | -0.0096 |
| contrast | 1.8 | 3 | 0.9423 | 0.0337 | 0.9712 | 0.0048 | 0.9038 | 0.0096 |
| contrast | 1.8 | 4 | 0.9519 | 0.0240 | 0.9615 | 0.0048 | 0.9087 | 0.0144 |
| contrast | 1.8 | 5 | 0.9471 | 0.0385 | 0.9615 | -0.0048 | 0.9038 | -0.0048 |
| contrast | 1.8 | 6 | 0.9663 | 0.0096 | 0.9712 | -0.0048 | 0.9038 | 0.0144 |
| glare | 0.65 | 1 | 0.9327 | 0.0385 | 0.9135 | 0.0673 | 0.8221 | -0.0865 |
| glare | 0.65 | 2 | 0.9471 | 0.0337 | 0.9327 | 0.0385 | 0.8750 | -0.0048 |
| glare | 0.65 | 3 | 0.9760 | 0.0000 | 0.9279 | 0.0481 | 0.9135 | 0.0000 |
| glare | 0.65 | 4 | 0.9375 | 0.0385 | 0.9375 | 0.0288 | 0.9279 | -0.0048 |
| glare | 0.65 | 5 | 0.9135 | 0.0721 | 0.9183 | 0.0385 | 0.9183 | -0.0192 |
| glare | 0.65 | 6 | 0.9615 | 0.0144 | 0.9471 | 0.0192 | 0.9327 | -0.0144 |
| occlusion | 0.2 | 1 | 0.7644 | 0.2067 | 0.7692 | 0.2115 | 0.6154 | 0.1202 |
| occlusion | 0.2 | 2 | 0.8077 | 0.1731 | 0.7692 | 0.2019 | 0.7115 | 0.1587 |
| occlusion | 0.2 | 3 | 0.8462 | 0.1298 | 0.8029 | 0.1731 | 0.7452 | 0.1683 |
| occlusion | 0.2 | 4 | 0.7163 | 0.2596 | 0.8317 | 0.1346 | 0.7788 | 0.1442 |
| occlusion | 0.2 | 5 | 0.7212 | 0.2644 | 0.7981 | 0.1587 | 0.7548 | 0.1442 |
| occlusion | 0.2 | 6 | 0.7500 | 0.2260 | 0.8077 | 0.1587 | 0.7885 | 0.1298 |
| occlusion | 0.55 | 1 | 0.4904 | 0.4808 | 0.6731 | 0.3077 | 0.7356 | 0.0000 |
| occlusion | 0.55 | 2 | 0.6587 | 0.3221 | 0.6106 | 0.3606 | 0.6923 | 0.1779 |
| occlusion | 0.55 | 3 | 0.6827 | 0.2933 | 0.6779 | 0.2981 | 0.7837 | 0.1298 |
| occlusion | 0.55 | 4 | 0.5000 | 0.4760 | 0.7260 | 0.2404 | 0.7981 | 0.1250 |
| occlusion | 0.55 | 5 | 0.5288 | 0.4567 | 0.7115 | 0.2452 | 0.7885 | 0.1106 |
| occlusion | 0.55 | 6 | 0.5096 | 0.4663 | 0.7356 | 0.2308 | 0.8125 | 0.1058 |
| jpeg | 30.0 | 1 | 0.9712 | 0.0000 | 0.9808 | 0.0000 | 0.7548 | -0.0192 |
| jpeg | 30.0 | 2 | 0.9760 | 0.0048 | 0.9712 | 0.0000 | 0.8750 | -0.0048 |
| jpeg | 30.0 | 3 | 0.9808 | -0.0048 | 0.9760 | 0.0000 | 0.8942 | 0.0192 |
| jpeg | 30.0 | 4 | 0.9712 | 0.0048 | 0.9760 | -0.0096 | 0.9087 | 0.0144 |
| jpeg | 30.0 | 5 | 0.9808 | 0.0048 | 0.9567 | 0.0000 | 0.9087 | -0.0096 |
| jpeg | 30.0 | 6 | 0.9808 | -0.0048 | 0.9663 | 0.0000 | 0.9135 | 0.0048 |
| rotation | 20.0 | 1 | 0.6635 | 0.3077 | 0.7837 | 0.1971 | 0.7452 | -0.0096 |
| rotation | 20.0 | 2 | 0.6538 | 0.3269 | 0.6346 | 0.3365 | 0.7981 | 0.0721 |
| rotation | 20.0 | 3 | 0.8173 | 0.1587 | 0.7837 | 0.1923 | 0.8413 | 0.0721 |
| rotation | 20.0 | 4 | 0.7644 | 0.2115 | 0.8317 | 0.1346 | 0.8462 | 0.0769 |
| rotation | 20.0 | 5 | 0.7500 | 0.2356 | 0.7837 | 0.1731 | 0.8606 | 0.0385 |
| rotation | 20.0 | 6 | 0.8462 | 0.1298 | 0.8558 | 0.1106 | 0.8077 | 0.1106 |
| perspective | 0.1 | 1 | 0.9327 | 0.0385 | 0.9375 | 0.0433 | 0.6827 | 0.0529 |
| perspective | 0.1 | 2 | 0.9375 | 0.0433 | 0.9327 | 0.0385 | 0.8029 | 0.0673 |
| perspective | 0.1 | 3 | 0.9471 | 0.0288 | 0.9663 | 0.0096 | 0.8798 | 0.0337 |
| perspective | 0.1 | 4 | 0.9519 | 0.0240 | 0.9375 | 0.0288 | 0.8990 | 0.0240 |
| perspective | 0.1 | 5 | 0.9375 | 0.0481 | 0.9135 | 0.0433 | 0.8894 | 0.0096 |
| perspective | 0.1 | 6 | 0.9327 | 0.0433 | 0.9327 | 0.0337 | 0.8846 | 0.0337 |
| scale | 0.7 | 1 | 0.9712 | 0.0000 | 0.9808 | 0.0000 | 0.7356 | 0.0000 |
| scale | 0.7 | 2 | 0.9760 | 0.0048 | 0.9712 | 0.0000 | 0.8750 | -0.0048 |
| scale | 0.7 | 3 | 0.9712 | 0.0048 | 0.9760 | 0.0000 | 0.9038 | 0.0096 |
| scale | 0.7 | 4 | 0.9760 | 0.0000 | 0.9663 | 0.0000 | 0.9183 | 0.0048 |
| scale | 0.7 | 5 | 0.9808 | 0.0048 | 0.9663 | -0.0096 | 0.8990 | 0.0000 |
| scale | 0.7 | 6 | 0.9760 | 0.0000 | 0.9712 | -0.0048 | 0.9183 | 0.0000 |
| sensor_noise | 12.0 | 1 | 0.9760 | -0.0048 | 0.9808 | 0.0000 | 0.7452 | -0.0096 |
| sensor_noise | 12.0 | 2 | 0.9808 | 0.0000 | 0.9712 | 0.0000 | 0.8798 | -0.0096 |
| sensor_noise | 12.0 | 3 | 0.9760 | 0.0000 | 0.9663 | 0.0096 | 0.9087 | 0.0048 |
| sensor_noise | 12.0 | 4 | 0.9760 | 0.0000 | 0.9663 | 0.0000 | 0.9183 | 0.0048 |
| sensor_noise | 12.0 | 5 | 0.9808 | 0.0048 | 0.9615 | -0.0048 | 0.9038 | -0.0048 |
| sensor_noise | 12.0 | 6 | 0.9760 | 0.0000 | 0.9663 | 0.0000 | 0.9135 | 0.0048 |


An earlier six-view corruption file remains at `results/qduig/eval/seed42/robustness.json`. Where the new file and that file disagree, the new file is the one for PRMVT and NDAL across one to six views, and the older file is the six-view Q-DUIG comparison already quoted in `FINAL_RESULTS.md`.

## 9. Edge deployment

Pi 5 latency, FPS, CPU, RAM, temperature, and a 30-minute sustained run are NOT_MEASURED (`EDGE_RESULTS.md`). TFLite and Android are NOT_MEASURED (`MOBILE_RESULTS.md`). A previous host timing on this CUDA laptop, median 42.6 ms for six views, is in `results/qduig/edge/edge.json`. That number is not a Pi 5 result.

## 10. Discussion

The one-view gap between fixed-count training and prefix-robust training is the result that matches the support proposition. Several later losses also land near 0.95–0.98 at one view on this same split, so the accuracy alone does not identify a unique algorithm. The paired tests in section 6 are the comparison against the CNN+ViT baseline, with Bonferroni applied across the 36 algorithm-by-view tests that had predictions. After that correction, several six-view differences are not significant. That belongs in the paper.

## 11. Limitations

SFPL v2 is above the majority-class rate and below the baseline from two views up. OGPD v2 is worse than OGPD v1. VCIE v2 recovered from a constant predictor and remains below PRMVT. CVS does not implement do-calculus. Federated training is simulated. There is no second dataset. There is no Pi 5. Three seeds, where they exist, are a small sample for a seed-level interval.

## 12. Conclusion

Prefix-robust training removes the fixed view-count support mismatch and, on this split and seed 42, holds accuracy from one view through six. The other algorithms are measured variants, including the weak ones. Claims that need a missing artifact are marked NOT_MEASURED.

## Figures

Measured plots are written only when the source JSON exists. A missing figure is named here and not drawn.

1. Architecture: the encoder and heads are described in section 3. A schematic that is not computed from data is not included.
2. PRMVT two-stage training curve: `results/qduig/prefix/seed42/train_log.csv` and `results/qduig/prefix_ft/seed42/train_log.csv` if present.
3. One-to-six view accuracy: the table in section 6.
4. Ablation: section 7.
5. Calibration: ECE columns in the per-algorithm `test_metrics.json` files. A new reliability diagram was not redrawn in this draft.
6. Robustness: section 8.
7. Oracle gap: the bound is in `THEORETICAL_ANALYSIS.md`. The measured oracle accuracy 0.9856 is the number already cited in `FINAL_RESULTS.md`.
8. VCDS: the proposition is an inequality, not a fitted curve.
9. Confusion matrices already saved under each `test/` or `test_views/` directory.
10. Pi 5 latency breakdown: NOT_MEASURED.
11. Failure analysis: notes where the one-view prediction disagrees with the label are in the prediction JSON files. They were not re-listed here.
12. End-to-end pipeline: capture, six views, encoder, head, threshold 0.5. No extra measured stage.

## Tables

1. Dataset: section 5.
2. Baseline versus the six leading algorithms: section 6.
3. One-to-six view performance: section 6.
4. Multi-seed: `MULTI_SEED_RESULTS.md`.
5. Ablation: section 7.
6. Calibration: ECE in `test_metrics.json`. A single calibration table for every algorithm was not recomputed in this draft beyond those files.
7. Robustness: section 8.
8. Edge: section 9.
9. External SOTA on JaalTaka: NOT_MEASURED. Published banknote papers use other currencies and other splits.
10. Negative results: SFPL v2, OGPD v2, and the v1 constant predictors, section 6 and section 11.
11. Theoretical bounds: `THEORETICAL_ANALYSIS.md`. Hoeffding's fixed-predictor count is 738 notes against a training split of 974. The measured fixed-count gap is 0.3798076923076923 (6-view accuracy 0.9663461538461539 minus 1-view accuracy 0.5865384615384616). The PAC-Bayes penalty on the stated Gaussian grid is 57.5892990573559 and the bound is vacuous.
