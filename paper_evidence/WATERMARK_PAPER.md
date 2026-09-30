# Watermark-Based Counterfeit Detection for Bangladeshi Taka: Print-Disjoint Evaluation and a Safety Guarantee

*Short paper from measured files only. 2026-09-30. Authors to be added.*

## Abstract

A public Bangladeshi counterfeit set is not print-disjoint: many counterfeit notes share a serial. On a split that keeps every test print out of training, a fine-tuned ResNet-50 that sees full-resolution views reaches 94.9 ± 1.8 % with one view and 94.4 ± 2.6 % with six (three seeds, 222 notes). A MobileNetV2 classifier on the back-lit watermark window alone reaches 92.9 % (AUC 0.976). Combined with a multi-view network by a validation-fitted logistic regression, accuracy is 94.4 ± 0.5 % and 95.0 ± 0.0 %. The combination does not significantly beat the full-resolution fine-tune. It is more stable across seeds, flags 4 of 121 genuine notes (3.3 %), and misses 7 of 101 unseen-print counterfeits. A serial blacklist misses every new print. The watermark model exports to 2.6 MB INT8 with the same decisions as FP32. Raspberry Pi 5 latency is not measured.

## 1. The split

JaalTaka counterfeits repeat serials. Note-disjoint accuracy near 98 % therefore recycles prints. The serial-disjoint test has 222 notes and 101 counterfeits whose prints are absent from training. About 24 prints carry those 101 notes (Theorem 16), so absolute accuracies have a print-level half-width near 0.28. Paired tests on the same notes remain valid.

## 2. Method

View 6 is the back-lit photograph. It is registered and the blank oval is cropped (`watermark_features.py`). MobileNetV2 is fine-tuned on training crops; the epoch is chosen by validation AUC. At test time a logistic regression sees the prefix-network logit, the watermark logit, and a flag for a crop that did not register. Coefficients are fit on validation only.

The crop pipeline decodes view 6 at half JPEG resolution and scales the width to 700. That is not the quarter-decode bug that affected the ResNet fine-tune. It was not rerun.

## 3. Results

| Method | 1 view | 6 views |
|---|---:|---:|
| Frozen ResNet-50 | 85.1 % | 83.8 % |
| Full-resolution fine-tune | 94.9 ± 1.8 % | 94.4 ± 2.6 % |
| Watermark alone | 92.9 % (AUC 0.976) | — |
| Hybrid | 94.4 ± 0.5 % | 95.0 ± 0.0 % |

The hybrid's six-view false-counterfeit counts are 4, 4, 4 and its miss counts are 7, 7, 7. Against the frozen probe the gain is significant on every seed. Against the full-resolution fine-tune it is not (`BEAT_FINETUNED_RESNET50.md`).

## 4. Safety

The deployed policy does not say "counterfeit". It says "likely genuine" only above the maximum validation counterfeit score, and otherwise asks for a human check. In domain, 0 of 88 test counterfeits passed that bar. A serial list is the wrong tool for a new print: recall on an unseen serial is zero.

## 5. Limits

One counterfeit collection. About 24 effective unseen prints. No user study. No Pi 5 timing. No glass-camera back-lit photos. The watermark does not, on these numbers, beat a corrected ResNet-50 by five points.

## 6. A denomination-matched residual

A second watermark algorithm builds a portrait from training genuine windows only and classifies the residual against that portrait (`WATERMARK_DMWR.md`). On the same 197 registered test notes it reaches 91.5 ± 0.3 % (AUC 0.950 ± 0.006), against 92.9 % (AUC 0.976) for MobileNetV2. Exact McNemar p-values are 0.375, 0.688 and 0.375. MobileNetV2 stays the single-model watermark number (92.9 %, AUC 0.976). An equal-weight ensemble of that checkpoint with three new seeds, using a validation-chosen threshold of 0.924, reaches 93.4 % on the same 197 notes (`WATERMARK_ENSEMBLE.md`). That is one note higher. Exact McNemar p = 1.0. At threshold 0.5 the ensemble is 92.4 %.
