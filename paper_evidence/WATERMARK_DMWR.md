# Denomination-matched watermark residual (2026-09-30)

The published watermark model is a MobileNetV2 fine-tune of the registered back-lit window. This note adds a different algorithm on the same crops and the same serial-disjoint split. The cited variant was fixed before the test was read. It does not replace MobileNetV2.

Source: `realtime_bangla_taka_detection/scripts/train/train_watermark_dmwr.py` → `results/watermark_dmwr/summary.json`. Figure: `paper_evidence/figures/fig31_watermark_dmwr.png`.

## Algorithm

View 6 is already registered and cropped (`results/watermark/crops`, 224 px). Each crop is resized to 128 px, divided by a Gaussian of σ = 16, and z-scored. That quotient image removes the backlight level and leaves the thickness portrait.

The genuine portrait of a denomination is the pixel-wise median of training genuine quotient images. Only 500 and 1000 have at least 8 such notes (235 and 260). Every other denomination uses the median of all 495 training genuine notes. Validation and test notes never enter a prototype.

The residual is the quotient image minus that portrait. Four scalars go with it: normalised correlation with the portrait, residual standard deviation, residual Laplacian variance, and the mean absolute fine detail after a Gaussian of σ = 2. Scalar means and standard deviations are fit on the training split.

The cited network has two streams of three stride-2 convolutions (16, 32, 64 channels). One reads the quotient image. One reads the residual. Their pooled features are concatenated with the four scalars and a linear layer. Training uses a rotation of ±5° and a shift of ±3 px, applied to the crop before the residual is formed. Twelve epochs, AdamW at 1e-3. The kept epoch is the best validation AUC. The decision threshold is 0.5, the same rule as the published MobileNetV2. Seeds 42, 43, 44. Checkpoints are about 0.20 MB each and do not overwrite `watermark_mobilenetv2.pt`.

Two ablations use the same schedule: the portrait stream alone, and the residual stream plus the scalars. A logistic regression on the four scalars plus Laplacian variance, grey standard deviation, edge density, and relative mean is fit on train; its threshold maximises validation accuracy.

Registered notes: 867 train, 197 validation, 197 test.

## Test, read once

| Method | Accuracy | AUC | Genuine flagged | Counterfeits missed |
|---|---:|---:|---:|---:|
| Matched-filter logistic | 87.3 % | 0.906 | 9 / 98 | 16 / 99 |
| Portrait stream only | 88.0 ± 3.3 % | 0.938 ± 0.003 | — | — |
| Residual stream only | 85.8 ± 5.3 % | 0.935 ± 0.003 | — | — |
| DMWR, both streams | 91.5 ± 0.3 % | 0.950 ± 0.006 | 1, 5, 2 / 98 | 16, 11, 15 / 99 |
| Published MobileNetV2, same 197 notes | 92.9 % | 0.976 | 2 / 98 | 12 / 99 |

Per seed, both streams: accuracy 91.4 / 91.9 / 91.4 %, AUC 0.945 / 0.948 / 0.957. Best validation AUC was 0.950, 0.956, 0.955.

Exact McNemar against the published MobileNetV2, DMWR-only correct versus MobileNet-only correct: (1, 4), (2, 4), (1, 4), with p = 0.375, 0.688, 0.375. The point estimates favour MobileNetV2. The difference is not significant on this test. Both streams beat either stream alone, so the portrait and the residual are both used, and that is still short of the ImageNet fine-tune.

## What this is for the paper

The new piece is the genuine-only denomination portrait and the residual against it, not a higher accuracy. MobileNetV2 remains the watermark number to cite (92.9 %, AUC 0.976). DMWR is the interpretable comparator, at 0.20 MB rather than the 8.9 MB FP32 MobileNet. It does not move the six-view hybrid, and it does not change the venue limits in `VENUE_READINESS.md`.
