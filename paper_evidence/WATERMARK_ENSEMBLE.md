# Watermark MobileNet ensemble (2026-09-30)

Four MobileNetV2 checkpoints, averaged with equal weight. The threshold was chosen on validation before the test comparison was read. Published weights were not overwritten.

Source: `realtime_bangla_taka_detection/scripts/train/train_watermark_ensemble.py` → `results/watermark_ensemble/summary.json`. Figure: `paper_evidence/figures/fig32_watermark_ensemble.png`.

## Rule

Seeds 42, 43 and 44 use the same crops, 10 epochs, and AdamW settings as the published run. Each seed keeps the epoch with the best validation AUC. The published checkpoint is the fourth member. The cited score is the mean of the four genuine-probabilities. The threshold is the one with the highest validation accuracy; a tie would keep the value closer to 0.5. That threshold is 0.924. Validation accuracy there is 98.5%.

A logistic regression of this mean with the DMWR mean was fit on validation, with the threshold left at 0.5. It is not the cited system.

## Test, 197 registered notes

| Rule | Accuracy | AUC | Genuine flagged | Counterfeits missed |
|---|---:|---:|---:|---:|
| Published MobileNetV2, threshold 0.5 | 92.9 % | 0.976 | 2 / 98 | 12 / 99 |
| Each new seed alone, threshold 0.5 | 92.4 % | 0.976, 0.975, 0.982 | 1 / 98 | 14 / 99 |
| Ensemble, threshold 0.5 | 92.4 % | 0.978 | 1 / 98 | 14 / 99 |
| Ensemble, validation threshold 0.924 | 93.4 % | 0.978 | 6 / 98 | 7 / 99 |
| Ensemble stacked with DMWR, threshold 0.5 | 92.4 % | 0.968 | 1 / 98 | 14 / 99 |

The cited row is 93.4 %, one note above the published model (184 / 197 versus 183 / 197). Exact McNemar is 5 notes correct only for the ensemble and 4 only for the published model, p = 1.0. The gain is the threshold: at 0.5 the same ensemble is 92.4 %. The higher threshold misses fewer counterfeits and flags more genuine notes.

Best validation AUC of the new seeds: 0.996, 0.997, 0.997.
