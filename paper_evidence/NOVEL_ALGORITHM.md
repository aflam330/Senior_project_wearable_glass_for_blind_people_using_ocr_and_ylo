# Algorithms on disk (2026-09-30)

No new network was trained in this pass. Training another fusion after the serial-disjoint test has already been read, then quoting that test, would choose on the test. The algorithms that have a finished three-seed measurement are these.

## Watermark hybrid

Logistic regression fitted on validation only, on the prefix-network logit, the MobileNetV2 watermark logit, and a missing-watermark flag. Test read once per seed.

| | 1 view | 6 views |
|---|---:|---:|
| Hybrid | 94.4 ± 0.5 % | 95.0 ± 0.0 % |
| Full-resolution ResNet-50 fine-tune | 94.9 ± 1.8 % | 94.4 ± 2.6 % |
| Watermark model alone | 92.9 %, AUC 0.976 | same score; it does not use the six views |

False counterfeit calls on genuine test notes, six views: 4, 4, 4 of 121 (3.3 %). Counterfeits missed: 7, 7, 7 of 101. Source: `BEAT_RESNET50_FINAL.md`.

## Safety-aware rejection

Answer only when confidence is at least a threshold chosen so that validation error among answered notes is at most 1 %. Otherwise the glass says to check by hand. On the unseen-print hybrid this is recorded in `hybrid_final_seeds.json` under `rejection`. The calibration assumed by the validation threshold does not fully carry to new prints: wrong answers among the notes that were answered are higher than the 1 % validation cap (`THEORY_FINAL.md`, item 3).

## What was not invented here

Print-contrastive learning and a tri-modal watermark–serial–image net are not in any result file. Serial anomaly on unseen prints is already known to be zero. Reporting either as a trained system would be a number without a run.
