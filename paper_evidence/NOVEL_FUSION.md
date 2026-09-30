# Learned multi-view fusion (2026-09-30)

The fusion that was trained is attention over view tokens and one watermark-window token (`Fusion` in `scripts/eval/beat_resnet.py`). Weights are learned on TRAIN. The checkpoint is chosen on VAL. Seeds 42, 43, 44. Split: serial-disjoint, 222 test notes.

| Method | 1 view | 6 views |
|---|---:|---:|
| Attention, views only | 87.8 ± 0.9 % | 88.0 ± 0.9 % |
| Attention, views + watermark token | 90.8 ± 0.3 % | 90.5 ± 0.0 % |
| Uncertainty-weighted ensemble (network, fusion, watermark) | 93.4 ± 0.3 % | 93.1 ± 0.3 % |
| Validation-fitted logistic regression on network logit + watermark logit | **94.4 ± 0.5 %** | **95.0 ± 0.0 %** |

Source: `results/serial_split/beat_resnet.json`. The learned attention head is real and helps. The strongest fusion on this split is still the validation-fitted logistic regression, not the attention head.
