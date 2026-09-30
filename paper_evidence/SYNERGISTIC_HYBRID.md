# Hybrid with the validation-winning watermark (2026-09-30)

The watermark input is the mean EfficientNet-B0 probability from seeds 42, 43 and 44. A logistic regression is fit on validation only, with features [prefix-network logit, watermark logit, missing-crop flag]. Test has 222 notes. A note whose watermark crop did not register gets probability 0.5 and the missing flag.

Compared with the saved full-resolution ResNet-50 fine-tune on the same notes. Exact McNemar.

| Network seed | Views | Hybrid | Fine-tuned ResNet-50 | Hybrid only correct | Fine-tune only correct | p |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 1 | 94.6% | 93.2% | 8 | 5 | 0.581 |
| 42 | 6 | 94.6% | 96.4% | 1 | 5 | 0.219 |
| 43 | 1 | 94.6% | 94.6% | 6 | 6 | 1.0 |
| 43 | 6 | 95.5% | 95.5% | 4 | 4 | 1.0 |
| 44 | 1 | 95.0% | 96.8% | 0 | 4 | 0.125 |
| 44 | 6 | 94.6% | 91.4% | 12 | 5 | 0.143 |

Six-view hybrid: 94.6%, 95.5%, 94.6%. The published prefix-plus-MobileNetV2 hybrid is 95.0% at six views on every seed. This refit does not raise it.

No six-view comparison reaches p < 0.001 against the fine-tune. None reaches 96%.

Serial, DMWR, denomination and safety-rejection were not added as further features. The published six-view hybrid already includes the watermark and the network, and an earlier four-feature combiner with the fine-tune did not help (`BEAT_FINETUNED_RESNET50.md`).
