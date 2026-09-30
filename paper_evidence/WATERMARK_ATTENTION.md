# Watermark attention (2026-09-30)

Two attention models were in the validation-ranked search (`WATERMARK_ARCH_SEARCH.md`).

| Model | Mean validation AUC | Test accuracy | Test AUC |
|---|---:|---:|---:|
| Swin-T | 0.9819 ± 0.0137 | 90.9 ± 1.3% | 0.954 |
| CNN stem + two transformer layers | 0.9527 ± 0.0124 | 86.3 ± 0.9% | 0.931 |

Neither is the validation winner. CBAM, squeeze-and-excitation and a separate cross-attention over serial or denomination features were not trained.
