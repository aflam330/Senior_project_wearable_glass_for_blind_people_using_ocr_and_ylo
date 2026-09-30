# Watermark architecture search (2026-09-30)

Ten candidates, seeds 42, 43 and 44, eight epochs, AdamW. The kept epoch is the best validation AUC. The cited architecture is the highest mean validation AUC. The test column was not used to choose it.

Source: `realtime_bangla_taka_detection/scripts/train/train_watermark_arch_search.py` → `results/watermark_arch/summary.json`.

Registered crops only: 867 train, 197 validation, 197 test. Published MobileNetV2 on those 197 test notes is 92.9% (AUC 0.976).

| Rank by validation AUC | Architecture | Parameters | Mean validation AUC | Test accuracy | Test AUC |
|---|---|---:|---:|---:|---:|
| 1, cited | EfficientNet-B0 | 4.01M | 0.9975 ± 0.0041 | 91.2 ± 1.6% | 0.964 |
| 2 | MobileNetV2 + CLAHE | 2.23M | 0.9961 ± 0.0024 | 91.2 ± 0.6% | 0.972 |
| 3 | MobileNetV3-Large | 4.20M | 0.9960 ± 0.0035 | 92.6 ± 0.3% | 0.964 |
| 4 | MobileNetV3-Small | 1.52M | 0.9952 ± 0.0023 | 90.2 ± 1.1% | 0.968 |
| 5 | DenseNet-121 | 6.96M | 0.9951 ± 0.0010 | 92.0 ± 0.6% | 0.963 |
| 6 | ResNet-34 | 21.3M | 0.9947 ± 0.0024 | 90.7 ± 1.6% | 0.960 |
| 7 | ConvNeXt-Tiny | 27.8M | 0.9941 ± 0.0024 | 90.0 ± 4.1% | 0.967 |
| 8 | ResNet-18 | 11.2M | 0.9893 ± 0.0066 | 87.3 ± 5.5% | 0.951 |
| 9 | Swin-T | 27.5M | 0.9819 ± 0.0137 | 90.9 ± 1.3% | 0.954 |
| 10 | CNN + transformer, from scratch | 0.11M | 0.9527 ± 0.0124 | 86.3 ± 0.9% | 0.931 |

The mean of the three EfficientNet-B0 probabilities, threshold 0.5, is 92.4% (AUC 0.962): 1 of 98 genuine notes flagged, 14 of 99 counterfeits missed. That is below the published MobileNetV2.

MobileNetV3-Large has a higher test accuracy than EfficientNet-B0. It is not the cited model, because its validation AUC is lower. Switching to it would be a test-set choice.

Not in this torchvision build, and not scored: EfficientFormer, LeViT, ViT-Small, ConvNeXt-Nano. ResNet-50 and EfficientNet-B1/B2 were left out of the ten-candidate budget.
