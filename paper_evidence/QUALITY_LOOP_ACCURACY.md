# Quality loop for the accuracy search (2026-09-30)

| # | What was measured | Result | Kept as the cited system |
|---|---|---|---|
| 1 | Ten watermark models, three seeds, epoch and winner by validation AUC | Winner EfficientNet-B0, test 91.2 ± 1.6%; three-seed mean 92.4% | no; published MobileNetV2 stays at 92.9% |
| 2 | CLAHE before MobileNetV2 | test 91.2 ± 0.6% | no |
| 3 | Swin-T and a small CNN-transformer | 90.9 ± 1.3% and 86.3 ± 0.9% | no |
| 4 | Equal-weight mean of the winning seeds | 92.4% | no |
| 5 | Validation logistic regression of that mean with the prefix network | six-view 94.6 / 95.5 / 94.6%; McNemar vs full-resolution ResNet-50 p = 0.219, 1.0, 0.143 | no; published hybrid stays at 95.0% |

Stopped. The test for these models has been read. Another round of architectures, losses or thresholds, kept only if the test rose, would be chosen by that test. 96% with p < 0.001 is not available from the discordant counts above.
