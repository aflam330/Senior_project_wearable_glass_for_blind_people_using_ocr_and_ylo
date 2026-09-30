# Accuracy after the validation-ranked search (2026-09-30)

Every number below is from `results/watermark_arch/summary.json` or from the earlier published files cited in the table. The winning architecture was chosen by mean validation AUC. The test set did not choose it.

| System | Accuracy | Where |
|---|---:|---|
| Published MobileNetV2 watermark | 92.9% (AUC 0.976) | 197 registered test notes |
| EfficientNet-B0, validation winner, three-seed mean | 92.4% (AUC 0.962) | same 197 notes |
| EfficientNet-B0, mean of the three seed accuracies | 91.2 ± 1.6% | same 197 notes |
| Hybrid, published MobileNetV2 + prefix network | 95.0% on every seed | 222 notes, six views |
| Hybrid, EfficientNet-B0 + prefix network | 94.6%, 95.5%, 94.6% | 222 notes, six views |
| Full-resolution fine-tuned ResNet-50 | 94.4 ± 2.6% | 222 notes, six views |

Against that fine-tune, the new hybrid's exact McNemar p-values at six views are 0.219, 1.0 and 0.143.

Best accuracy in this pass: **95.5%** on one network seed at six views (seed 43). The three-seed picture is 94.6 / 95.5 / 94.6%, which is not above the published hybrid's 95.0%.

Versus the published watermark (92.9%): the new watermark mean is **−0.5 points**.
Versus the published hybrid (95.0%): the new six-view hybrid is **not higher**.
p-value versus the fine-tune: **0.219, 1.0, 0.143**. Not p < 0.001.
Model size of the validation winner: **4.01 million parameters**. A checkpoint file size and an inference time were not measured.

**96% with p < 0.001 was not reached.**

On 222 notes, 96% is 213 correct. The new hybrid is 210, 212 and 210 correct. The gap to a paired p < 0.001 is larger than one or two notes: the discordant counts against the fine-tune are (1, 5), (4, 4) and (12, 5). More architectures on this same test will not create that margin. The limit is the number of unseen prints, about 24 effective prints, already stated with the earlier results.
