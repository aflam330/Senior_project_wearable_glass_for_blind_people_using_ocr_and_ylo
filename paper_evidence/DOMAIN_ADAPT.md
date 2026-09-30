# Domain adaptation that was actually run (2026-09-30)

Two different adaptations are on disk. Neither was tuned on the serial-disjoint test split.

## Fine-tuned ResNet-50 on the serial-disjoint split

ImageNet ResNet-50, layer3, layer4, and a new 2-way head trained for 5 epochs on TRAIN views. Earlier layers stay frozen. The epoch is the one with the best VAL mean over k = 1..6. Seeds 42, 43, 44. Script: `scripts/train/ft_resnet_serial.py`. Files: `results/serial_split/ft_resnet50/seed42.json`, `seed43.json`, `seed44.json`.

| Seed | Best VAL mean | TEST, 1 view | TEST, 6 views |
|---|---:|---:|---:|
| 42 | 0.803 | 0.923 | 0.851 |
| 43 | 0.839 | 0.896 | 0.910 |
| 44 | 0.899 | 0.941 | 0.937 |
| Mean ± sd | | 92.0 ± 2.3 % | 89.9 ± 4.4 % |

This is the fair baseline. The watermark hybrid is +2.4 points at 1 view and +5.1 points at 6 views (`BEAT_RESNET50_FINAL.md`).

## Unsupervised adaptation of the whole-note counterfeit check

Source: `results/jaal_whole/domain_adapt.json` (88 adaptation crops).

| Method | AUC on counterfeit-set originals |
|---|---:|
| S4 | 0.810 |
| S4 + AdaBN | 0.641 |
| R2 | 0.961 |
| R2 + CORAL | 0.916 |

AdaBN and CORAL both lowered AUC. Neither met its success rule. They were not kept.
