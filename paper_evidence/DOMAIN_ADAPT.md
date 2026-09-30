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

## Full-resolution fine-tune

The first fine-tune was trained on `cache/views256`, which had been decoded at one-quarter JPEG resolution. A second run, same layers and the same validation rule, uses `cache/views256_full` (full JPEG, short side 256, then 224). Eight epochs, best epoch on validation. Files: `results/serial_split/ft_resnet50_fullres/seed42.json`, `seed43.json`, `seed44.json`.

| | 1 view | 6 views |
|---|---:|---:|
| Quarter-resolution cache, 5 epochs | 92.0 ± 2.3 % | 89.9 ± 4.4 % |
| Full-resolution cache, 8 epochs | 94.9 ± 1.8 % | 94.4 ± 2.6 % |

Best validation means were 0.950 (seed 42, epoch 7), 0.949 (seed 43, epoch 8) and 0.939 (seed 44, epoch 1). Seed 44's later epochs fell on validation, so epoch 1 was kept. Detail and paired tests: `BEAT_RESNET50_FINAL.md`.
