# Watermark-aware fusion head (2026-09-30)

End-to-end attention fusion over frozen ImageNet ResNet-50 tokens: the first k view features plus one watermark-window feature. Trained on the serial-disjoint TRAIN split with random k in 1..6 and the watermark token dropped with probability 0.2. The epoch (of 40) is the one with the best VAL mean accuracy. TEST was read once. Seeds 42, 43, 44.

Source: `realtime_bangla_taka_detection/scripts/eval/beat_resnet.py` → `results/serial_split/beat_resnet.json`.

| | 1 view | 6 views |
|---|---:|---:|
| Fusion with the watermark token | 90.8 ± 0.3 % | 90.5 ± 0.0 % |
| Same fusion, watermark token removed | 87.8 ± 0.9 % | 88.0 ± 0.9 % |
| Frozen ResNet-50 probe | 85.1 % | 83.8 % |

The watermark token adds about 3 points. This head does not beat the separate MobileNetV2 watermark classifier combined with the prefix network (94.4 ± 0.5 % / 95.0 ± 0.0 %). That combination is the one to cite.
