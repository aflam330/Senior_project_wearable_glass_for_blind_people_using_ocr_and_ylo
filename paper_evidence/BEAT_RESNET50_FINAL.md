# Beating the ResNet-50 baselines on unseen counterfeit prints (2026-09-30)

- **Split:** serial-disjoint (print-disjoint), 222 test notes, 101 counterfeits from prints never seen in training.
- **Protocol:** seeds 42–44; all training on TRAIN; every choice, combiner and weight on VAL; TEST read once.
- **Sources:** `scripts/eval/beat_resnet.py` → `results/serial_split/beat_resnet.json`; `scripts/train/ft_resnet_serial.py` → `results/serial_split/ft_resnet50/seed*.json`.

## Results (mean ± sd over 3 seeds)

| Method | Requested as | 1 view | 6 views |
|---|---|---:|---:|
| ResNet-50, frozen features + logistic regression | baseline | 85.1 % | 83.8 % |
| **ResNet-50, fine-tuned** (layer3–4 + head, 5 epochs, epoch on VAL) | stronger baseline (D) | 92.0 ± 2.3 % | 89.9 ± 4.4 % |
| Prefix network | — | 89.9 ± 0.7 % | 89.3 ± 1.3 % |
| Attention fusion, views only | — | 87.8 ± 0.9 % | 88.0 ± 0.9 % |
| Attention fusion, views + watermark token, trained end-to-end | A (watermark head), C (novel fusion) | 90.8 ± 0.3 % | 90.5 ± 0.0 % |
| Ensemble: network + fusion + watermark, weights 1 / VAL log-loss | E | 93.4 ± 0.3 % | 93.1 ± 0.3 % |
| **Hybrid: network + MobileNetV2 watermark, combiner fitted on VAL** | final | **94.4 ± 0.5 %** | **95.0 ± 0.0 %** |
| Serial-anomaly head | B | catches no unseen print by construction (Proposition 12), so not trained as a head | — |

## Paired tests (exact McNemar, per seed; the conservative reading)

| Hybrid vs | 1 view: p per seed | 6 views: p per seed | Mean gain |
|---|---|---|---|
| Frozen ResNet-50 | 0.00032, 0.00032, 0.000027 | 0.0000046 (each seed) | +9.3 / +11.2 points |
| Fine-tuned ResNet-50 | 0.42, **0.041**, 0.75 | **0.000027**, **0.035**, 0.51 | +2.4 / +5.1 points |

Pooling the three seeds gives p = 0.023 (1 view) and 2.0 × 10⁻⁶ (6 views) against the fine-tuned model. That pooling counts every test note three times, so the per-seed values above are the ones to cite.

## Error types (6 views, per seed)

| | Genuine called counterfeit (of 121) | Counterfeits missed (of 101) |
|---|---|---|
| Fine-tuned ResNet-50 | 1, 1, 1 | 32, 19, 13 |
| Hybrid | 4, 4, 4 | 7, 7, 7 |

The hybrid catches far more new-print counterfeits and is stable across seeds. The fine-tuned ResNet-50 raises fewer false alarms but its miss count swings with the seed (13–32).

## Verdict against the targets

- **"+8–10 points over ResNet-50, p < 0.001":**
  - **met against the frozen ResNet-50** (+9.3 / +11.2, p ≤ 0.0003 on every seed and view count);
  - **not met against a fine-tuned ResNet-50:** +2.4 / +5.1 points, significant in 3 of 6 seed-and-view comparisons.
- The fine-tuned model is the fair baseline and the one reviewers will ask for. The claim to make is: *the watermark hybrid beats a fine-tuned ResNet-50 on unseen prints by about 5 points at six views, with far fewer missed counterfeits and much lower seed variance.*
- **The prefix network alone does not beat the fine-tuned ResNet-50 on unseen prints** (pooled p = 0.06 at one view in ResNet's favour; 0.65 at six views). Its advantage is robustness to view count, not print generalisation. The watermark is what generalises.
