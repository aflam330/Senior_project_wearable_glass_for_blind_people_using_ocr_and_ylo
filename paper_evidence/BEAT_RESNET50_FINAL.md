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
  - **met against the frozen ResNet-50** (+9.3 / +11.2 for the hybrid, p ≤ 0.0003 on every seed and view count);
  - **not met against the quarter-resolution fine-tune below, and not met against the full-resolution fine-tune in the next section.**
- The prefix network alone does not beat a fine-tuned ResNet-50 on unseen prints. Its advantage is robustness to view count. The watermark is what generalises past a weak image baseline. Once the fine-tune sees full-resolution views, that gap closes.

## Full-resolution fine-tune (2026-09-30, after the cache fix)

`cache/views256` was built with `cv2.IMREAD_REDUCED_COLOR_4`, so a 1672×1929 view was decoded at about 418×483 and then resized. The published fine-tune above used that cache. `scripts/train/cache_views_256.py` now writes `cache/views256_full` from the full JPEG. The new run uses that cache, the same frozen-layer recipe, 8 epochs, and the epoch with the best validation mean. It does not overwrite `ft_resnet50/seed*.json`.

Sources: `results/serial_split/ft_resnet50_fullres/seed{42,43,44}.json` and `results/serial_split/hybrid_ft_fullres.json`.

| Method | 1 view | 6 views |
|---|---:|---:|
| Frozen ResNet-50 probe | 85.1 % | 83.8 % |
| Fine-tune on the quarter-resolution cache (published) | 92.0 ± 2.3 % | 89.9 ± 4.4 % |
| **Fine-tune on full-resolution views** | **94.9 ± 1.8 %** | **94.4 ± 2.6 %** |
| Watermark hybrid (unchanged) | 94.4 ± 0.5 % | **95.0 ± 0.0 %** |
| Hybrid plus the full-resolution fine-tune | 94.9 ± 1.0 % | 94.6 ± 0.5 % |

Per-seed test accuracy of the full-resolution fine-tune: 93.2 / 94.6 / 96.8 % at one view, and 96.4 / 95.5 / 91.4 % at six views. Genuine notes called counterfeit: 1 of 121 on every seed. Counterfeits missed, of 101: 14, 11, 6 at one view and 7, 9, 18 at six views.

Exact McNemar, full-resolution fine-tune versus the frozen probe, per seed: one view p = 1.2×10⁻⁴, 5.7×10⁻⁶, 3.0×10⁻⁸; six views p = 5.8×10⁻⁸, 3.0×10⁻⁸, 1.5×10⁻⁵. Mean gain +9.8 / +10.7 points.

Versus the quarter-resolution fine-tune the gain is real on four of six seed-and-view comparisons (six views, seeds 42 and 43: p = 4.2×10⁻⁷ and 0.002). Seed 44 at six views is lower (91.4 % versus 93.7 %, p = 0.13).

Adding the full-resolution fine-tune to the watermark hybrid does not help. At six views the hybrid alone is 95.0 % and the four-feature combiner is 94.6 %, with per-seed McNemar p = 1.0, 0.5, 1.0. The watermark hybrid stays the six-view result to cite. The full-resolution fine-tune is the fair image baseline, and it already clears +8 points over the frozen probe by itself.
