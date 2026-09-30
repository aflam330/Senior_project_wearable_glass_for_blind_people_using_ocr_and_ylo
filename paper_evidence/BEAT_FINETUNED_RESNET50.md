# Beating the full-resolution fine-tuned ResNet-50 (2026-09-30)

**Target:** +5 points over the full-resolution fine-tune, p < 0.001, three seeds, serial-disjoint test.
**Result:** not met. No new training run in this pass beat 94.9 ± 1.8 % / 94.4 ± 2.6 %.

The fine-tune is `results/serial_split/ft_resnet50_fullres/`. The watermark hybrid is `results/watermark/hybrid_final_seeds.json`. The four-feature combiner is `results/serial_split/hybrid_ft_fullres.json`.

| Approach | What was actually run | Six-view result | Versus the full-resolution fine-tune |
|---|---|---|---|
| A. Attention fusion with a watermark token | `beat_resnet.py`, seeds 42–44, before this fine-tune | 90.5 ± 0.0 % | Worse |
| B. Serial blacklist / duplicate serial | `SERIAL_WATERMARK_DETECTOR.md` | 0 of 19 unseen prints | No signal on new prints (Proposition 12) |
| C. Attention fusion, views only | same file | 88.0 ± 0.9 % | Worse |
| D. Fine-tune itself | the baseline | 94.4 ± 2.6 % | — |
| E. Uncertainty-weighted ensemble of network, fusion, watermark | `beat_resnet.json` | 93.1 ± 0.3 % | Below the fine-tune and below the hybrid |
| F–G. Contrastive or meta-learning over prints | not run | — | The test has about 24 effective prints. A new loss fit on those prints and then scored on the same test would be tuning on the test |
| H. Network + watermark + fine-tune | `fuse_ft_fullres.py`, combiner fit on validation only | 94.6 ± 0.5 % | Not better. Per-seed McNemar against the three-feature hybrid: p = 1.0, 0.5, 1.0 |
| I. Multi-scale watermark pyramid | not run | — | Would require new crops and a new test read of a model chosen after the current test has already been read many times |
| J. A new joint loss | not run | — | Same reason as F–G |

The watermark hybrid at six views is 95.0 ± 0.0 %, about half a point above the fine-tune mean, with per-seed exact McNemar p = 0.38, 1.0, and 0.057. That is not +5 points and not p < 0.001.

Seed 44 of the fine-tune is 91.4 % at six views because validation selected epoch 1. The hybrid does not have that swing. Stability is the hybrid's measured advantage. Accuracy is not.
