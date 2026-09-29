# Does prefix training fix every fusion design? (2026-09-30)

MVP-N, 44 classes, frozen ResNet-50 features, 3 seeds; epoch chosen on VALID; test at k = 1 and 6 (%).
Sources: `results/mvpn/vcds.json`, `fusion_fix.json`; scripts: `mvpn_vcds.py`, `mvpn_fusion_fix.py`.

| Fusion head | Fixed 6-view: k1 / k6 | Prefix: k1 / k6 | Prefix effect |
|---|---|---|---|
| Attention pooling | 40.7 / 75.8 | 48.0 / 82.2 | **better at both** |
| Mean pooling | 47.1 / 75.2 | 49.5 / 77.2 | **better at both** |
| Concatenation (zero-padded slots) | 48.7 / 80.5 | 49.7 / 71.9 | +1.0 / −8.6 |
| Concat + view mask | 47.1 / 78.8 | 48.6 / 70.7 | +1.5 / −8.1 |
| Concat + view-count one-hot | 46.7 / 78.4 | 48.5 / 69.4 | +1.8 / −9.0 |
| Concat, slots rescaled by 6/k | 48.6 / 80.5 | 50.9 / 76.1 | +2.3 / −4.4 |
| Concat, prefix with k = 6 half the time | — | 49.5 / 76.4 | vs fixed: +0.8 / −4.1 |
| Per-view reference (no fusion) | — | 55.3 / 84.7 | — |

**Finding.** Prefix training improves **pooling** fusion (attention, mean) at every view count. For **slot concatenation** it always trades full-view accuracy for fewer-view accuracy, and none of four repairs removes that trade. JaalTaka's PRMVT uses attention-style pooling, which is consistent with its gains there.

**Design rule for the paper:** use a pooling fusion head and train it on view prefixes; avoid slot concatenation when the view count varies.
