# View-count shift across datasets (2026-09-30)

Fixed-view versus prefix training, in % (one view / six views).

| Dataset | Kind | Model | Fixed | Prefix | Source |
|---|---|---|---|---|---|
| JaalTaka (seed-42 split, 3 seeds) | Real banknote photos, 6 ordered views | Same network, shared BN | 71.2 / 98.1 | 98.2 / 98.7 | `SAME_ARCH_RESULTS.md` |
| JaalTaka (serial-disjoint split, 1 seed) | Unseen counterfeit prints | Prefix network, shared BN | — | 90.1 / 87.8 | `SERIAL_FIX.md` |
| MVP-N (3 seeds) | Real object photos, 2–6 views | Attention pooling | 40.7 / 75.8 | 48.0 / 82.2 | `VCDS_MVPN.md` |
| MVP-N (3 seeds) | same | Mean pooling | 47.1 / 75.2 | 49.5 / 77.2 | `FUSION_GENERAL.md` |
| MVP-N (3 seeds) | same | Concatenation | 48.7 / 80.5 | 49.7 / 71.9 | same |
| ModelNet40, 10 classes | Rendered objects | 10-class render network | 64 / 77 | 64 / 77 (higher at 2–3 views) | `VCDS_UNIVERSAL.md` |
| NSTU-BDTAKA | One photo per note | — | not applicable | — | `NSTU_FULL.md` |

**Reading.**
- View-count shift appears whenever a joint fusion network is trained on a fixed view count and tested on fewer views: JaalTaka, and MVP-N with pooling heads.
- Prefix training removes it for pooling fusion on both real datasets.
- It does not help slot concatenation.
- It makes no difference on the small rendered ModelNet subset.
- NSTU cannot test it: one photo per note.
