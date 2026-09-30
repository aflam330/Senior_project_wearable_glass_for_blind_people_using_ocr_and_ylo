# Cross-dataset status (2026-09-30)

No new cross-dataset run was started. The measurements already on disk are in `CROSS_DATASET_TAKA.md` and `LARGE_SCALE_DATASET.md`. None of NSTU-BDTAKA, BanglaTaka, Bangla Money, or Large Scale BDT DB 2026 labels genuine versus counterfeit, so they cannot validate a counterfeit detector.

| Check | Result | Source |
|---|---|---|
| Denomination, Bangla Money, 1,536 photos, no retraining | 91.5 % | `CROSS_DATASET_TAKA.md` |
| Denomination, NSTU hand-held close-ups | 18.5 % | same |
| Detection, NSTU test, 186 images | AP50 0.796 | same |
| BanglaTaka "Diverse" folder | not an external test; byte-identical to training composites | same |
| Large Scale BDT DB 2026 | coins and a demonetized class; not the nine notes | same |
| Counterfeit, unseen JaalTaka prints | hybrid 95.0 ± 0.0 % at 6 views | `hybrid_final_seeds.json` |
| VCDS, JaalTaka prefix network | 98.2 ± 0.7 % at 1 view, 98.7 ± 0.3 % at 6 views on the note-disjoint split | `SAME_ARCH_RESULTS.md` |

A watermark model was not run on those external sets: they are not back-lit watermark photographs.
