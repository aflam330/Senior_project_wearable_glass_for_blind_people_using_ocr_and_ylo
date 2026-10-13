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

## Addendum, 2026-10-13: three more datasets, one module each

Earlier sections are unchanged. No model was retrained for these.

| Dataset | Module | Result | File |
|---|---|---|---|
| UV photographs of genuine 200, 500, 1,000 Taka notes (118, made in this project) | Taka detector under UV light | Note box found on 111 of 118. The value it names is often wrong under UV (1,000 Taka: 11 of 30), so the value has to be read in normal light first | `savior_glass/results/uv_check/uv_fibre_check.json` |
| ICDAR 2019 MLT word images (1,000 real words) | Text reader | CER 39.2 % Bangla, 62.5 % Latin script, against 2.5 % on the rendered benchmark | `OCR_REAL.md` |
| COCO val2017, all 5,000 photographs | Object detector | YOLOv8n 0.453, YOLOv8s 0.551, YOLOv8m 0.607 mAP@0.5 | `OBJECT_IMPROVED.md` |

Not run again in this pass: the Taka detector on Bangla Money and NSTU-BDTAKA, PRMVT on JaalTaka, and the view-count experiments on MVP-N and ModelNet. Their results are in the sections above and nothing that affects them has changed.
