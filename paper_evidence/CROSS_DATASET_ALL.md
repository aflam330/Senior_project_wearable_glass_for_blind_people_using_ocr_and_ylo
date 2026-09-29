# Cross-dataset validation, all Bangladeshi datasets on disk (2026-09-29)

The Taka detector (`realtime_bangla_taka_detection/models/best.pt`) was trained only on composites built from BanglaTaka raw crops and COCO backgrounds. None of the sets below was used to train or select it. Intervals are 95 % Wilson intervals over images.

## Denomination (detector top box, confidence 0.25)

| Set | n | Correct denomination | 95 % CI | Source |
|---|---:|---:|---|---|
| Bangla Money (Kaggle), 8 denominations | 1,536 | 91.5 % | 90.0–92.8 % | `results/external/external_taka.json` |
| Counterfeit-currency dataset, all images | 1,286 | 92.6 % | 91.1–93.9 % | `results/external/detector_counterfeit_ds.json` |
| – 500 BDT genuine | 829 | 92.4 % | 90.4–94.0 % | same |
| – 1000 BDT genuine | 370 | 100 % | 99.0–100 % | same |
| – 1000 BDT counterfeit | 55 | 100 % | 93.5–100 % | same |
| – 500 BDT counterfeit | 32 | **0 %** | 0–10.7 % | same |
| NSTU-BDTAKA hand-held close-ups | 1,144 | 18.5 % | 16.4–20.9 % | `results/external/external_taka.json` |
| NSTU-BDTAKA detection test (box only, class "Taka") | 186 | AP@0.5 0.796 (class-agnostic) | — | same |

**The 500 BDT counterfeit images** are 32 `augmented_*.png` files: the dataset's own augmented copies of a few tightly cropped, partial notes. Of these, 19 are read as 5 taka and 12 are not detected. This is the same failure as on NSTU close-ups (part of a note filling the frame), not a counterfeit-specific effect. The counterfeit photos of whole 1000 BDT notes are all read correctly.

## Sets not used, and why

- **"A Diverse Image Dataset for Bangladeshi Currency Recognition":** byte-identical (MD5) to the training source (`CROSS_DATASET_TAKA.md`).
- **Large Scale BDT DB (coins, demonetized notes):** outside the 9 denominations. It is used only as validation unknowns for open-set rejection (`OPEN_SET_REJECTION.md`).
- **Bangla Money "Testing" folder:** it has no labels.

## PRMVT on these sets

PRMVT is a genuine/counterfeit model. It has no denomination output, so "PRMVT denomination accuracy" is not defined.

Its genuine/counterfeit behaviour on whole-note photos from these sets is in `JAAL_VERDICT_FIXED.md`: wrong on 20–59 % of genuine notes. That is why the glass verdict was switched off.

**Update, 2026-09-29 evening: counterfeit transfer to whole notes, 500 / 1,000 BDT.** Source: `results/jaal_whole/summary.json`. Rates are genuine photos called counterfeit at p = 0.5.

| Checker | Counterfeit-set genuine (1,194) | Bangla Money genuine (288) | BanglaTaka genuine (1,657) | AUC, counterfeit-set originals |
|---|---:|---:|---:|---:|
| PRMVT, whole crop as 1 view | 27.1 % | 45.1 % | 61.5 % | 0.640 |
| PRMVT, 4 views cut from the crop | 5.6 % | 43.4 % | 60.2 % | 0.829 |
| ResNet-50 probe, 2 cut views | 1.6 % | 12.5 % | 16.5 % | 0.966 |
| Probe trained on the counterfeit set | 0.1 % | 26.4 % | 86.7 % | 0.886 |

The counterfeit side of the set has about four physical notes and is separable from file metadata alone, so it cannot certify a checker. The glass now runs a policy that never says "counterfeit". On JaalTaka test notes it passed 0 / 88 counterfeits, and on whole-note photographs 0 / 25 (`JAAL_VERDICT_FIXED.md`).

NSTU-BDTAKA, Bangla Money, BanglaTaka and Large Scale BDT have no counterfeit labels, so counterfeit accuracy is measurable only on JaalTaka and the counterfeit set.

## Multi-view datasets (fixed-view vs prefix)

The only multi-view sets on disk are JaalTaka (6 ordered views per note) and ModelNet40 (rendered to 6 views).

- **ModelNet40** (three seeds, 184 test objects): `VCDS_UNIVERSAL.md`.
- **JaalTaka, same architecture:** `SAME_ARCH_RESULTS.md`. Three seeds, finished 2026-09-29. With shared BN at 1 view, fixed-view training scores 71.2 ± 11.4 % and prefix training 98.2 ± 0.7 %. Prefix wins on every seed (p ≤ 9.3 × 10⁻¹⁰). No difference at 6 views.
- **MVP-N:** not on disk (not downloaded); not measured.

No other dataset on disk has multiple aligned views of one object. Foreign-currency sets: none on disk.
