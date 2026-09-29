# Unified Bangladeshi Taka benchmark (2026-09-30)

Script: `realtime_bangla_taka_detection/scripts/eval/make_unified_bdt.py` → `results/unified_bdt/manifest.json` (one row per unit, with task, label and leakage-aware split) and `summary.json`. No image was copied or created; the manifest points at the datasets on disk.

## Size

| Task | Dataset | Units | Split used | Leakage handling |
|---|---|---:|---|---|
| Counterfeit, multi-view (6 views) | JaalTaka | 1,390 notes (8,340 images) | **Print-disjoint** (222 test notes, 101 counterfeit) and note-disjoint (seed 42) | Counterfeit prints grouped by OCR serial; no test print in train |
| Counterfeit, whole note | Counterfeit Currency Image Dataset | 1,286 images (87 counterfeit, about 4 physical notes) | Test only | Bursts and augmented copies grouped |
| Denomination | NSTU-BDTAKA Recognition | 28,875 | Its own train / validation / test | 88 test originals that also occur in train flagged |
| Denomination | BanglaTaka raw | 5,073 | Training source only | Byte-identical "Diverse" copy excluded |
| Denomination | Bangla Money Training | 1,536 | Test only | Never used for training or selection |
| Detection | NSTU-BDTAKA Detection | 3,111 | Its own splits | — |
| Open-set unknowns | Large Scale BDT DB 2026 (coins, demonetized notes) | 100,000 | Validation unknowns | — |
| Open-set unknowns | Bangla Money 1-taka | 101 | Test unknowns | — |
| **Total** | | **141,372 units, 148,322 images** | | |

## What this dataset can and cannot support

- **Denomination, detection and open-set:** tens of thousands of images across five sources, with test sets independent of training (Bangla Money, and NSTU test minus leaked originals). This is the large-scale part.
- **Counterfeit detection:** limited to the two labelled sources, 2,676 counterfeit-labelled units in total.
  - Only 101 test counterfeits come from prints unseen in training, on about 20 small prints.
  - A 5,000-note counterfeit test set cannot be built from the data on disk. The other datasets have no genuine / counterfeit labels, and inventing labels would be fabrication.
  - The largest honest counterfeit evaluation available is the five-fold serial-grouped cross-validation over all 1,390 JaalTaka notes (`EXPANDED_TEST.md`).

## Existing results on each task

| Task | Result | Source |
|---|---|---|
| Denomination, Bangla Money (1,536) | 91.5 % | `CROSS_DATASET_TAKA.md` |
| Denomination, NSTU hand-held close-ups | 18.5 % | same |
| Detection, NSTU (186 test) | AP50 0.796 | same |
| Open-set, 1-taka notes announced as known | 43.6 % → 22.8 % with the validation-chosen threshold | `OPEN_SET_REJECTION.md` |
| Counterfeit, unseen prints | 94.4 / 95.0 % (hybrid, 3 seeds) | `SERIAL_WATERMARK_DETECTOR.md` |
| Counterfeit, all notes (serial-grouped 5-fold) | 95.8 / 98.9 % (1 / 6 views, probe) | `EXPANDED_TEST.md` |
