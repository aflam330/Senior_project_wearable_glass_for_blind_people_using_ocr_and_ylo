# NSTU-BDTAKA, all measurements in one place (2026-09-30)

NSTU-BDTAKA (Mendeley) has a detection part (boxes labelled "Taka", no denomination) and a recognition part (9 denominations, 256 × 256 hand-held close-ups). It has no genuine/counterfeit labels and one photo per note. No NSTU image was used to train or select the deployed detector.

| Question | Result | Source |
|---|---|---|
| Detection, class-agnostic AP50 (186 test images) | **0.796**; at confidence 0.25: precision 0.832, recall 0.694, IoU of matches 0.820 | `CROSS_DATASET_TAKA.md` |
| Denomination on hand-held close-ups (1,144) | **18.5 %** correct; no note found in 66.9 %; 55.9 % correct when found | same |
| Hand-held robustness with a close-up fallback (YOLO first, then a classifier trained on NSTU train) | NSTU test 85.0 % (1,039, after removing 88 leaked originals); but Bangla Money only 93.1 %, and the classifier alone 48.2 % on Bangla Money | same |
| Unknown-note detection | NSTU validation images are the "known" pool that fixes the rejection threshold (τ = 0.60). On Bangla Money, one-taka notes announced as a known value: 43.6 % → 22.8 % | `OPEN_SET_REJECTION.md` |
| Leakage inside NSTU | 88 of the 1,127 original recognition-test photos also occur in NSTU train; photo IDs are sequential video frames | `CROSS_DATASET_TAKA.md` |
| View-count shift | Not measurable: one photo per note, no multi-view structure | — |
| Counterfeit detection | Not measurable: no genuine/counterfeit labels | — |

**Reading.**
- The detector finds whole notes in independent photos (AP50 0.796) but fails on folded hand-held close-ups (18.5 %).
- A close-up classifier trained on NSTU scores well on NSTU and badly elsewhere. It learned NSTU's recording conditions, so it stays out of the app.
