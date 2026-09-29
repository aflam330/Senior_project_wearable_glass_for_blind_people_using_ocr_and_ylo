# All problems found and fixed, 2026-09-29 to 2026-09-30

Earlier rows (1–21) are in `CORRECTIONS.md` and `FINAL_SCAN.md`. This page lists this round.

| # | Problem | Fix or answer | Result | File |
|---|---|---|---|---|
| 22 | JaalTaka split is not print-disjoint (shared counterfeit serials) | Serial-disjoint split built; prefix network retrained on it | 98.2 → 90.1 % at 1 view on unseen prints | `SERIAL_FIX.md` |
| 23 | No counterfeit cue that transfers to new prints | Back-lit watermark-window detector plus a validation-fitted combination | 87.8 → 95.5 % at 6 views on unseen prints, p = 0.0002 | `SERIAL_WATERMARK_DETECTOR.md` |
| 24 | Serial blacklist proposed as a detector | Measured | 0 / 19 unseen-serial counterfeits caught; redundant once the watermark is used | same |
| 25 | Only 208 test notes | 5-fold CV over all 1,390 notes | 98.3 / 95.8 % (note / serial folds, 1 view) | `EXPANDED_TEST.md` |
| 26 | No session IDs for a session-disjoint split | Replaced by serial-disjoint split and serial-grouped CV | as above | `SECOND_COUNTERFEIT.md` |
| 27 | Prefix fix failed for the concat head (MVP-N) | Four concat repairs and a mean-pool head tried | Pooling heads fixed; concat not | `FUSION_GENERAL.md` |
| 28 | Whole-note transfer | Seven approaches tabulated; synthetic whole notes (A) fail on real photos | Policy E stays | `SYNTHETIC_FIX.md` |
| 29 | `load_splits()` could not select another split | `JAALTAKA_SPLIT_DIR` override (default unchanged) | used by the serial-disjoint runs | `roboeye/camva/notes.py` |
| 30 | First watermark box captured the serial number | Box moved below the serial and checked visually | no digits in the crops | `scripts/eval/watermark_features.py` |
| 31 | Back-lit views failed registration at 25 inliers | Threshold 12 plus a window-coverage check | 1,261 / 1,390 registered | same |
| 32 | SUS / NASA-TLX scoring missing | Scorer with self-checks | self-checks pass | `scripts/eval/score_sus_tlx.py` |
| 33 | PAC-Bayes certificate unclear | Consolidated; the data-dependent-prior bound of 0.1127 is the non-vacuous certificate (for the half-data network) | — | `PACBAYES_FULL.md` |
| 34 | Whole-note serial OCR too slow on CPU | Moved to GPU, 150 photos per genuine set | — | `scripts/eval/serial_whole_note.py` |
