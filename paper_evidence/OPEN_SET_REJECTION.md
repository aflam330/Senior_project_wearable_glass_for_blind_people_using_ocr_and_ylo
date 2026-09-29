# Unknown-note rejection for the Taka detector (2026-09-29)

**Problem** (`CROSS_DATASET_TAKA.md`). The detector has no "unknown note" output. At the app's default confidence of 0.25, 44 of 101 one-taka notes (a denomination it was not trained on) were announced as a known value, 37 of them as 5 taka.

## Rule, fixed before test scoring

Script: `realtime_bangla_taka_detection/scripts/eval/eval_open_set.py`.

- **Score:** confidence of the top YOLO box.
- **Answer:** announce the top box's class only if the score ≥ τ.
- **Choice of τ:** grid 0.25–0.95 in steps of 0.05, maximising (correct − wrong) on validation. Any announcement for an unknown image counts as wrong.
- **Validation pool,** never used to train or select the detector:
  - known: NSTU-BDTAKA Recognition/validation (1,624 images)
  - unknown: 600 demonetized notes plus 100 each of 1-, 2- and 5-taka coins (Large Scale BDT DB, seed 42)
- **Test,** scored once after τ was fixed:
  - known: Bangla Money, 8 denominations (1,536)
  - unknown: Bangla Money one-taka notes (101)

**Chosen τ = 0.60.** Validation utility peaks there at 135.

## Test result

Source: `results/open_set/summary.json`.

| | τ = 0.25 (previous app) | τ = 0.60 (chosen) |
|---|---:|---:|
| Known notes correct | 1,405 / 1,536 (91.5 %) | 1,376 (89.6 %) |
| Known notes wrong | 71 | 50 |
| Known notes, no answer | 60 | 110 |
| Wrong share of answered known notes | 4.8 % | 3.5 % |
| One-taka notes announced as a known value | 44 / 101 (43.6 %; Wilson 34.3–53.3 %) | **23 / 101 (22.8 %; Wilson 15.7–31.9 %)** |
| Correct − wrong (test) | 1,290 | 1,303 |

## Reading

- Rejection halves the unknown-note announcements and lowers wrong answers on known notes. The cost is 29 more correct answers withheld.
- 22.8 % of one-taka notes are still announced as another value, so the detector is not open-set safe.
- **Caveat on the validation unknowns.** They were easier than one-taka notes: no demonetized note or coin was announced at τ ≥ 0.45. The choice of τ was therefore driven mainly by the right/wrong trade-off on known NSTU close-ups.
- **Better validation unknowns** would be one-taka or other out-of-scope notes that are not in the test set. None are on disk.

## Deployed

- `savior_glass/config.py`: `CURRENCY_ANNOUNCE_CONF = 0.60`.
- `savior_glass/modes/currency_mode.py`: below that confidence the glass says "নোট নিশ্চিত করা যায়নি। আরও কাছে ধরুন।" ("Could not confirm the note. Hold it closer.").
- `scripts/test_buttons_haptics.py` still passes.
