# New test sets and selection pressure (2026-09-29)

## Detector: fresh composites never seen by any decision

**Script:** `realtime_bangla_taka_detection/scripts/eval/make_new_test_set.py`.

**How the set was built**
- **Foregrounds:** only the BanglaTaka photos that the original generator placed in the **test** split (recovered from the test file names). No training or validation source photo is reused.
- **Backgrounds:** **COCO train2017**. Training used val2017 only.
- **Seed:** new base 900000.
- **Size:** one composite per source photo (513 notes) plus 53 background-only negatives. Stored in `data set/currency_yolo_newtest/`.

**Scoring.** `models/best.pt` was scored once, with no change to the model or thresholds (`results/new_test/detector_newtest.json`).

| Set | Images | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|
| Original composite test | 2,282 | 0.995 | 0.996 | 0.995 | 0.849 |
| **New composite test** | 566 | 0.996 | 0.997 | 0.992 | 0.848 |

The two agree, so the composite score does not depend on the particular test backgrounds or seeds. Both are still **synthetic**. The real-photo tests are in `CROSS_DATASET_ALL.md`: 91.5 % on Bangla Money and 18.5 % on NSTU close-ups.

## JaalTaka: no unused notes

All 1,390 notes on disk are in the 974 / 208 / 208 split (`split_metadata.json`), so there is no untouched JaalTaka note left to build a fresh authentication test from.

The whole-note counterfeit-currency dataset is the only other genuine/counterfeit source on disk. It has already been used as an out-of-domain test (`JAAL_VERDICT_FIXED.md`).

**What would create one.** A fresh set of notes photographed in the JaalTaka six-view protocol, ideally with a different phone and on a different day. That is a data-collection task.

## Selection pressure on the JaalTaka test split

About **110 distinct runs** have written test results for the same 208 test notes under `results/` (excluding the new same-architecture runs). That covers the novel algorithms, ablations, fixes, seeds and robustness fine-tunes.

**How much the best of them could be inflated**
- The standard error of one accuracy near 0.97 on 208 notes is √(0.97 · 0.03 / 208) ≈ 0.012.
- If one picked the maximum of about 110 runs by test score, the expected optimism for independent runs would be up to about 2.5 standard errors, or roughly 3 points.
- The runs are highly correlated, since they share the encoder and the split, so the true inflation is smaller. It is not zero.

**Consequences**
- **PRMVT (the headline):** its checkpoint and epoch were fixed by a validation rule written before its test scoring (`FINAL_RESULTS.md`). It was not chosen as the best test row. Its 3-seed mean (96.5 % at 1 view, 98.1 % at 6) is less exposed than any single-seed maximum.
- **Other rows:** any statement of the form "variant X reached 99.0 % at 1 view", for example `her_base`, is a best-of-many observation on a reused test split. It should not be presented as a result without a fresh test set.
- **Same-architecture runs** (`SAME_ARCH_RESULTS.md`): these use a design fixed in advance (three arms, three seeds, validation-selected checkpoints). They are reported in full rather than selected.
