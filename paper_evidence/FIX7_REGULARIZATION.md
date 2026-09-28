# Fix 7: auxiliary loss on every fourth batch

> **Correction 2026-09-28.** Q-DUIG accuracies in this file were computed before three evaluation bugs were fixed: NaN entropy for fully confident notes (scored as p = 0.5), a volume feature that changed when views were masked, and a missing view self-gate in the policy path. Every checkpoint was re-evaluated with the fixed code; see `WEAK_RESULTS_FIX.md`. Where those numbers differ from the ones below, they supersede them, and verdicts based on the old numbers should be re-read. The original text is kept unchanged below.

Seed 42. Two epochs from the saved full PRMVT checkpoint. Auxiliary weights are applied when the batch index modulo 4 is 0, and set to 0 on the other batches.

Test accuracy from `results/qduig/auxfix_regularize/seed42/test_views/views_1_to_6.json`:

| views | accuracy |
|---:|---:|
| 1 | 0.9471153846153846 |
| 2 | 0.9375 |
| 3 | 0.9423076923076923 |
| 4 | 0.9471153846153846 |
| 5 | 0.9663461538461539 |
| 6 | 0.9711538461538461 |

1-view is below 0.9711538461538461. Verdict: FAIL.

The other two schedules in the hypothesis (probability 0.25, and auxiliary loss only on epochs 4–6) were not trained. The curriculum run covers a delayed schedule and is reported in `FIX5_CURRICULUM.md`.
