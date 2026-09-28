# Fix 4: Kendall uncertainty weighting

> **Correction 2026-09-28.** Q-DUIG accuracies in this file were computed before three evaluation bugs were fixed: NaN entropy for fully confident notes (scored as p = 0.5), a volume feature that changed when views were masked, and a missing view self-gate in the policy path. Every checkpoint was re-evaluated with the fixed code; see `WEAK_RESULTS_FIX.md`. Where those numbers differ from the ones below, they supersede them, and verdicts based on the old numbers should be re-read. The original text is kept unchanged below.

Seed 42. Two epochs from the saved full PRMVT checkpoint. Each auxiliary term is weighted by `exp(-s) * λ * L + s`, with a learned scalar `s` per term, starting at 0.

Test accuracy from `results/qduig/auxfix_kendall/seed42/test_views/views_1_to_6.json`:

| views | accuracy |
|---:|---:|
| 1 | 0.9711538461538461 |
| 2 | 0.9759615384615384 |
| 3 | 0.9903846153846154 |
| 4 | 0.9855769230769231 |
| 5 | 0.9855769230769231 |
| 6 | 0.9807692307692307 |

1-view equals full PRMVT (0.9711538461538461) and is below `her_base` (0.9903846153846154). Verdict: PARTIAL.

Last-epoch auxiliary terms are nonzero. Source: `results/qduig/auxfix_kendall/seed42/val_metrics.json`.
