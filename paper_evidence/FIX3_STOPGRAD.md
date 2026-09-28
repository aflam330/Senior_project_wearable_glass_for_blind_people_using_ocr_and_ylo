# Fix 3: stop-gradient on auxiliary heads

> **Correction 2026-09-28.** Q-DUIG accuracies in this file were computed before three evaluation bugs were fixed: NaN entropy for fully confident notes (scored as p = 0.5), a volume feature that changed when views were masked, and a missing view self-gate in the policy path. Every checkpoint was re-evaluated with the fixed code; see `WEAK_RESULTS_FIX.md`. Where those numbers differ from the ones below, they supersede them, and verdicts based on the old numbers should be re-read. The original text is kept unchanged below.

Seed 42. Two epochs from the saved `her_base` checkpoint. The authenticator stays on the `her_base` forward (mean-pool, no RSQA). Quality, uncertainty, and the information-gain module are trained on detached features. Diversity and cost stay at 0 in the log because `use_diversity` and `use_cost` stay off. Turning those flags on would change the authenticator.

Last-epoch training terms (`results/qduig/auxfix_stopgrad/seed42/val_metrics.json`): quality 0.3310413922984497, uncertainty 0.06115470142859116, info-gain 3.619767900927325e-05, diversity 0.0, cost 0.0. Authentication loss is 0.17874407626578998 and still updates the shared encoder.

Test accuracy from `results/qduig/auxfix_stopgrad/seed42/test_views/views_1_to_6.json`:

| views | accuracy |
|---:|---:|
| 1 | 0.9903846153846154 |
| 2 | 0.9375 |
| 3 | 0.9182692307692307 |
| 4 | 0.9086538461538461 |
| 5 | 0.9326923076923077 |
| 6 | 0.9086538461538461 |

1-view equals `her_base` (0.9903846153846154). Verdict on the 1-view target: SUCCESS.

6-view is 0.9086538461538461. `her_base` 6-view is 0.9182692307692307. The extra authentication updates lowered 6-view accuracy.
