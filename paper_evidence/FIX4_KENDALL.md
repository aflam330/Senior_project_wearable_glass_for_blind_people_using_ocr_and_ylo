# Fix 4: Kendall uncertainty weighting

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
