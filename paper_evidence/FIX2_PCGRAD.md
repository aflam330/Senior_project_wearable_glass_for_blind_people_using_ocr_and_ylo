# Fix 2: PCGrad

> **Correction 2026-09-28.** Q-DUIG accuracies in this file were computed before three evaluation bugs were fixed: NaN entropy for fully confident notes (scored as p = 0.5), a volume feature that changed when views were masked, and a missing view self-gate in the policy path. Every checkpoint was re-evaluated with the fixed code; see `WEAK_RESULTS_FIX.md`. Where those numbers differ from the ones below, they supersede them, and verdicts based on the old numbers should be re-read. The original text is kept unchanged below.

Seed 42. Two epochs from the saved full PRMVT checkpoint. For each auxiliary term, if its gradient has a negative dot product with the authentication gradient, that auxiliary gradient is projected onto the orthogonal complement of the authentication gradient. The mask-count cost has no parameter gradient, so it is skipped.

Test accuracy from `results/qduig/auxfix_pcgrad/seed42/test_views/views_1_to_6.json`:

| views | accuracy |
|---:|---:|
| 1 | 0.9663461538461539 |
| 2 | 0.9519230769230769 |
| 3 | 0.9615384615384616 |
| 4 | 0.9471153846153846 |
| 5 | 0.9615384615384616 |
| 6 | 0.9759615384615384 |

1-view is below 0.9711538461538461. Verdict: FAIL.

Logged auxiliary terms on the last epoch are nonzero (quality 0.08869824252338522, diversity 0.33324768178516834, info-gain 0.009996725218571463, cost 0.8085215528642862). Source: `results/qduig/auxfix_pcgrad/seed42/val_metrics.json`.
