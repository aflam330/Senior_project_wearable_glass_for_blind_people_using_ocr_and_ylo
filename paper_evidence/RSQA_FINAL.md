# RSQA final

> **Correction 2026-09-28.** Q-DUIG accuracies in this file were computed before three evaluation bugs were fixed: NaN entropy for fully confident notes (scored as p = 0.5), a volume feature that changed when views were masked, and a missing view self-gate in the policy path. Every checkpoint was re-evaluated with the fixed code; see `WEAK_RESULTS_FIX.md`. Where those numbers differ from the ones below, they supersede them, and verdicts based on the old numbers should be re-read. The original text is kept unchanged below.

The 1-view gap between full PRMVT (0.9711538461538461) and `her_base` (0.9903846153846154) is not removed by swapping only the fusion or only the auxiliary loss weights. Those two runs are in `RSQA_VS_MEANPOOL_ANALYSIS.md`: mean-pool with the auxiliary losses kept scores 0.9663461538461539, and RSQA with the auxiliary weights at 0 scores 0.9375.

The setting that keeps auxiliary training and matches `her_base` is a separate tower. The authenticator is the `her_base` checkpoint. Its weights and batch-norm statistics are frozen. Quality, diversity, and information-gain heads train on detached features and are not used at inference.

Test accuracy, seed 42, n=208: `results/qduig/auxfix_separate_frozen/seed42/test_views/views_1_to_6.json`

| views | accuracy |
|---:|---:|
| 1 | 0.9903846153846154 |
| 2 | 0.9711538461538461 |
| 3 | 0.9471153846153846 |
| 4 | 0.9086538461538461 |
| 5 | 0.9759615384615384 |
| 6 | 0.9182692307692307 |

These six numbers match `her_base`. The last-epoch tower losses are nonzero (quality 0.08368236527183462, diversity 0.009134438387988407, information gain 0.014033642198768057). Source: `results/qduig/auxfix_separate_frozen/seed42/val_metrics.json`.

This is the final RSQA result: auxiliary losses stay on a frozen side tower, and the authenticator stays the mean-pool `her_base` model.
