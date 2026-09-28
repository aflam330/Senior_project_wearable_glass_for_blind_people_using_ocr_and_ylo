# Fix 5: curriculum auxiliary loss

> **Correction 2026-09-28.** Q-DUIG accuracies in this file were computed before three evaluation bugs were fixed: NaN entropy for fully confident notes (scored as p = 0.5), a volume feature that changed when views were masked, and a missing view self-gate in the policy path. Every checkpoint was re-evaluated with the fixed code; see `WEAK_RESULTS_FIX.md`. Where those numbers differ from the ones below, they supersede them, and verdicts based on the old numbers should be re-read. The original text is kept unchanged below.

Seed 42. Six epochs from the saved full PRMVT checkpoint. The authenticator forward is the full model, including RSQA.

Schedule that was run:

| epochs | auxiliary weight scale |
|---:|---:|
| 1–2 | 0 |
| 3–4 | 0.01 |
| 5–6 | 0.1 |

Epochs 7–9 at the full auxiliary weight were not run.

Test accuracy from `results/qduig/auxfix_curriculum/seed42/test_views/views_1_to_6.json`:

| views | accuracy |
|---:|---:|
| 1 | 0.9807692307692307 |
| 2 | 0.9759615384615384 |
| 3 | 0.9807692307692307 |
| 4 | 0.9759615384615384 |
| 5 | 0.9807692307692307 |
| 6 | 0.9807692307692307 |

The selected epoch is 6, by validation mean 1–6. 1-view is above full PRMVT (0.9711538461538461) and below `her_base` (0.9903846153846154). Verdict: PARTIAL.

This is the highest 1-view score among the runs that keep the full PRMVT forward path.
