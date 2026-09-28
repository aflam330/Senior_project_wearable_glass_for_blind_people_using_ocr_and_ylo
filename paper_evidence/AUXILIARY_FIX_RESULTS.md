# Auxiliary fix results

> **Correction 2026-09-28.** Q-DUIG accuracies in this file were computed before three evaluation bugs were fixed: NaN entropy for fully confident notes (scored as p = 0.5), a volume feature that changed when views were masked, and a missing view self-gate in the policy path. Every checkpoint was re-evaluated with the fixed code; see `WEAK_RESULTS_FIX.md`. Where those numbers differ from the ones below, they supersede them, and verdicts based on the old numbers should be re-read. The original text is kept unchanged below.

Full PRMVT 1-view is 0.9711538461538461. `her_base`, with auxiliary losses off, is 0.9903846153846154 at 1 view and 0.9182692307692307 at 6 views.

The loss logs do not show the auxiliary terms overpowering authentication after weighting. See `AUXILIARY_ANALYSIS.md`. Gradient norms and encoder feature drift are NOT_MEASURED.

| fix | 1-view | 6-view | verdict |
| --- | ---: | ---: | --- |
| Loss scale, best of 0.1 / 0.01 / 0.001 / 0.0001 | 0.9663461538461539 | 0.9759615384615384 | FAIL |
| PCGrad | 0.9663461538461539 | 0.9759615384615384 | FAIL |
| Stop-gradient | 0.9903846153846154 | 0.9086538461538461 | SUCCESS at 1 view |
| Kendall | 0.9711538461538461 | 0.9807692307692307 | PARTIAL |
| Curriculum, 6 epochs | 0.9807692307692307 | 0.9807692307692307 | PARTIAL |
| Separate tower, BN still updating | 0.9519230769230769 | 0.9134615384615384 | FAIL |
| Separate tower, BN frozen | 0.9903846153846154 | 0.9182692307692307 | SUCCESS |
| Every 4th batch | 0.9471153846153846 | 0.9711538461538461 | FAIL |

The two successes keep the `her_base` authenticator. Stop-gradient still trains that authenticator with the authentication loss, and 6-view accuracy falls to 0.9086538461538461. The frozen separate tower does not change the authenticator. Its 1–6 view accuracies match `her_base`, while quality, diversity, and info-gain losses on the tower are nonzero.

Curriculum is the best run that keeps the full PRMVT forward, including RSQA. Its 1-view accuracy is 0.9807692307692307. That is above full PRMVT and below `her_base`.

Details: `FIX1_LOSS_SCALE.md` through `FIX7_REGULARIZATION.md`.
