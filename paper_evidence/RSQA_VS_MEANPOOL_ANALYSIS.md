# RSQA versus mean-pool

> **Correction 2026-09-28.** Q-DUIG accuracies in this file were computed before three evaluation bugs were fixed: NaN entropy for fully confident notes (scored as p = 0.5), a volume feature that changed when views were masked, and a missing view self-gate in the policy path. Every checkpoint was re-evaluated with the fixed code; see `WEAK_RESULTS_FIX.md`. Where those numbers differ from the ones below, they supersede them, and verdicts based on the old numbers should be re-read. The original text is kept unchanged below.

Seed 42. Test set, n=208. Checkpoint chosen by validation mean accuracy over 1–6 views.

Two saved runs already occupy two cells:

| fusion | auxiliary losses | 1-view | source |
| --- | --- | ---: | --- |
| RSQA | on | 0.9711538461538461 | `results/qduig/prefix_ft/seed42/test_views/views_1_to_6.json` |
| mean-pool | off, and the quality, diversity, information-gain, cost, and redundancy modules are off | 0.9903846153846154 | `results/qduig/ablation_prefix/her_base/seed42/test_views/views_1_to_6.json` |

The 1-view gap is 0.019230769230769273.

`use_quality: true` both turns on RSQA fusion and keeps the quality loss. `her_base` also sets multitask and contrastive weights to 0. The two new runs change one factor inside the full PRMVT recipe (multitask 0.5, contrastive 0.1 then 0.05, 6 epochs then 3).

## Mean-pool, auxiliary losses kept

`force_mean_pool: true`. Quality, uncertainty, diversity, information-gain, and cost losses stay at the full weights. RSQA is not used for fusion.

`results/qduig/fusion_meanpool_aux_ft/seed42/test_views/views_1_to_6.json`

| views | accuracy |
|---:|---:|
| 1 | 0.9663461538461539 |
| 2 | 0.9759615384615384 |
| 3 | 0.9807692307692307 |
| 4 | 0.9759615384615384 |
| 5 | 0.9759615384615384 |
| 6 | 0.9807692307692307 |

1-view is below full PRMVT. Replacing RSQA with mean-pool, while the auxiliary losses stay on, does not close the gap.

## RSQA, auxiliary loss weights set to 0

The modules stay in the forward pass, including RSQA. The five auxiliary loss weights are 0. Multitask and contrastive stay at the full-recipe values.

`results/qduig/fusion_rsqa_noaux_ft/seed42/test_views/views_1_to_6.json`

| views | accuracy |
|---:|---:|
| 1 | 0.9375 |
| 2 | 0.9663461538461539 |
| 3 | 0.9711538461538461 |
| 4 | 0.9759615384615384 |
| 5 | 0.9759615384615384 |
| 6 | 0.9711538461538461 |

1-view is below both full PRMVT and the mean-pool run. Turning the auxiliary loss weights off, while leaving RSQA in the forward pass, does not close the gap.

## What the four cells say

Neither switch inside the shared full recipe reaches 0.9903846153846154. Mean-pool with the auxiliary losses on scores 0.9663461538461539. RSQA with those loss weights at 0 scores 0.9375. The 0.9903846153846154 result is the `her_base` forward, where those modules are off and multitask and contrastive are 0.

The separate auxiliary tower with that forward frozen still scores 0.9903846153846154 at 1 view, with nonzero tower losses (`AUXILIARY_FIX_RESULTS.md`). That is auxiliary training beside the authenticator, not RSQA inside it.

An older quality-only prefix run (`rsqa` in `ABLATION_RESULTS.md`) scored 0.9327 at 1 view. That row is a different recipe and is not one of these two cells.

Verdict against 1-view ≥ 0.9904 with the full shared auxiliary stack: FAIL.
