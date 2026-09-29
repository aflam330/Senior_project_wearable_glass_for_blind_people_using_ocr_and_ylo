# Prefix stopping policy

> **Correction 2026-09-28.** Q-DUIG accuracies in this file were computed before three evaluation bugs were fixed: NaN entropy for fully confident notes (scored as p = 0.5), a volume feature that changed when views were masked, and a missing view self-gate in the policy path. Every checkpoint was re-evaluated with the fixed code; see `WEAK_RESULTS_FIX.md`. Where those numbers differ from the ones below, they supersede them, and verdicts based on the old numbers should be re-read. The original text is kept unchanged below.

The saved acquisition policy is unchanged. It scores 0.41346153846153844 at 1.00 view when λ = 0.02, because it can request a view other than the ordered prefix the classifier was trained on.

This file is a different rule on the same saved PRMVT checkpoint. It only uses the ordered prefix. The entropy threshold is chosen on the validation split. Source: `results/qduig/prefix_ft/seed42/prefix_stop_policy.json`. n = 208.

## Fixed prefix, no threshold

| rule | split | accuracy | mean views |
|---|---|---:|---:|
| always first view | val | 0.9807692307692307 | 1.0 |
| always first view | test | 0.9711538461538461 | 1.0 |
| always first 3 views | val | 0.9855769230769231 | 3.0 |
| always first 3 views | test | 0.9759615384615384 | 3.0 |

## Lambda grid

For each λ the validation rule maximizes accuracy minus λ times mean views. Test is scored after that choice.

| λ | val accuracy | val mean views | test accuracy | test mean views |
|---:|---:|---:|---:|---:|
| 0.0 | 0.9855769230769231 | 6.0 | 0.9759615384615384 | 6.0 |
| 0.01 | 0.9807692307692307 | 1.0144230769230769 | 0.9711538461538461 | 1.0096153846153846 |
| 0.02 | 0.9807692307692307 | 1.0144230769230769 | 0.9711538461538461 | 1.0096153846153846 |
| 0.05 | 0.9807692307692307 | 1.0144230769230769 | 0.9711538461538461 | 1.0096153846153846 |
| 0.1 | 0.9807692307692307 | 1.0144230769230769 | 0.9711538461538461 | 1.0096153846153846 |

The primary λ is 0.02, the validation cost weight already stored in `configs/proposed_prefix_ft.yaml`. On test it scores 0.9711538461538461 at 1.0096153846153846 views.

The oracle in `results/qduig/eval/seed42/oracle.json` is 0.9855769230769231 at 2.0961538461538463 views. It was not used to choose the threshold. The test gap from that oracle to the primary rule is 0.014423076923076923.

## Attempts that were not trained

A new policy network, oracle imitation, and beam search over arbitrary view subsets were not trained. Arbitrary subsets are the failure mode of the saved policy. The prefix rule meets the accuracy and view-count target on the saved classifier, so those retrainings were not run.

Target accuracy at most 3 views: met by the primary rule and by the fixed first-view rule.


> **Correction 2026-09-29.** The saved sequential policy (λ = 0.02) re-run with the fixed evaluation code scores 0.8606 at 1.00 view, not 0.4135 (`results/qduig/prefix_ft/seed42/cost_policy_test.json`; `CORRECTIONS.md` row 20). It remains below fixed view 1 (0.9712).

