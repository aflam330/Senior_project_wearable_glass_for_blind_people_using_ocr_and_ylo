# Oracle gap

Definition. On one checkpoint and one split, the oracle gap is the accuracy of the best view-subset selector minus the accuracy of the deployed policy. The oracle reads labels, so it is an analysis ceiling. It is not a deployable policy and was not used to train.

## Measured ceiling

`results/qduig/eval/seed42/oracle.json`, n = 208:

| quantity | value |
| --- | ---: |
| oracle accuracy | 0.9855769230769231 |
| oracle mean views | 2.0961538461538463 |
| learned accuracy in that file | 0.9663461538461539 |
| learned mean views | 6.0 |
| gap | 0.019230769230769273 |
| notes no subset can solve | 3 |

The gap is zero only if the deployed policy matches the oracle's subset on every solvable note. Three notes are wrong under every subset, so the ceiling is not 1. The stored accuracies are 205/208 and 201/208, so the gap is 4 notes, and 208 − 205 = 3.

## Prefix-stop policy, different row

`results/qduig/prefix_ft/seed42/prefix_stop_policy.json`. Lambda 0.02 was chosen on validation. Test accuracy is 0.9711538461538461 at 1.0096153846153846 views. The forced 6-view test accuracy in the same file is 0.9759615384615384. That difference is 0.0048076923076923. It is not subtracted from the oracle file above, because that file's learned policy is a different operating point (6.0 views, accuracy 0.9663461538461539).

No second dataset was used for the JaalTaka oracle file. A fitted predictor of the gap was not trained on one point.

## Rendered ModelNet subset, 2026-09-29

On `subset10_seed42_e40.json`, the prefix model scores 0.77 at 6 views. An analysis oracle that keeps the object if any of the six prefixes is correct scores 0.85. The difference is 0.08 on 100 objects. That oracle reads labels. It is not the JaalTaka subset oracle, and the two gaps are not subtracted from each other.

---

## Correction 2026-09-29: which model the oracle file describes

`results/qduig/eval/seed42/oracle.json` was produced by `scripts/evaluate_all.py` for the early
`results/qduig/proposed` model (its 6-view accuracy, 0.966, is that model's), not for the deployed
PRMVT. It was also computed before three evaluation bugs were fixed on 2026-09-28 (missing view
self-gate in `run_oracle`, NaN entropy for confident notes, masked-view volume; see
`WEAK_RESULTS_FIX.md`). The oracle was re-run on the deployed checkpoints with the fixed code,
test split, n = 208 (`results/qduig/oracle_20260929/`):

| checkpoint | oracle accuracy | oracle mean views | 6-view accuracy | gap (notes) | unsolvable by any subset |
|---|---:|---:|---:|---:|---:|
| PRMVT prefix_ft seed 42 | 0.9952 (207) | 1.00 | 0.9808 (204) | 3 | 1 |
| occlusion-robust seed 42 | 1.0000 (208) | 1.00 | 0.9808 (204) | 4 | 0 |
| occlusion-robust seed 43 | 1.0000 (208) | 1.00 | 0.9904 (206) | 2 | 0 |
| occlusion-robust seed 44 | 0.9952 (207) | 1.00 | 0.9904 (206) | 1 | 1 |

The oracle reads the label, so these are analysis ceilings, not policies. For the deployed model
almost every note has at least one single view it classifies correctly; the "three notes no subset
can solve" in the section above belongs to the early model and to the pre-fix code.
