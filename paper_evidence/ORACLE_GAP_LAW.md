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

The gap is zero only if the deployed policy matches the oracle's subset on every solvable note. Three notes are wrong under every subset, so the ceiling is not 1.

## Prefix-stop policy, different row

`results/qduig/prefix_ft/seed42/prefix_stop_policy.json`. Lambda 0.02 was chosen on validation. Test accuracy is 0.9711538461538461 at 1.0096153846153846 views. The forced 6-view test accuracy in the same file is 0.9759615384615384. That difference is 0.0048076923076923. It is not subtracted from the oracle file above, because that file's learned policy is a different operating point (6.0 views, accuracy 0.9663461538461539).

No second dataset was used. A fitted predictor of the gap was not trained.
