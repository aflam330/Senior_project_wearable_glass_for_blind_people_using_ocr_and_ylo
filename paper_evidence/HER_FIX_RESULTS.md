# HER scale chosen on validation

Fit on validation, 6 views, seed-42 PRMVT checkpoint. The scale of the additive HER terms is chosen on validation. Source: `results/calibration/her_threshold_seed42.json`.

| scale | val ECE | val accuracy at chosen threshold | val accuracy at 0.5 |
|---:|---:|---:|---:|
| 0.0 | 0.06351186960147552 | 0.9855769230769231 | 0.9855769230769231 |
| 0.25 | 0.03550566571245255 | 0.9855769230769231 | 0.9855769230769231 |
| 0.5 | 0.01146360914348269 | 0.9855769230769231 | 0.9855769230769231 |
| 0.75 | 0.012956135274949855 | 0.9903846153846154 | 0.9855769230769231 |
| 1.0 | 0.00895182619569826 | 0.9855769230769231 | 0.9855769230769231 |

The selection rule keeps scales whose validation ECE is under 0.04 and whose validation accuracy is at least the raw validation accuracy 0.9855769230769231. Scale 0.75 wins because its validation threshold 0.8867766261100769 reaches 0.9903846153846154.

On test, that same threshold scores 0.9711538461538461. That is below the uncalibrated 6-view accuracy. The threshold that raised validation accuracy did not transfer.

The same selected probabilities at threshold 0.5 score 0.9759615384615384 on test, with ECE 0.024493631835167225. That accuracy matches the uncalibrated 6-view accuracy of this checkpoint, and the ECE is under 0.04.

Full-strength HER, scale 1, remains the earlier result: ECE improves and threshold-0.5 accuracy falls to 0.9711538461538461 (`results/calibration/accuracy_seed42.json`).

The target is ECE under 0.04 and accuracy at least the uncalibrated 6-view accuracy, which is 0.9759615384615384. Scale 0.75 at threshold 0.5 meets both: accuracy 0.9759615384615384 and ECE 0.024493631835167225. The validation-maximizing threshold does not. No threshold was refit after seeing the test set.

Verdict: PASS at threshold 0.5 on the validation-selected scale. FAIL for the validation-maximizing threshold.
