# JaalTaka sequential benchmark

The split is 974 / 208 / 208 notes, seed 42. A policy sees views in file order and may stop. Metrics that were saved are accuracy and mean views. An oracle that reads labels is analysis only.

Measured policies on the prefix fine-tune, test n = 208, from `results/qduig/prefix_ft/seed42/prefix_stop_policy.json`:

| policy | test accuracy | mean views |
| --- | ---: | ---: |
| always 1 view | 0.9711538461538461 | 1.0 |
| always 3 views | 0.9759615384615384 | 3.0 |
| always 6 views (lambda 0) | 0.9759615384615384 | 6.0 |
| stop, lambda 0.02, chosen on validation | 0.9711538461538461 | 1.0096153846153846 |

The lambda 0.01, 0.05, and 0.1 rows in that file selected the same validation threshold and the same test numbers as lambda 0.02.

Nine additional named agents were not given a new leaderboard pass in this file. Their existing test tables, where present, stay in `results/novel` and `results/novel_v2`. This benchmark does not invent a ninth score.
