# VCIE

Target was at least 0.9182692307692307, the baseline 6-view accuracy, starting from the v2 1-view accuracy 0.8798076923076923.

## Standing v2

`results/novel_v2/vcie/seed42/test/test_metrics.json`

| views | accuracy |
|---:|---:|
| 1 | 0.8798076923076923 |
| 2 | 0.875 |
| 3 | 0.9086538461538461 |
| 4 | 0.8990384615384616 |
| 5 | 0.9038461538461539 |
| 6 | 0.9134615384615384 |

6-view is 0.00480769230769229 below 0.9182692307692307.

## Eight-epoch continuation of the same 2-block encoder

Checkpoint chosen by validation mean 1–6. Best validation mean was epoch 2. Test: `results/novel_v2/vcie_long/seed42/test/test_metrics.json`

| views | accuracy |
|---:|---:|
| 1 | 0.7548076923076923 |
| 2 | 0.8509615384615384 |
| 3 | 0.8942307692307693 |
| 4 | 0.9182692307692307 |
| 5 | 0.9134615384615384 |
| 6 | 0.9086538461538461 |

1-view is worse than v2. This run does not replace v2. A 6-block set transformer was not trained after this run lowered 1-view accuracy.

Verdict: FAIL. v2 remains the VCIE result.
