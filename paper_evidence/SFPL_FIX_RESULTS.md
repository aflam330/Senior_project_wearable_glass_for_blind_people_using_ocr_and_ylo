# SFPL

Target: at least the CNN+ViT baseline at every view count. Baseline test accuracy is 0.7355769230769231, 0.8701923076923077, 0.9134615384615384, 0.9182692307692307, 0.8990384615384616, 0.9182692307692307 for 1–6 views.

## Standing v2

`results/novel_v2/sfpl/seed42/test/test_metrics.json`

| views | accuracy |
|---:|---:|
| 1 | 0.7403846153846154 |
| 2 | 0.8509615384615384 |
| 3 | 0.8605769230769231 |
| 4 | 0.8461538461538461 |
| 5 | 0.8461538461538461 |
| 6 | 0.8317307692307693 |

1-view is above the baseline. From 2 views on, it is below.

## Full-view FedAvg, 6 rounds

Every client saw all 6 views. `results/novel_v2/sfpl_fullviews/seed42/test/test_metrics.json`

| views | accuracy |
|---:|---:|
| 1 | 0.6875 |
| 2 | 0.7884615384615384 |
| 3 | 0.7067307692307693 |
| 4 | 0.7355769230769231 |
| 5 | 0.7403846153846154 |
| 6 | 0.7548076923076923 |

Below the baseline at every view count, and below v2 from 2 views on.

## Few local steps plus FedProx

8 rounds, 4 local steps, μ = 0.01, learning rate 0.0003. `results/novel_v2/sfpl_localsteps/seed42/test/test_metrics.json`

| views | accuracy |
|---:|---:|
| 1 | 0.7596153846153846 |
| 2 | 0.7451923076923077 |
| 3 | 0.7596153846153846 |
| 4 | 0.7548076923076923 |
| 5 | 0.7788461538461539 |
| 6 | 0.8509615384615384 |

1-view is above the baseline. Views 2–6 are below it. This does not replace v2.

30 communication rounds, an 80% client drop, hierarchical aggregation, and a personalized head were not trained after these runs stayed below the baseline from 2 views.

Verdict: FAIL. v2 remains the SFPL result.
