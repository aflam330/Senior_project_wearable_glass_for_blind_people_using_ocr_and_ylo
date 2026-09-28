# SFPL final result

Federated prefix learning on 974 training notes across 10 simulated clients underperforms centralized training. This is a negative result. Federated learning may require larger datasets to match centralized performance.

The standing run is v2, seed 42, test n=208. Source: `realtime_bangla_taka_detection/results/novel_v2/sfpl/seed42/test/test_metrics.json`.

| views | SFPL v2 | CNN+ViT baseline |
|---:|---:|---:|
| 1 | 0.7403846153846154 | 0.7355769230769231 |
| 2 | 0.8509615384615384 | 0.8701923076923077 |
| 3 | 0.8605769230769231 | 0.9134615384615384 |
| 4 | 0.8461538461538461 | 0.9182692307692307 |
| 5 | 0.8461538461538461 | 0.8990384615384616 |
| 6 | 0.8317307692307693 | 0.9182692307692307 |

1-view is above the baseline. From 2 views on, SFPL v2 is below the baseline.

A later full-view FedAvg run and a short-step FedProx run did not beat v2 from 2 views on. They are in `SFPL_FIX_RESULTS.md` and do not replace this table.

Verdict: negative result. v2 is the SFPL number.
