# Diagnosis: SFPL

Standing result is v2, seed 42, test n=208. Source: `results/novel_v2/sfpl/seed42/test/test_metrics.json`.

| views | SFPL v2 | CNN+ViT baseline |
| ---: | ---: | ---: |
| 1 | 0.7403846153846154 | 0.7355769230769231 |
| 2 | 0.8509615384615384 | 0.8701923076923077 |
| 3 | 0.8605769230769231 | 0.9134615384615384 |
| 4 | 0.8461538461538461 | 0.9182692307692307 |
| 5 | 0.8461538461538461 | 0.8990384615384616 |
| 6 | 0.8317307692307693 | 0.9182692307692307 |

1-view is above the baseline. From 2 views on, v2 is below the baseline. The 6-view gap is 0.08653846153846145.

## Later runs

Full-view FedAvg and a short-step FedProx run (8 rounds, 4 local steps, mu 0.01) did not replace v2. Both are recorded in `paper_evidence/SFPL_FIX_RESULTS.md`. Neither reaches the 6-view baseline.

## Root cause

This is a data-size limit of the federated protocol, not an architecture depth issue. There are 974 training notes and 10 simulated clients, so each client sees on the order of 100 notes. Averaging those client weights removes fit that centralized prefix training keeps. Gradient norms per client were not logged. A 50-round, 5-local-epoch FedProx run was not in the saved logs. The existing shorter FedProx run already failed to beat v2, which is the evidence that more averaging on this split is the wrong direction unless a new run says otherwise.
