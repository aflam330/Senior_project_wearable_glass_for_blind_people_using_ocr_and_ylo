# Ablation results

| config | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| quality | 0.5625 | 0.5529 | 0.7500 | 0.8606 | 0.8654 | 0.9519 |
| uncertainty | 0.6971 | 0.7019 | 0.7115 | 0.7356 | 0.7452 | 0.8173 |
| diversity | 0.7019 | 0.8654 | 0.9183 | 0.9135 | 0.9135 | 0.8942 |
| qd | 0.6587 | 0.4615 | 0.6010 | 0.7019 | 0.6971 | 0.8269 |
| ud | 0.7596 | 0.8942 | 0.9038 | 0.9231 | 0.9231 | 0.9567 |
| full | 0.6298 | 0.6394 | 0.6490 | 0.7212 | 0.7067 | 0.7404 |
| no_cost | 0.6298 | 0.6394 | 0.6490 | 0.7212 | 0.7067 | 0.7404 |
| no_calibration | 0.6298 | 0.6394 | 0.6490 | 0.7212 | 0.7067 | 0.7404 |
| no_redundancy | 0.5721 | 0.6635 | 0.7692 | 0.8413 | 0.8606 | 0.9519 |
| no_infogain | 0.5625 | 0.7212 | 0.7644 | 0.8798 | 0.8510 | 0.9231 |

These rows are the 4-epoch component configs, not the 6-epoch plus 3-epoch prefix-robust checkpoint. A config that only adds PCR-IG, with every other term off, has no yaml. That row is NOT_MEASURED. The names RSQA, CVR, and HER in the older design notes correspond to the quality, diversity, and uncertainty configs. If two configs show the same six accuracies, that is what the test files contain.
