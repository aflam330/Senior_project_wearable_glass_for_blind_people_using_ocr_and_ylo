# Calibration, seed 42, 6 views

Temperature and HER are fit on the validation split only. Metrics are on the test split. ECE uses 10 bins. MC dropout is 10 stochastic passes.

| model | method | ECE | adaptive ECE | Brier | NLL |
|---|---|---:|---:|---:|---:|
| prmvt | predictive_confidence | 0.0146 | 0.0151 | 0.0167 | 0.0866 |
| prmvt | temperature | 0.0168 | 0.0221 | 0.0177 | 0.0798 |
| prmvt | entropy | 0.0153 | 0.0227 | 0.0185 | 0.0839 |
| prmvt | her | 0.0178 | 0.0232 | 0.0224 | 0.0955 |
| prmvt | mc_dropout | 0.0136 | 0.0155 | 0.0163 | 0.0784 |
| baseline | predictive_confidence | 0.0647 | 0.0832 | 0.0673 | 0.2337 |
| baseline | temperature | 0.0641 | 0.0764 | 0.0676 | 0.2402 |
| baseline | entropy | 0.1027 | 0.1294 | 0.0804 | 0.2805 |
| baseline | her | 0.0459 | 0.0577 | 0.0646 | 0.2295 |
| baseline | mc_dropout | 0.0662 | 0.0841 | 0.0668 | 0.2304 |
| ndal | predictive_confidence | 0.0763 | 0.0763 | 0.0363 | 0.1511 |
| ndal | temperature | 0.0330 | 0.0307 | 0.0274 | 0.1058 |
| ndal | entropy | 0.1678 | 0.1681 | 0.0668 | 0.2545 |
| ndal | her | 0.0224 | 0.0321 | 0.0331 | 0.1258 |
| ndal | mc_dropout | 0.0782 | 0.0782 | 0.0373 | 0.1546 |
| ugf | predictive_confidence | 0.0341 | 0.0382 | 0.0425 | 0.1663 |
| ugf | temperature | 0.0267 | 0.0375 | 0.0418 | 0.1595 |
| ugf | entropy | 0.0374 | 0.0432 | 0.0413 | 0.1601 |
| ugf | her | 0.0325 | 0.0266 | 0.0388 | 0.1632 |
| ugf | mc_dropout | 0.0365 | 0.0374 | 0.0435 | 0.1674 |
