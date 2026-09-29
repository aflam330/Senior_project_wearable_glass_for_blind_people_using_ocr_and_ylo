# Calibration, seed 42, 6 views

Temperature and HER are fit on the validation split only. Metrics are on the test split. ECE uses 10 bins. MC dropout is 10 stochastic passes.

| model | method | ECE | adaptive ECE | Brier | NLL |
|---|---|---:|---:|---:|---:|
| prmvt | predictive_confidence | 0.0608 | 0.0652 | 0.0443 | 0.1622 |
| prmvt | temperature | 0.0603 | 0.0726 | 0.0451 | 0.1536 |
| prmvt | entropy | 0.0622 | 0.0722 | 0.0457 | 0.1573 |
| prmvt | her | 0.0328 | 0.0348 | 0.0307 | 0.1309 |
| prmvt | mc_dropout | 0.0737 | 0.0582 | 0.0408 | 0.1471 |
| baseline | predictive_confidence | 0.0645 | 0.0831 | 0.0672 | 0.2336 |
| baseline | temperature | 0.0639 | 0.0762 | 0.0676 | 0.2401 |
| baseline | entropy | 0.1029 | 0.1293 | 0.0804 | 0.2805 |
| baseline | her | 0.0529 | 0.0577 | 0.0647 | 0.2296 |
| baseline | mc_dropout | 0.0615 | 0.0731 | 0.0676 | 0.2335 |
| ndal | predictive_confidence | 0.0763 | 0.0763 | 0.0363 | 0.1511 |
| ndal | temperature | 0.0329 | 0.0307 | 0.0274 | 0.1058 |
| ndal | entropy | 0.1679 | 0.1682 | 0.0668 | 0.2546 |
| ndal | her | 0.0225 | 0.0321 | 0.0331 | 0.1259 |
| ndal | mc_dropout | 0.0784 | 0.0685 | 0.0376 | 0.1558 |
| ugf | predictive_confidence | 0.0342 | 0.0382 | 0.0425 | 0.1663 |
| ugf | temperature | 0.0267 | 0.0375 | 0.0418 | 0.1595 |
| ugf | entropy | 0.0375 | 0.0432 | 0.0413 | 0.1602 |
| ugf | her | 0.0325 | 0.0263 | 0.0387 | 0.1631 |
| ugf | mc_dropout | 0.0292 | 0.0361 | 0.0412 | 0.1615 |
