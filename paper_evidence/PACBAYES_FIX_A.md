# PAC-Bayes fix A: data-dependent prior

A her_base network was trained on 487 training notes (D1) and never saw the other 487 (D2) or the test split. The prior and the posterior are the same Gaussian, N(w_D1, rho^2 I), with rho fixed at 0.0001. KL is 0. Gibbs risk is the mean 0-1 error of four posterior draws on D2, seed 42.

Source: `results/theory/pacbayes_d1_bound_seed42.json`. The checkpoint is epoch 5 of `results/theory/pacbayes_d1_model/seed42/checkpoint.pt` (best validation mean 0.96875).

| quantity | value |
| --- | ---: |
| D2 deterministic error | 0.028747433264887063 |
| D2 Gibbs mean, 4 samples | 0.029260780287474333 |
| McAllester penalty | 0.0834510487140428 |
| McAllester bound | 0.11271182900151713 |

The bound is below 1. The Gibbs error is about 0.029, not the constant-predictor error near 0.423. This certifies the D1-trained network on D2. It is not a bound on the published PRMVT test accuracy. The zero-mean full-network penalty of 57.5892990573559 remains the result for that different prior.
