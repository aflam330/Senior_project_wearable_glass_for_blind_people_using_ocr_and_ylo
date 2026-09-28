# PAC-Bayes on a linear head

The Gaussian posterior on all 4,121,404 saved PRMVT weights remains vacuous. Its smallest McAllester penalty on the fixed prior grid is 57.5892990573559. Source: `results/theory/pacbayes_prmvt_seed42.json`.

A separate posterior was fit on half of the training notes. The encoder is the frozen PRMVT checkpoint. Only a linear head is random. The prior is N(0, I), chosen without the held-out half. The test set is not used. Source: `results/theory/pacbayes_head_d1d2_seed42.json`.

| quantity | value |
|---|---:|
| D1 / D2 notes | 487 / 487 |
| weight decay chosen on an inner split of D1 | 0.001 |
| KL(N(theta, I) \|\| N(0, I)) | 2.381910800933838 |
| D2 deterministic error | 0.020533880218863487 |
| D2 Gibbs error, 8 samples, seed 42 | 0.3475359324365854 |
| McAllester bound | 0.444538876551335 |
| PAC-Bayes-kl upper bound | 0.44287811904217583 |

The McAllester value is below 1. It bounds the Gibbs classifier on the held-out training half. It is not a bound on the published PRMVT test accuracy.

## Initialization prior on the full weight vector

The prior mean is the network before JaalTaka training. Displacement ||w - w_init||^2 is 18390.416960419072 over 4,121,404 parameters (`results/theory/pacbayes_init_prior_seed42.json`). With posterior variance equal to the prior variance, the KL is 0.5 * that displacement / sigma0^2.

Gibbs risk was measured on the training split, 4 samples, seed 42. The test set was not used. Source: `results/theory/pacbayes_gibbs_seed42.json`.

| sigma0 = rho | train Gibbs error | penalty | McAllester |
| ---: | ---: | ---: | ---: |
| 3 | 0.42325462012320325 | 0.7265042546120731 | 1.1497588747352765 |
| 5 | 0.502053388090349 | 0.43858112628372986 | 0.9406345143740789 |
| 10 | 0.42299794661190965 | 0.2254624459813364 | 0.6484603925932461 |

Two of these sums are below 1. The Gibbs error near 0.423 is the error of a constant predictor on this training split. The posterior standard deviation is 5 or 10, so the bound is not a certificate for the deterministic PRMVT checkpoint. The zero-mean prior penalty of 57.5892990573559 remains the result for a tight posterior.

Rademacher complexity, covering numbers, empirical Bernstein, and a Gaussian-mixture posterior were not computed.

## Data-dependent prior

A separate network was trained on 487 training notes and scored on the other 487. Prior and posterior are N(w_D1, 1e-8 I). KL is 0. Test was not used. Source: `results/theory/pacbayes_d1_bound_seed42.json`.

| quantity | value |
| --- | ---: |
| D2 Gibbs error, 4 samples | 0.029260780287474333 |
| McAllester bound | 0.11271182900151713 |
| PAC-Bayes-kl upper bound | 0.06646264504399683 |
| empirical Bernstein upper bound | 0.06702474420705011 |

The McAllester value is below 1, and the Gibbs error is not a constant predictor. It bounds this D1-trained network. It does not replace the vacuous zero-mean penalty on the published PRMVT weights.

Verdict: PASS for a non-vacuous bound on a network trained without the evaluation notes. The published full-network zero-mean penalty stays 57.5892990573559.
