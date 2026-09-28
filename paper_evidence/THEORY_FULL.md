# Theory

Numbers below are from saved artifacts. A convergence rate for the non-convex trainer was not proved.

## Data-dependent PAC-Bayes

A her_base network was trained on 487 of the 974 training notes and scored on the other 487. The test split was not used. Prior and posterior are the same Gaussian N(w_D1, 10^{-8} I), so KL is 0. Source: `results/theory/pacbayes_d1_bound_seed42.json`.

| quantity | value |
| --- | ---: |
| D2 Gibbs error, 4 samples | 0.029260780287474333 |
| McAllester bound | 0.11271182900151713 |
| PAC-Bayes-kl upper bound | 0.06646264504399683 |
| one-sided Hoeffding upper bound | 0.08420643150859015 |
| empirical Bernstein upper bound | 0.06702474420705011 |

The Gibbs error is not a constant predictor. The bound is below 1. It certifies that half-data network. It does not certify the published PRMVT checkpoint.

## Published full network

The zero-mean Gaussian prior on all 4,121,404 saved PRMVT weights has McAllester penalty 57.5892990573559 (`results/theory/pacbayes_prmvt_seed42.json`). That penalty is vacuous. A linear head on frozen features, fit on one half of the training notes, has McAllester 0.444538876551335 (`results/theory/pacbayes_head_d1d2_seed42.json`). That is a different classifier. The encoder that made those features had seen both halves.

Full-network Rademacher complexity: NOT_MEASURED.

## Prefix training, finite sample

Let the training mask be a random prefix. The only finite-sample certificate computed for a prefix-trained network in this repository is the half-data bound above, and that network was her_base, whose fusion is mean-pool rather than the published RSQA prefix model. A separate sample-complexity theorem with a new rate constant was not derived.
