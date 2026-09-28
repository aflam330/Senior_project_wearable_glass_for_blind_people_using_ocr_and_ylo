# Diagnosis: PAC-Bayes

The full-network McAllester penalty on the published PRMVT weights is 57.5892990573559. Any bound that adds a risk in [0, 1] is then above 1. Source: `results/theory/pacbayes_prmvt_seed42.json`. Parameter count is 4,121,404. The prior in that file is N(0, sigma0^2 I).

## Why the penalty is large

The squared L2 of the saved weights is about 129211.72. A zero-mean Gaussian prior charges KL proportional to that norm over sigma0^2, plus a dimension term when the posterior variance differs from the prior variance. With 4.1 million dimensions, a tight posterior around a large weight vector is vacuous.

## What a wide posterior does

An initialization-centered prior with posterior standard deviation 5 or 10 has McAllester sums below 1 on the training split (`results/theory/pacbayes_gibbs_seed42.json`). At sigma 10 the Gibbs error is 0.42299794661190965, which matches a constant predictor on this training split (majority-class error about 0.423). The bound is non-vacuous only because the predictor is no longer the trained network.

## What is already non-vacuous

A linear head on frozen PRMVT features, fit on 487 training notes and scored on the other 487, has McAllester 0.444538876551335 (`results/theory/pacbayes_head_d1d2_seed42.json`). That bound is not a certificate for the published 4.1 million-weight network. The encoder that produced those features was trained on all training notes, including the held-out half.

## Root cause

The vacuous number is the KL of a tight posterior against a zero-mean prior in 4.1 million dimensions. It is not a data-label error. A prior centered on a network that was trained without the evaluation notes, with the posterior kept at that same center, is the setting that has not been trained yet. Layer-wise KL, Rademacher complexity, and an empirical Bernstein bound on that network were not computed.
