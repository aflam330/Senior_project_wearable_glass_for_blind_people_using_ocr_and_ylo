# PAC-Bayes fix C: PAC-Bayes-kl

Same posterior, same D2 Gibbs risk, KL 0, n=487, delta=0.05. The kl inversion of that risk is 0.06646264504399683. Source: `pacbayes_kl_upper` in `results/theory/pacbayes_d1_bound_seed42.json`.

It is tighter than the McAllester value 0.11271182900151713 and is below 1. It bounds the same D1-trained Gibbs classifier, not the published PRMVT test accuracy.
