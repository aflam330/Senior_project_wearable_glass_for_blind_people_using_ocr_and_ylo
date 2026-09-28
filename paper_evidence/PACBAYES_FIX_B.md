# PAC-Bayes fix B: layer-wise KL

The posterior mean equals the prior mean, so the displacement of every floating-point tensor is 0 and every layer KL is 0. There are 334 such tensors. Their sum is 0, which is the same McAllester bound as the full vector: 0.11271182900151713.

Source: `results/theory/pacbayes_d1_bound_seed42.json`. Splitting the KL by layer does not tighten this bound, because there is no KL to split.
