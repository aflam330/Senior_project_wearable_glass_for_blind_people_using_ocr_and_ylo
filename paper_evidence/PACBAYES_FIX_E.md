# PAC-Bayes fix E: Hoeffding and empirical Bernstein

These use the deterministic 0-1 losses of the D1 network on D2, n=487, delta=0.05. The test split was not used. Source: `results/theory/pacbayes_d1_bound_seed42.json`.

| bound | value |
| --- | ---: |
| D2 deterministic error | 0.028747433264887063 |
| one-sided Hoeffding upper bound | 0.08420643150859015 |
| empirical Bernstein upper bound | 0.06702474420705011 |

Both are below 1. They bound the fixed D1 classifier, not a posterior over the published PRMVT weights. Full-network Rademacher complexity was not computed and is NOT_MEASURED. Weight normalization was not run, because the data-dependent prior already produced a non-vacuous bound.
