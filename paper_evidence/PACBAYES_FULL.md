# PAC-Bayes bounds, all results (2026-09-30)

No new training was needed. The results already measured are:

| Posterior / prior | Bound | Status | Source |
|---|---:|---|---|
| Full published PRMVT, Gaussian posterior, zero-mean prior (grid) | McAllester penalty 57.59 | vacuous | `results/theory/pacbayes_prmvt_seed42.json` |
| Full network, posterior std 10 | McAllester 0.648 (Gibbs training error 0.423) | non-vacuous, but for a very noisy stochastic network, not the deployed one | `PACBAYES_FIX_RESULTS.md` |
| Linear head, prior from half the training notes | McAllester 0.445 | non-vacuous | same |
| **Data-dependent prior**: network trained on 487 training notes, bound evaluated on the other 487 | **McAllester 0.1127** | **non-vacuous** | `results/theory/pacbayes_d1_bound_seed42.json` |

**What can be claimed.** With a data-dependent prior (half the training data fixes the prior, the other half evaluates the bound), a PRMVT-architecture network has a certified error bound of 11.3 %. That bound is for that half-data network, not the published checkpoint. The published checkpoint's own bound stays vacuous under a data-independent prior. The paper should state the half-data result as the certificate, with this scope.
