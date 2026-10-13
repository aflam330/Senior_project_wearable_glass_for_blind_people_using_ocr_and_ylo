# Theory index for the eight requested items (2026-10-13)

The project already has 16 numbered theorems and propositions in `THEOREMS.md` and `THEORY_FINAL.md` (written 29–30 September) and six propositions in `THEORY_STRONG.md` (13 October). This page maps the eight requested items to them, adds one statement that was missing, and says which items cannot be theorems.

Markers: **new** (formulated in this project), **applied** (a standard result applied here), **not a theorem**.

| Item | Statement | Where | Kind | What the data say |
|---|---|---|---|---|
| A. View-count shift: definition, collapse condition, bound | Proposition 11: training on a fixed number of views does not identify the predictor for fewer views. Theorem 3: per-count risk bound under prefix-mixture training. Theorem 1: the Bayes risk cannot fall when views are dropped | `THEOREMS.md` | new (Prop. 11, Thm 3), applied (Thm 1) | Same network: 71.2 ± 11.4 % against 98.2 ± 0.7 % at one view. Not universal: on MVP-N it depends on the fusion head |
| B. Safe threshold; multi-class; under shift | Theorem 8 (rank argument, 1/(n+1)); Theorem 9 (plus the total-variation distance under shift); **Proposition 17 below** (several note values) | `THEOREMS.md`, here | applied (8, 9), new (17) | 0 of 88 test counterfeits passed |
| C. UV fibre detection bounds | Propositions 2–4 of `THEORY_STRONG.md`: lamp intensity cancels, colour balance does not, Poisson bounds on both errors inside one capture condition | `THEORY_STRONG.md` | new | The Poisson model fits one capture condition only. See the warning below |
| D. DMWR: quotient image, portrait | Proposition 5 (quotient image removes a smooth lamp within a factor 1/(1 ± ε)); Proposition 6 (the residual carries the optimal statistic under a Gaussian model). **Proposition 18 below** (the median portrait converges) | `THEORY_STRONG.md`, here | new (5), applied (6, 18) | Both streams together 91.5 %, either alone 85.8–88.0 % |
| E. Print-disjoint generalisation | Theorem 16: with correlated notes, the effective sample size is the number of prints. Proposition 13: how shared prints inflate note-disjoint accuracy | `THEORY_FINAL.md`, `THEOREMS.md` | new | About 24 test prints: differences of a few points cannot be established |
| F. Cross-currency transfer | — | — | **not a theorem** | No second currency has been measured. A domain-adaptation bound could be quoted, but its divergence term cannot be estimated without data from the second currency, so it would say nothing |
| G. Prefix training converges | Theorem 15 | `THEOREMS.md` | applied | — |
| H. Prefix training sample complexity | Theorem 3: the bound for view count k scales with 1/π_k, the share of training batches with k views | `THEOREMS.md` | new | With a uniform choice of k from 1 to 6, each count gets one sixth of the batches |

## Proposition 17 (safe threshold for several note values)

Let there be K note values. For value c, let τ_c be the largest score among n_c validation counterfeits of that value, and let the glass say "likely genuine" for a note named c only if its score exceeds τ_c. Suppose that, for each c, a new counterfeit of value c is exchangeable with the n_c validation counterfeits of that value, with no ties. Then:

(a) a counterfeit of value c passes with probability exactly 1/(n_c + 1);

(b) if a counterfeit's value is drawn with probabilities w_1, …, w_K, it passes with probability Σ_c w_c/(n_c + 1) ≤ 1/(min_c n_c + 1);

(c) if one counterfeit of each value is presented, the probability that at least one passes is at most Σ_c 1/(n_c + 1).

*Proof.* (a) is Proposition 1 of `THEORY_STRONG.md` applied to value c alone. (b) is the law of total probability over the value, and each term is at most w_c/(min n + 1). (c) is the union bound over the K events of (a). ∎

*Use.* The glass has one threshold for 500 and 1,000 Taka together (n = 88). A threshold per value would keep the guarantee per value but with smaller n_c, so a larger 1/(n_c + 1). Pooling is the better choice only if scores of the two values are exchangeable with each other, which the data are too few to test.

## Proposition 18 (the median portrait converges)

Fix a pixel. Let the quotient-image values of genuine notes of one value at that pixel be independent draws from a distribution with a unique median m and a density that is at least κ > 0 on [m − η, m + η]. Then the sample median T of n notes satisfies, for 0 < t ≤ η,

  P(|T − m| > t) ≤ 2·exp(−2nκ²t²).

*Proof.* The event T > m + t requires at least n/2 of the n draws to exceed m + t. Each does so with probability at most 1/2 − κt. By Hoeffding's inequality the probability that the fraction exceeds its mean by κt is at most exp(−2nκ²t²). The lower side is symmetric. ∎

*Use.* The portraits are built from 235 (500 Taka) and 260 (1,000 Taka) genuine notes, so the per-pixel deviation falls like 1/√n. The assumption of independence across notes is reasonable; a unique median with positive density is the usual regularity condition. Nothing here says the portrait is the right reference for a new series of notes.

## Warning on item C (13 October, evening)

Propositions 2–4 describe the fibre method correctly, but a simulation run after they were written shows that the count they bound is not a count of fibres only: on simulated fibre-free printed sheets, JPEG compression produced as many "fibres" as genuine notes have (`UV_SYNTHETIC_TEST.md`). In the language of Proposition 4, the false-count mean ν on printed paper is not small, so the bound gives no protection. The propositions stand; the claim that the method detects fibres did not.
