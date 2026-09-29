# Theory, final index (2026-09-30)

Full statements and proofs are in `THEOREMS.md`. This page maps the seven requested items to proved results, and adds Theorem 16. Each item is marked **new** (formulated here), **applied** (a standard result applied to this setting), or **checked**.

| # | Requested | Result | Status | Measured check |
|---|---|---|---|---|
| 1 | Print-disjoint generalisation bound | **Theorem 16** (below) + Proposition 13 | new | 101 test counterfeits from about 20 prints: the print-level margin is much wider than the note-level one |
| 2 | Watermark detection guarantee | Theorem 8 applied to the watermark score (max-of-validation threshold) + Theorem 9 under shift | applied / new | In-domain 0 / 88 passed; shift term not estimable from 4 whole-note prints |
| 3 | Safety-aware rejection optimality | Theorem 7 (Chow's rule, constrained Neyman–Pearson form, monotonicity) | applied | Rejection on unseen prints: about 3 % wrong among answered vs 1 % on validation: the calibration assumption fails under print shift |
| 4 | Serial-anomaly detection bound | Proposition 12 (recall = share of known prints, 0 on new prints) | new | 0 / 19 unseen-serial counterfeits; real photos: 1 / 4 prints, 0 / 450 genuine |
| 5 | Multi-detector fusion bound | Proposition 14 (OR / AND error bounds) | new (elementary) | Hybrid false-counterfeit 3.3 %; misses 7 / 101 |
| 6 | View-count collapse theorem | Theorem 1 (Bayes monotonicity), Theorem 3 (prefix-mixture bound with counterexample), Proposition 11 (non-identifiability), Theorem 15 (convergence, standard) | new (3, 11) / applied (1, 15) | Fixed-view seed spread 131–194 correct at 1 view; prefix 197–206 |
| 7 | Oracle gap law | Theorem 5 (exact form and bounds) | new (elementary) | Gap 3 notes for PRMVT seed 42 |

## Theorem 16 (evaluation with correlated notes: the effective sample size is the number of prints)

**Setting.**
- Test counterfeits come from m distinct prints; print j has n_j notes, with total n = Σ n_j.
- Let a_j ∈ [0, 1] be the detector's catch rate on print j's notes, and ĉ = Σ_j n_j a_j / n the observed catch rate.
- Prints are drawn i.i.d. from a print population; the target is the population catch rate c = E[a_J] (J a random print).
- Assume equal print sizes n_j = n/m for the clean statement.

**Claim.** With probability at least 1 − δ over the draw of prints,

  |ĉ − c| ≤ √( ln(2/δ) / (2m) ),

and no bound of order √(1/n) holds in general.

**Proof.**
1. With equal sizes, ĉ = (1/m) Σ_j a_j is a mean of m i.i.d. [0, 1] variables with mean c, and Hoeffding's inequality gives the bound.
2. For the second part, let every print be either always caught (a_j = 1) or never caught (a_j = 0), each with probability ½. Then ĉ has the variance of a mean of m fair coins, whatever n is, so the error is of order 1/√m, not 1/√n. ∎

**Numbers.** On the serial-disjoint test the 101 counterfeits fall into 62 serial groups: many unread serials are counted as their own group; the largest groups have 12, 10, 9, 5 and 3 notes. The effective number of prints is m_eff = n² / Σ n_j² = **23.6**. At δ = 0.05:
- print-level half-width √(ln 40 / (2 · 23.6)) ≈ 0.28;
- the note-level Hoeffding width that would wrongly be used, √(ln 40 / 202) ≈ 0.14.

**Consequence.**
- The unseen-print accuracies are estimates with wide uncertainty: the evidence is about 20 prints, not 101 notes.
- The paired McNemar tests stay valid for comparing methods on the **same** test notes. They do not generalise the absolute numbers to new prints.
- More counterfeit prints, not more notes per print, is what narrows the estimate.

Unequal print sizes: replace m by the effective number m_eff = n² / Σ n_j² (Hoeffding for weighted sums), which is ≤ m.
