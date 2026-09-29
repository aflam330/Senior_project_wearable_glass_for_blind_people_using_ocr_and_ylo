# Theorems for view-count robustness, view selection and rejection (2026-09-29)

This file states each result with its assumptions and a complete proof, then gives the empirical check that exists for it. It supersedes the informal statements in `VCDS_THEOREM.md`, `SEQUENTIAL_PROBLEM.md`, `ORACLE_GAP_LAW.md` and `SAFETY_FRAMEWORK.md`, and keeps their measured numbers.

One result requested in the task list is **false as stated** and is replaced by a correct one (Theorem 3).

## Setting and notation

An instance is (X_1, …, X_N, Y), where the X_i are the views of one physical object and Y ∈ {0, 1} is its label. For a view set S ⊆ {1..N}, X_S is the tuple of the views in S. The k-view prefix is P_k = {1..k}.

A predictor h for view count k maps X_{P_k} to {0, 1}. For a network built for N inputs, the missing views are masked. The masked input is a function of X_{P_k}, so this is still a predictor from X_{P_k}.

- R_k(h) = P(h(X_{P_k}) ≠ Y) is the k-view risk.
- R*(S) is the Bayes risk from X_S. R*(k) = R*(P_k).
- The data are n i.i.d. instances. The test set has n_test instances.

Assumptions used below:

- **A1.** The views are observations of one object. They may be dependent.
- **A2.** Y is a property of the object and does not change when views are dropped.
- **A3.** Every model choice (checkpoint, threshold, calibrator) is made without the test split.

---

## Theorem 1 (Bayes risk cannot fall when views are dropped)

Under A2, if S ⊆ T then R*(T) ≤ R*(S). In particular R*(N) ≤ R*(k) for every k ≤ N.

**Proof.** Let g be any measurable predictor from X_S. Since S ⊆ T, X_S = π(X_T) for the coordinate projection π, so g∘π is a measurable predictor from X_T with the same risk as g. The Bayes predictor from X_T minimises risk over all predictors from X_T, including g∘π. Hence R*(T) ≤ R(g) for every g, and taking the infimum over g gives R*(T) ≤ R*(S). ∎

**Scope.** This is about Bayes risk. A trained network need not be monotone in k.

**Empirical check.** PRMVT seed 42 (post-fix) is correct on 205 notes at 5 views and 204 at 6 views. That does not contradict the theorem, which is about Bayes risk, not about a network (`results/qduig/prefix_ft/seed42/test_views_20260928/`).

---

## Theorem 2 (test error of a fixed classifier, simultaneously over view counts)

Let h be fixed before the test set is drawn (A3), and let R̂_k(h) be its test error at view count k on n_test i.i.d. notes. With probability at least 1 − δ, simultaneously for all k = 1..N:

  |R_k(h) − R̂_k(h)| ≤ √( log(2N/δ) / (2 n_test) ).

**Proof.** For each fixed k, the 0-1 losses 1[h(X_{P_k}) ≠ Y] are i.i.d. in [0, 1] with mean R_k(h). Hoeffding's inequality gives P(|R̂_k − R_k| > ε) ≤ 2 exp(−2 n_test ε²). A union bound over the N view counts gives failure probability at most 2N exp(−2 n_test ε²). Setting this equal to δ and solving for ε gives the display. ∎

**Numbers.** For n_test = 208, N = 6 and δ = 0.05, ε = √(log 240 / 416) = 0.1147. With N = 1 the value is 0.0942 (`vcds_theorem_constants.json`).

The CNN+ViT baseline's 1-view versus 6-view difference, 0.1827, is smaller than 2ε = 0.229. So two separate Hoeffding intervals alone do not separate R_1 from R_6. Paired tests on the same notes are much sharper. For example, the baseline versus PRMVT at 1 view gives McNemar p = 4.3×10⁻¹¹ (`STATISTICAL_ANALYSIS.md`).

---

## Theorem 3 (prefix-mixture training): corrected statement

**The requested statement is false.** "Prefix-robust training achieves err_k ≤ err_N + O(1/√n)" cannot hold in general.

*Counterexample.* Let Y be a fair bit, X_N = Y, and X_1..X_{N−1} be independent noise. Then R*(N) = 0 and R*(k) = 1/2 for every k < N. By Theorem 1 no predictor, however trained, has R_k ≤ 1/2 − c for any c > 0. So err_k − err_N ≥ 1/2 for every method, and no O(1/√n) term can close that gap.

**The correct comparison is against the best achievable risk at k views.**

**Setting.** Let H be a hypothesis class of predictors that accept any prefix (masked inputs). Let π = (π_1..π_N) be the prefix-length distribution used in training, with π_k > 0 for all k. The mixture risk is R_π(h) = Σ_k π_k R_k(h). Its empirical version R̂_π averages the loss over training instances and prefix lengths; for simplicity take the expectation over k exactly. The learned model ĥ minimises R̂_π over H.

Let U_n(δ) be a uniform-convergence bound: with probability ≥ 1 − δ, sup_{h∈H} |R_π(h) − R̂_π(h)| ≤ U_n(δ). For finite H, Hoeffding plus a union bound gives U_n(δ) = √(log(2|H|/δ)/(2n)). For general H it is 2·Rad_n(H) + √(log(1/δ)/(2n)), where Rad_n is the Rademacher complexity of the mixture loss class. Either way U_n(δ) = O(1/√n) for fixed H.

Define the **joint approximation gap** ε_H = inf_{h∈H} max_k [R_k(h) − R*(k)]. This is the smallest amount by which one model in H can be simultaneously near-Bayes at every view count.

**Theorem 3.** With probability at least 1 − δ, for every k:

  R_k(ĥ) ≤ R*(k) + ( ε_H + 2 U_n(δ) ) / π_k.

**Proof.**

1. Let h° attain ε_H to within an arbitrary η > 0. Then R_π(h°) ≤ Σ_j π_j R*(j) + ε_H + η.
2. On the uniform-convergence event:
   R_π(ĥ) ≤ R̂_π(ĥ) + U_n ≤ R̂_π(h°) + U_n ≤ R_π(h°) + 2U_n.
3. Combining, Σ_j π_j [R_j(ĥ) − R*(j)] ≤ ε_H + η + 2U_n.
4. Every term R_j(ĥ) − R*(j) is ≥ 0, because R*(j) is the minimum possible risk from X_{P_j}. Dropping the terms j ≠ k leaves π_k [R_k(ĥ) − R*(k)] ≤ ε_H + η + 2U_n.
5. Divide by π_k and let η → 0. ∎

**Corollary (fixed-view training gives no guarantee at k < N).** Fixed-view training is the case π_N = 1 and π_k = 0 for k < N. The argument then bounds only R_N(ĥ), and nothing constrains R_k(ĥ) for k < N. The resulting excess risk at k views,

  Δ_k = R_k(ĥ_fixed) − R_k(ĥ_prefix),

is what `VCDS_THEOREM.md` calls view-count distribution shift. It is not bounded by any O(1/√n) term. Theorem 1 separately bounds the loss due to information, R*(k) − R*(N).

**Decomposition used in the experiments.** Take ĥ_prefix as a proxy for the best achievable k-view risk. The fixed model's excess error at k views relative to its own N-view error splits into two parts:

  R_k(ĥ_fixed) − R_N(ĥ_fixed)
   = [R_k(ĥ_fixed) − R_k(ĥ_prefix)]   (view-count shift)
   + [R_k(ĥ_prefix) − R_N(ĥ_fixed)]   (information loss plus estimation difference)

**Empirical check** (`SAME_ARCH_RESULTS.md`, same PRMVT network, seeds 42–44, 208 test notes, finished 2026-09-29).

| Training | 1 view | 6 views |
|---|---:|---:|
| Fixed 6-view, shared BN (π_N = 1) | 71.2 ± 11.4 % | 98.1 ± 0.5 % |
| Prefix mixture, shared BN (π_k > 0) | 98.2 ± 0.7 % | 98.7 ± 0.3 % |
| Fixed 6-view, per-count BN | 87.2 ± 9.7 % | 99.0 ± 0.0 % |
| Prefix mixture, per-count BN (PRMVT) | 96.5 ± 1.5 % | 98.1 ± 1.0 % |

The corollary says fixed-view training gives no guarantee at k < N. That is what is seen: its 1-view accuracy is unconstrained and varies widely across seeds (138, 131, 175 correct notes with shared BN; 158, 194, 192 with per-count BN). Prefix training keeps it within 197–206 on every seed. At 6 views no pair differs significantly (paired McNemar).

---

## Theorem 4 (information gain is monotone and submodular under conditional independence)

Let f(S) = I(Y; X_S), with discrete views or finite differential entropies.

- (a) f is monotone for any joint law.
- (b) If X_1..X_N are conditionally independent given Y, f is submodular.
- (c) Without conditional independence, f need not be submodular.

**Proof.**

(a) By the chain rule, f(S ∪ {i}) − f(S) = I(Y; X_i | X_S) ≥ 0.

(b) Write f(S) = H(X_S) − H(X_S | Y).
- Under conditional independence, H(X_S | Y) = Σ_{i∈S} H(X_i | Y), which is modular.
- Joint entropy S ↦ H(X_S) is submodular. For S ⊆ T and i ∉ T, the marginal gain H(X_i | X_S) ≥ H(X_i | X_T), because conditioning reduces entropy.
- A submodular function minus a modular function is submodular.

(c) Let X_1 and X_2 be independent fair bits and Y = X_1 XOR X_2. Then f({2}) − f(∅) = I(Y; X_2) = 0, but f({1, 2}) − f({1}) = I(Y; X_2 | X_1) = 1 bit. The marginal gain grows, which violates submodularity. ∎

**Consequence.** Under (b), greedy selection of k views attains at least (1 − 1/e) of max_{|S|=k} f(S) (Nemhauser, Wolsey and Fisher, 1978).

**Empirical status.** Conditional independence given Y is **not testable from labels alone** on JaalTaka and is doubtful there: the six photos share the note, its wear and the lighting. The guarantee applies to mutual information, not to PRMVT's 0-1 accuracy. Measured greedy-style learned policies did not beat fixed order (`WEAK_RESULTS_FIX.md`).

---

## Theorem 5 (oracle gap: exact form and bounds)

Fix a trained model M. For each note, let C be the family of view subsets on which M is correct.

- **Label-reading subset oracle:** it is correct on a note iff C ≠ ∅.
- **Deployable policy ρ:** it picks a subset S_ρ from the views alone and is correct iff S_ρ ∈ C.

Define gap(ρ) = acc(oracle) − acc(ρ). Then:

- (i) gap(ρ) = P(C ≠ ∅ and S_ρ ∉ C) ≥ 0.
- (ii) gap(ρ) ≤ 1 − acc(ρ), with equality iff every note is solvable by some subset.
- (iii) For the fixed full-view policy, gap = P(M wrong on all N views but correct on some subset).

**Proof.** The oracle is correct exactly on the event {C ≠ ∅}, and ρ is correct exactly on {S_ρ ∈ C} ⊆ {C ≠ ∅}. Accuracy is the probability of each event, and the difference of nested events is the stated event, whose probability is non-negative. This gives (i). For (ii), gap ≤ 1 − acc(ρ) because acc(oracle) ≤ 1, with equality iff P(C ≠ ∅) = 1. (iii) is (i) with S_ρ = {1..N}. ∎

**Empirical check** (post-fix code, `results/qduig/oracle_20260929/`, n = 208):

| Model | Oracle correct | 6-view correct | Gap |
|---|---:|---:|---:|
| PRMVT seed 42 | 207 | 204 | 3 notes |
| Occlusion-robust seed 42 | 208 | 204 | 4 notes |
| Occlusion-robust seed 43 | 208 | 206 | 2 notes |
| Occlusion-robust seed 44 | 207 | 206 | 1 note |

Each gap equals (i) by construction. The oracle reads labels and is an analysis ceiling only.

---

## Theorem 6 (budgeted view selection is NP-hard)

**Problem VIEWSELECT.** Input: a finite joint law of (Y, X_1..X_N), given explicitly, a budget k, and a target a. Question: is there S with |S| = k whose Bayes accuracy 1 − R*(S) ≥ a?

VIEWSELECT is NP-hard. For the search version, the advantage over chance, 1 − R*(S) − 1/2, cannot be approximated within a factor better than (1 − 1/e) unless P = NP (Feige 1998). This is a statement about the advantage, not about accuracy itself: an affine shift does not preserve multiplicative approximation ratios. (Corrected 2026-09-29.)

**Proof (reduction from Max-k-Cover).** Take a Max-k-Cover instance: a universe U with |U| = m, sets A_1..A_N ⊆ U, a budget k, and a target t. Question: do k sets cover at least t elements?

Construct the law:
- Y is uniform on {0, 1}.
- A latent element u is uniform on U, independent of Y.
- View i is X_i = Y if u ∈ A_i, and the symbol ⊥ otherwise.

The law has 2m atoms, so it is polynomial in the input size.

For a set S of views:
- If u ∈ ∪_{i∈S} A_i, some view reveals Y exactly.
- Otherwise every view in S is ⊥, which is independent of Y, and the best guess is correct with probability 1/2.

So the Bayes accuracy from X_S is 1/2 + |∪_{i∈S} A_i| / (2m). With a = 1/2 + t/(2m), a yes-instance of Max-k-Cover maps to a yes-instance of VIEWSELECT and conversely. Max-k-Cover is NP-complete (it contains Set Cover), so VIEWSELECT is NP-hard. The advantage over chance is |∪_{i∈S} A_i| / (2m), a positive multiple of the coverage, so Feige's (1 − 1/e) threshold for Max-k-Cover transfers to the advantage exactly. ∎

**Scope.**
- This hardness is for general N with the law given as input.
- On JaalTaka, N = 6 is fixed, so all 63 subsets can be enumerated. That instance is easy.
- The theorem explains why view selection cannot be solved exactly at scale. It does not explain why the learned policies here lost to fixed order.

---

## Theorem 7 (optimal rejection)

Let η(x) = P(Y = 1 | x). A selective classifier answers with ŷ(x) = 1[η(x) ≥ 1/2] or abstains, and pays cost 1 per wrong answer and c ∈ [0, 1/2] per abstention.

- **(a) Chow's rule.** The Bayes-optimal rule answers iff m(x) = max(η(x), 1 − η(x)) ≥ 1 − c.
- **(b) Constrained version.** Among all (possibly randomised) rules with coverage exactly α, the rule that answers the α-fraction of inputs with the largest m(x) (randomising at ties) minimises the probability of an answered error.
- **(c) Monotonicity.** For any score s and threshold rule "answer iff s ≥ t", both the number of answered notes and the number of answered wrong notes are non-increasing in t, on every sample.

**Proof.**

(a) At a point x, answering costs the conditional error 1 − m(x), and abstaining costs c. The pointwise minimiser answers iff 1 − m(x) ≤ c. Minimising pointwise minimises the expectation.

(b) Let r(x) ∈ [0, 1] be the answer probability, with E[r] = α. The answered-error probability is E[r(x)(1 − m(x))]. Let r* be the top-α rule and r any other rule with the same coverage. Then E[(r* − r)(1 − m)] ≤ 0: wherever r* > r, m is at least the threshold value m_α; wherever r* < r, m ≤ m_α; and E[r* − r] = 0. This is the Neyman–Pearson exchange argument.

(c) The answered set {s ≥ t} shrinks as t grows, and so does its intersection with the wrong-prediction set. ∎

**Caveat.** (a) and (b) are optimal for the true posterior η. With a learned confidence p̂, the rule is optimal only to the extent that p̂ is calibrated. That is why calibration is reported (ECE 0.0136–0.0178 for PRMVT 6-view post-fix, `WEAK_RESULTS_FIX.md`).

**Finite-sample guarantee for a validation-chosen threshold.** Fix t on validation (A3). The test answered-error rate among the n_a answered test notes then satisfies the Clopper–Pearson / Wilson bound for a binomial proportion, because t is independent of the test sample.

**Empirical check** (`results/robustness/occlusion_decision.json`, t = 0.99 chosen on validation):

| Condition | Coverage | Wrong verdicts (share of all notes) |
|---|---:|---:|
| Clean | 96.2 % | 0.96 % |
| 55 % occlusion | 46.6 % | 0.48 % |

The low-light check and the whole-note check are in `SAFETY_FRAMEWORK.md` (2026-09-29 addendum).

---

# New results, 2026-09-29 (evening)

Three theorems and one proposition, each with a proof and a measured check. They formalise the one-sided jaal policy, its behaviour under shift, the image-quality gate, and the seed instability of fixed-view training.

## Theorem 8 (safety of the max-of-validation threshold)

**Setting.** Let S_1, …, S_n be the scores p(genuine) of n validation counterfeit notes. Let S_{n+1} be the score of a new counterfeit note. Assume the n+1 scores are exchangeable and almost surely distinct. The policy says "likely genuine" iff S > τ, where τ = max(S_1..S_n).

**(a)** P(S_{n+1} > τ) = 1/(n+1).

**(b)** Let π(τ) = P(S > τ | τ) be the true pass rate of the fixed threshold, and let F be continuous. Then π(τ) ~ Beta(1, n), so for every ε ∈ (0, 1):

  P_val( π(τ) > ε ) = (1 − ε)^n.

**Proof.**
(a) By exchangeability, each of the n+1 scores is equally likely to be the largest. S_{n+1} > τ exactly when S_{n+1} is the largest, which has probability 1/(n+1).
(b) U_i = F(S_i) are i.i.d. uniform, and π(τ) = 1 − max_i U_i. P(max U_i < 1 − ε) = (1 − ε)^n. ∎

**Numbers** (policy E, S4; `results/jaal_whole/policy.json`). n = 88 JaalTaka validation counterfeits.
- Expected pass rate 1/89 = 1.12 %.
- P(true pass rate > 5 %) = 0.95^88 = 0.011. With 98.9 % confidence over the validation draw, the in-domain pass rate is at most 5 %.
- Measured on the 88 test counterfeits: 0 passed, consistent with the theorem.

**Scope.** Exchangeability holds between JaalTaka validation and test notes (same collection, random note-disjoint split). It does not hold for whole-note photographs; see Theorem 9.

## Theorem 9 (the guarantee under distribution shift)

Let P and Q be the score distributions of counterfeit notes in the source (JaalTaka views) and target (whole-note camera photos) conditions. For any fixed τ,

  Q(S > τ) ≤ P(S > τ) + TV(P, Q),

and if the density ratio dQ/dP is at most w on {S > τ}, then Q(S > τ) ≤ w · P(S > τ).

**Proof.** {S > τ} is one event, and |P(A) − Q(A)| ≤ TV(P, Q) for every event A. The second bound is Q(A) = ∫_A (dQ/dP) dP ≤ w P(A). ∎

**Consequence.** Combining with Theorem 8, the whole-note pass rate is at most 1/(n+1) + TV(P, Q) in expectation. The shift term cannot be estimated from four physical counterfeit notes. That is why `JAAL_VERDICT_FIXED.md` reports the whole-note result (0 / 25 passed) as supporting evidence, not a guarantee.

**Measured shift.** The two conditions differ a lot. The looser threshold (99th validation percentile) passed 1 / 88 validation counterfeits but 16 / 25 whole-note counterfeit photos. The highest whole-note counterfeit score (0.99567) sits 2.4 log-odds below τ.

## Theorem 10 (clean false-rejection of the image-quality gate)

For one image statistic T (mean luma, saturated share, or near-black share), let the gate threshold t̂ be the empirical (1 − α) quantile of T over n clean validation views. Let the upper tail be rejected, or the lower tail for mean luma. Assume clean test views are i.i.d. with the validation views. Then with probability at least 1 − δ over the validation sample,

  P_clean(view rejected by T) ≤ α + √( ln(1/δ) / (2n) ).

For a note with k views and the three statistics, the union bound gives P_clean(note rejected) ≤ 3k (α + √(ln(1/δ)/(2n))).

**Proof.** The one-sided Dvoretzky–Kiefer–Wolfowitz inequality (Massart's constant) gives sup_t (F_n(t) − F(t)) ≤ √(ln(1/δ)/(2n)) with probability ≥ 1 − δ. At the empirical quantile, 1 − F_n(t̂) ≤ α, so 1 − F(t̂) ≤ α + √(ln(1/δ)/(2n)). The note-level bound is a union over the 3k view-statistic pairs. ∎

**Numbers** (δ = 0.05, α = 0.01; `results/safety/quality_gate_seed42.json`).
- One view, n = 208 validation views: each statistic rejects at most 0.01 + 0.085 = 9.5 % of clean views.
- Six views, n = 1,248: at most 0.01 + 0.035 = 4.5 % per view-statistic. The 18-way union makes the note-level bound vacuous.
- Measured: the gate alone costs 9 of 202 answered clean test notes at 1 view (202 → 193) and 24 of 208 at 6 views (208 → 184). Both are inside the per-statistic bounds.

**Reading.** The theorem is loose at the note level. It is included to show that the gate's clean cost is controlled by α and n, and that it grows with k.

## Proposition 11 (fixed-view training does not identify the fewer-view predictor)

**Setting.** A network h maps a masked view tuple (x_1..x_N, m) to a score. Fixed-view training uses only inputs with the full mask m = 1_N. Assume the hypothesis class H is rich on masked inputs: for any h ∈ H and any function g on inputs with m ≠ 1_N, there is h' ∈ H that equals h on full-mask inputs and equals g elsewhere. A mask-conditioned head, or per-count batch-norm whose k < N statistics are never trained, has this property.

**Claim.** The fixed-view training objective L(h) = E[ℓ(h(X, 1_N), Y)] takes the same value on h and h'. So the set of minimisers of L contains predictors with every achievable k-view risk R_k, k < N. Which one training returns is decided by initialisation, data order and optimisation, not by the objective.

**Proof.** L depends on h only through its values on full-mask inputs, where h and h' agree. So L(h) = L(h'). If h minimises L, so does h' for every choice of g on the k-view inputs, and R_k(h') ranges over all values attainable by some g. ∎

**What it predicts, and what is seen.** The fixed-view k-view risk should vary across seeds while the 6-view risk does not. In `SAME_ARCH_RESULTS.md` the fixed arms' one-view correct counts are 138, 131, 175 (shared BN) and 158, 194, 192 (per-count BN). Their six-view counts are 203–206 on every seed. The prefix-mixture objective puts weight π_k > 0 on every k, so it constrains the k-view values directly. That is Theorem 3, and it matches the observed 197–206 at one view on every seed.

**Scope.** Real networks share parameters between full-mask and masked inputs, so H is not fully rich. The proposition gives the limiting case: nothing in fixed-view training pushes the k-view predictor toward low risk. How far a real network drifts is empirical, and was measured.

---

# Serial and watermark results, 2026-09-30

## Proposition 12 (a serial blacklist cannot generalise across prints)

Let counterfeits come from prints j = 1..J with population shares π_j, and let a note carry its print's serial. A blacklist B holds the serials seen among training counterfeits. Its recall on a population with print shares π′ is Σ_{j∈B} π′_j, and it is 0 on any print not in B.

**Proof.** A counterfeit is flagged exactly when its serial is in B, which happens exactly when its print is in B. Take the expectation over π′. ∎

**Measured.** On the JaalTaka seed-42 test split, the blacklist catches 68 of 88 counterfeits, and 0 of the 19 with an unseen or unread serial (`results/watermark/hybrid.json`). On the serial-disjoint split, all 101 test counterfeits are from unseen prints, so its recall there is 0 by construction.

## Proposition 13 (how print sharing inflates note-disjoint accuracy)

Let q be the share of test counterfeits whose print is absent from training. The counterfeit catch rate on a note-disjoint split is (1 − q)·c_seen + q·c_unseen. A new deployment population, where every print is new, has catch rate c_unseen. The optimism of the note-disjoint estimate is (1 − q)(c_seen − c_unseen), which is non-negative whenever seen prints are easier.

**Proof.** Law of total probability over the event "print seen in training". ∎

**Measured.**
- Seed-42 split: q = 19/88. PRMVT at one view catches 67/68 seen versus 17/19 unseen (`JAALTAKA_SERIAL_AUDIT.md`).
- With five-fold CV of the ResNet-50 probe over all 1,390 notes, one-view accuracy is 98.3 % with note folds and 95.8 % with serial-grouped folds (`results/sota/cv_probe.json`).
- On the serial-disjoint split, where q = 1, the probe scores 85.1 % at one view (`results/watermark/serial_split_eval.json`).

## Proposition 14 (combining detectors: error bounds)

Let detectors D_1..D_m each flag "counterfeit".
- **OR rule** (flag if any flags): false-counterfeit rate on genuine ≤ Σ_i FCR_i (union bound), and miss rate ≤ min_i miss_i.
- **AND rule** (flag only if all flag): FCR ≤ min_i FCR_i, and miss rate ≤ Σ_i miss_i.

**Proof.**
- OR: a genuine note is flagged if some D_i flags it, so the union bound applies; a counterfeit is missed only if every D_i misses it, a subset of the event that the best one misses.
- AND: the argument is symmetric. ∎

**Use.** For an assistive device, where a false "counterfeit" harms the user, the AND rule (or a learned combination fitted to keep FCR low) is the safe design. The OR rule catches more counterfeits but adds false alarms.

**Measured.** The validation-fitted logistic combination of view 1 and the watermark window has test FCR 1/120 on the seed-42 split and 5/121 on the serial-disjoint split (`hybrid.json`, `serial_split_eval.json`).
