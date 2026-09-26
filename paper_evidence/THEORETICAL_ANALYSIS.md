# Theoretical analysis

These statements are definitions and short proofs. Numerical constants are not fitted. Where a quantity was not computed from the trained models, it is NOT_MEASURED.

## 1. View-count distribution shift

Let each note have an ordered tuple of views \((x_1,\ldots,x_N)\) and a label \(y\in\{0,1\}\). A prefix of length \(k\) is \((x_1,\ldots,x_k)\). Let \(K\) be the number of views the predictor is allowed to see.

**Definition (VCDS).** Training and test induce distributions \(P_{\mathrm{tr}}(K)\) and \(P_{\mathrm{te}}(K)\). View-count distribution shift is the condition \(P_{\mathrm{tr}}\neq P_{\mathrm{te}}\). The fixed-\(N\) special case is \(P_{\mathrm{tr}}(K=N)=1\) while \(P_{\mathrm{te}}(K<N)>0\).

**Proposition.** If a predictor \(f_N\) is applied unchanged to a prefix of length \(k<N\), its input is not a draw from the training distribution of \(f_N\). In particular, there is no guarantee that \(\mathbb{E}[\mathbf{1}\{f_N(x_{1:k})=y\}]=\mathbb{E}[\mathbf{1}\{f_N(x_{1:N})=y\}]\).

**Proof.** Under \(P_{\mathrm{tr}}(K=N)=1\), every training input contains views \(x_{k+1},\ldots,x_N\). The test input of length \(k\) does not. Equality of the two risks would be an extra assumption (invariance of \(f_N\) to the missing views). Nothing in the fixed-\(N\) training objective enforces that assumption. The seed-42 Q-DUIG run before prefix-robust training is the measured illustration: 6-view accuracy 0.9663 and 1-view accuracy 0.5865, from `FINAL_RESULTS.md`. That gap is an empirical instance, not a universal constant.

No finite-sample rate is claimed. A uniform deviation bound would need a complexity measure of \(f_N\) and was NOT_MEASURED.

## 2. Prefix-robust training

**Definition.** Prefix-robust training draws \(K\) from a distribution \(Q\) whose support is \(\{1,\ldots,N\}\) and minimizes the risk of \(f(x_{1:K})\). In the implemented mixer, a training step uses a random prefix or a random subset. \(Q(K=k)>0\) for each \(k\) is the design intent of that sampler; the exact \(Q\) is the sampler in `roboeye/qduig/engine.py` (`random_view_mask`), not a closed form written here.

**Proposition (support).** If \(Q(K=k)>0\) for every \(k\in\{1,\ldots,N\}\), then every view count that appears at test time in \(\{1,\ldots,N\}\) is inside the support of the training view-count distribution. Fixed-\(N\) training does not have this property.

**Proof.** By the definition of support. This removes the support mismatch in the definition of VCDS. It does not by itself prove a rate.

**What is not proved.** Non-convex SGD on this network has no finite-sample convergence guarantee computed here. Sample complexity is NOT_MEASURED. The claim that is used in the paper is the support proposition plus the measured seed-42 prefix-robust accuracies.

## 3. Oracle gap

Let \(S^\star\) be an oracle view set chosen with labels, and let \(S\) be the set chosen by a policy that does not see labels at test time. Let \(A(S)\) be accuracy when the classifier sees \(S\).

**Proposition.** \(0 \le A(S^\star)-A(S) \le 1\). If \(P(S=S^\star)\ge 1-\varepsilon\) and the classifier's conditional error on \(S^\star\) is at most \(\delta\), then the policy accuracy is at least \((1-\varepsilon)(1-\delta)\), so the gap to a perfect oracle on \(S^\star\) is at most \(\varepsilon+\delta-\varepsilon\delta\).

**Proof.** The first inequality is because both accuracies lie in \([0,1]\). For the second, condition on the event \(\{S=S^\star\}\), which has probability at least \(1-\varepsilon\). On that event the accuracy contribution is at least \(1-\delta\). On the complementary event the contribution is at least 0. Hence accuracy \(\ge (1-\varepsilon)(1-\delta)\). Subtracting from 1 gives the gap bound.

The measured oracle accuracy 0.9856 at mean 2.10 views and the learned 1-view policy accuracy 0.5865 are from the seed-42 artifacts cited in `FINAL_RESULTS.md`. They are not a proof that the bound is tight. OGPD did not close the adaptive-view gap; its v1 result is a fixed view count, not this policy comparison.

## 4. Generalization

A PAC-Bayes bound of the McAllester form says that, with probability at least \(1-\eta\) over the sample,
\[
\mathbb{E}_{h\sim\rho}[R(h)] \le \mathbb{E}_{h\sim\rho}[\hat R(h)] + \sqrt{\frac{\mathrm{KL}(\rho\|\pi)+\ln(2\sqrt{n}/\eta)}{2n}},
\]
where \(\pi\) is a prior, \(\rho\) a posterior, \(n\) the sample size, and \(R,\hat R\) the true and empirical risk.

**Status.** Section 5 evaluates this expression for a Gaussian posterior on the saved prefix-robust weights. On the fixed prior grid the penalty exceeds 1, so the bound is vacuous. That evaluation is not a non-vacuous guarantee for JaalTaka.

## LaTeX

```latex
\begin{definition}[VCDS]
Training and test view-count laws differ:
\(P_{\mathrm{tr}}(K)\neq P_{\mathrm{te}}(K)\).
The fixed-\(N\) case is \(P_{\mathrm{tr}}(K=N)=1\) with \(P_{\mathrm{te}}(K<N)>0\).
\end{definition}

\begin{proposition}[Support]
If the training law \(Q\) satisfies \(Q(K=k)>0\) for every \(k\in\{1,\ldots,N\}\),
then every test view count in that set is in the support of \(Q\).
\end{proposition}

\begin{proposition}[Oracle gap]
If \(P(S=S^\star)\ge 1-\varepsilon\) and the conditional error on \(S^\star\) is at most \(\delta\),
then the gap from perfect accuracy on \(S^\star\) is at most \(\varepsilon+\delta-\varepsilon\delta\).
\end{proposition}
```

## 5. Numerical bounds computed from the split

These numbers are consequences of the stated inequalities and the split size. They are not a fit to the test accuracy.

### Sample complexity for a fixed predictor

Assume notes are i.i.d. and the predictor is chosen before seeing them. Hoeffding's inequality gives
\[
n \ge \frac{\ln(2/\delta)}{2\varepsilon^2}
\]
for an additive accuracy deviation of \(\varepsilon\) with probability at least \(1-\delta\).

For \(\varepsilon=0.05\) and \(\delta=0.05\), that expression equals **738** notes. The training split has **974** notes (`split_metadata.json`). 974 is at least this fixed-predictor count. This does not bound prefix-robust SGD. The hypothesis class of the network was not measured, so a uniform-convergence sample size for the training algorithm is NOT_MEASURED.

Fixed-view training and prefix-robust training use the same notes. The difference is the support of \(K\), not a smaller \(n\). No claim is made that prefix-robust training needs fewer notes.

### PAC-Bayes

Posterior \(\rho = \mathcal{N}(w, \rho^2 I)\) is centered at the saved seed-42 prefix-robust weights. Prior \(\pi = \mathcal{N}(0, \sigma_0^2 I)\). The grid is \(\sigma_0 \in \{0.1, 1, 10\}\) and \(\rho \in \{0.001, 0.01, 0.1\}\), fixed in `scripts/eval/compute_pacbayes.py` before the KL was evaluated. A union bound adds \(\ln 9\) inside the square root. \(n = 974\) training notes. The test set is not used.

The checkpoint has 4,121,404 floating parameters and squared weight norm 129211.71963995504. The smallest McAllester penalty on that grid is **57.5892990573559**, at \(\sigma_0 = 0.1\), \(\rho = 0.1\), with KL 6460585.981997751. Every grid point has a penalty above 1, so the bound exceeds 1 for any empirical risk in \([0, 1]\). The numerical PAC-Bayes statement for this posterior is that the bound is vacuous. Source: `results/theory/pacbayes_prmvt_seed42.json`. The expected Gibbs risk was not needed once the penalty alone exceeded 1.

### Oracle gap, measured

Fixed-count Q-DUIG, seed 42, test, n=208:

- 1 view: accuracy 0.5865384615384616 from `results/qduig/eval/seed42/policies/full_proposed_1view/test_metrics.json`
- 6 views: accuracy 0.9663461538461539 from `results/qduig/eval/seed42/policies/full_proposed_6view/test_metrics.json`
- Difference, 6-view minus 1-view: 0.3798076923076923

That is the measured VCDS gap for this checkpoint. It is not a universal constant.

Oracle subset, same split, from `results/qduig/eval/seed42/oracle.json`:

- Oracle accuracy 0.9855769230769231 at mean 2.0961538461538463 views
- Learned 6-view accuracy 0.9663461538461539 at 6 views
- Oracle minus learned accuracy: 0.019230769230769273

The proposition in section 3 still has no numerical \(\varepsilon\) and \(\delta\) fitted to this policy. The file marks the oracle as analysis only.
