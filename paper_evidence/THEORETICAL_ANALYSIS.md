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

**Status.** The KL term for a posterior over any of these networks was NOT_MEASURED. The display above is the standard inequality, not a number for JaalTaka. Plugging in \(n=974\) training notes without a KL would not be a bound.

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
