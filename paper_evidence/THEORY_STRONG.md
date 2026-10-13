# What can be proved, and what cannot (2026-10-13)

This note covers the six items asked for. Each one is given as a statement with its assumptions and a proof, or as a plain "this is not a theorem" with the reason. Earlier notes are not replaced: `VCDS_THEOREM.md`, `THEORY_FULL.md`, `THEOREMS.md`.

A proof here is a statement about a model. Whether the model fits the data is a separate, empirical question, and it is answered under each item.

---

## 1. Safe threshold (proved; standard)

**Proposition 1.** Let the scores of n validation counterfeits and of one new counterfeit be exchangeable, with no ties. Let τ be the largest validation score. Then P(new score > τ) = 1/(n + 1).

*Proof.* Under exchangeability every ordering of the n + 1 scores is equally likely, so the new score is the largest of the n + 1 with probability 1/(n + 1). It exceeds τ exactly in that case. ∎

With n = 88, the probability is 1/89 = 1.1 %. This is the rank argument behind split conformal prediction.

**Does the assumption hold?** Only for counterfeits like the validation ones. A new print or a new camera breaks exchangeability, and the bound then grows by at most the total-variation distance between the two score distributions, which the data cannot measure. On the 20 whole-note counterfeit photographs, none passed (Wilson upper limit 16.1 %).

---

## 2. UV fibre method: what is invariant (proved), and what is not

Notation from `savior_glass/modes/uv_check.py`: a = median(G/B), b = median(R/B) over the lit pixels of the note; E = max(G − aB, R − bB); r = tophat(E) / (blur(B) + 1).

**Proposition 2 (lamp intensity).** Suppose the camera answers linearly and no pixel is clipped. If the lamp intensity is multiplied by c > 0, then a and b are unchanged, E is multiplied by c, and r is multiplied by c·(blur(B) + 1)/(c·blur(B) + 1), which tends to 1 as blur(B) grows.

*Proof.* All three channels are multiplied by c. Ratios G/B and R/B do not change, so their medians do not. E is positively homogeneous of degree 1: max(cG − a·cB, cR − b·cB) = cE. The white top-hat (image minus its opening) is also homogeneous of degree 1, and so is the Gaussian blur. The ratio follows. ∎

So the fibre map does not depend on lamp brightness or distance, up to the "+1" that guards against division by zero. In the test, halving the brightness left acceptance at 29 of 43.

**Proposition 3 (colour balance: not invariant).** If the camera multiplies the channels by gains (g_B, g_G, g_R), then on paper pixels E stays 0, but on a fibre the green part of the residual becomes g_G·(G − aB) while blur(B) becomes g_B·blur(B). The contrast r of a fibre is therefore multiplied by about g_G / g_B (or g_R / g_B for the red part).

*Proof.* The new ratio is a′ = a·g_G/g_B, so G′ − a′B′ = g_G·G − a·(g_G/g_B)·g_B·B = g_G·(G − aB). The denominator scales by g_B. ∎

**Consequence.** An earlier description said the camera's colour balance "cancels out". That was wrong and has been corrected in the code and the feature summary. The fixed threshold `UV_REL_THRESHOLD` holds for one camera setting; in the test, a colour cast (blue × 0.8, green × 1.2) moved acceptance from 29 to 34 of 43, as Proposition 3 predicts (gain ratio 1.5).

---

## 3. UV fibre count: a bound, under a model the data only partly support

**Model.** In one capture condition (fixed distance, focus and side of the note), the number of fibres the method finds on a genuine note is Poisson with mean μ, and on a fibre-free sheet Poisson with mean ν < μ. The note is accepted when the count is at least T.

**Proposition 4.** For ν < T < μ:

- P(genuine note rejected) = P(Poisson(μ) < T) ≤ e^(−μ) (eμ/T)^T,
- P(fibre-free sheet accepted) = P(Poisson(ν) ≥ T) ≤ e^(−ν) (eν/T)^T.

*Proof.* Both are the Chernoff bounds for a Poisson variable, from Markov's inequality applied to e^(sX) with the optimal s = ln(T/μ) or ln(T/ν). ∎

Example with T = 6. A condition with μ = 20 gives a rejection bound of 0.0011 (the exact Poisson value is 0.00007). A fibre-free sheet with ν = 1 gives an acceptance bound of 0.0032 (exact 0.0006).

**Does the model hold?** Only inside one capture condition. Across the 111 photographs the counts run from 1 to 366 with a median of 41, far more spread than any single Poisson mean allows, because distance, focus and the side of the note change μ. There is no measurement of ν on printed paper at all: the 71 fibre-free patches are smooth surfaces and all gave 0. The bound therefore says how the error falls with μ once the capture is controlled (a hood at a fixed distance); it does not give the error rate of the present photographs. That rate is the measured one: 103 of 111 accepted.

**Several frames.** If K frames were independent, the chance that all miss would be the single-frame chance to the power K. Frames of one note held in one hand are not independent, so this is an upper limit on the benefit, not a guarantee.

---

## 4. DMWR: what is optimal about it, and what is not

**Proposition 5 (quotient image).** Let I = L·t with L > 0, and let G be a non-negative averaging kernel. If L varies by at most a relative amount ε inside the kernel's support around a pixel x, that is (1 − ε)·L(x) ≤ L(y) ≤ (1 + ε)·L(x), then

  (1/(1 + ε)) · t(x)/(G∗t)(x) ≤ I(x)/(G∗I)(x) ≤ (1/(1 − ε)) · t(x)/(G∗t)(x).

*Proof.* (G∗I)(x) = Σ G(x − y) L(y) t(y) lies between (1 − ε)L(x)(G∗t)(x) and (1 + ε)L(x)(G∗t)(x), because G and t are non-negative. Divide I(x) = L(x)t(x) by these bounds. ∎

So the quotient image removes the lamp exactly when the lamp is uniform (ε = 0) and up to a factor 1/(1 ± ε) otherwise. A Gaussian has infinite support; the statement applies to its truncation, with the tail mass as an extra error.

**Proposition 6 (the residual is the right thing to look at, under a Gaussian model).** Suppose a genuine quotient image is Q = T_d + n and a counterfeit is Q = T_d + s + n, with n white Gaussian noise of variance σ². Then (a) for a known deviation s, the most powerful test at any false-alarm level thresholds ⟨Q − T_d, s⟩, a matched filter on the residual R = Q − T_d; (b) for an unknown s, the generalised likelihood-ratio test thresholds ‖R‖², the energy of the residual.

*Proof.* (a) is the Neyman–Pearson lemma: the log-likelihood ratio is (⟨R, s⟩ − ‖s‖²/2)/σ², monotone in ⟨R, s⟩. (b) The likelihood under the alternative is maximised at ŝ = R, giving a ratio monotone in ‖R‖². ∎

**What this does and does not say.** DMWR's inputs include the residual R, its standard deviation (the energy statistic of (b)) and the correlation with the portrait. So the statistic that is optimal under this model is among its features. It does not follow that the trained DMWR network is optimal: the noise on real notes is not white (wear, folds, print), the deviation s differs by print, and the network is fitted on 867 notes. The measured facts are that both streams together beat either alone (91.5 % against 85.8–88.0 %) and that a larger fine-tuned MobileNetV2 is not significantly different (92.9 %; McNemar p = 0.375, 0.688, 0.375).

---

## 5. View-count shift (VCDS) and print-disjoint generalisation

These are in the earlier notes and are not re-derived here.

- **View-count shift** (`VCDS_THEOREM.md`): a model trained only on six-view inputs is unconstrained on inputs with fewer views, so its risk there is not bounded by its six-view risk. The measured side is the controlled comparison: 71.2 ± 11.4 % against 98.2 ± 0.7 % at one view, same network. On MVP-N the benefit depends on the fusion head, so it is not universal.
- **Print-disjoint uncertainty** (Theorem 16 in `JOURNAL_PAPER.md` and `PRINT_DISJOINT_DATASET_PLAN.md`): when notes of one print are strongly correlated, the effective number of independent test items is close to the number of prints (about 24), not the number of notes (101). Differences of a few points between methods cannot be established.

## 6. Cross-currency transfer: not a theorem

No second currency with back-lit or UV photographs and counterfeit labels has been measured. Anything stated about transfer would be a guess. What can be said is structural: Propositions 2, 3 and 5 do not mention Taka, so the lamp removal carries over to any note; the portrait T_d and every threshold have to be built again for each currency.
