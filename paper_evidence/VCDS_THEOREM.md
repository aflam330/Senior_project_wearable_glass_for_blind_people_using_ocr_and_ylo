# View-count distribution shift

Numbers below are either proved from the stated assumptions or read from a saved file. A finite trained network is not the Bayes classifier.

## Definition

An instance is (X_1, ..., X_N, Y). The full-view law is the joint distribution of that tuple. The k-view observation is (X_1, ..., X_k). If a network was built for N inputs, the missing views are replaced by the fill used at evaluation (masked or empty). View-count distribution shift is the move from scoring under all N views to scoring under the first k.

## Assumptions

A1. One physical object produces the views. The views need not be independent.

A2. The label Y is a property of the object. It does not change when views are dropped.

A3. The hypothesis class and the checkpoint rule are fixed before the test split is scored. On JaalTaka the checkpoint was chosen on validation.

## Proposition 1 (Bayes error)

Let R*(S) be the minimum error of any predictor that sees only the variables in S. If the k-view observation is a function of the N-view observation, then R*(k) >= R*(N).

Proof. Any predictor of Y from the k-view observation is also a predictor from the N-view observation, because the k views are contained in the N views. The Bayes predictor for N is the best such function. Therefore its error is at most the error of every k-view predictor, including the Bayes predictor for k.

This does not say that a particular trained network has higher error at smaller k.

## What the JaalTaka networks do

Source files: `results/qduig/eval/seed42/baseline/baseline_{k}view/test_metrics.json` and `results/qduig/prefix_ft/seed42/test_views/views_1_to_6.json`. n = 208.

| k | baseline accuracy | PRMVT accuracy |
| ---: | ---: | ---: |
| 1 | 0.7355769230769231 | 0.9711538461538461 |
| 2 | 0.8701923076923077 | 0.9759615384615384 |
| 3 | 0.9134615384615384 | 0.9759615384615384 |
| 4 | 0.9182692307692307 | 0.9759615384615384 |
| 5 | 0.8990384615384616 | 0.9807692307692307 |
| 6 | 0.9182692307692307 | 0.9759615384615384 |

Baseline correct counts are 153 and 191 at 1 and 6 views. The drop is 38 notes. The stored accuracies differ by 0.1826923076923076. PRMVT correct counts are 202 and 203. The drop is 1 note, 0.004807692307692308. At k = 5, PRMVT is correct on 204 notes, one more than at k = 6. Proposition 1 is about Bayes error. This network's error is not monotone in k.

## Proposition 2 (estimation)

Fix a classifier h before seeing n i.i.d. notes. With probability at least 1 - delta,

|R(h) - R_n(h)| <= sqrt( log(2/delta) / (2n) ).

Proof. The 0-1 loss of h on each note is bounded in [0, 1]. Hoeffding's inequality gives the display.

For n = 208 and delta = 0.05 the right-hand side is 0.09416739715938784 (`paper_evidence/vcds_theorem_constants.json`). That is the O(1/sqrt(n)) term for one fixed classifier. It is not a proof that prefix training drives the view-count drop to zero. The measured PRMVT drop, 0.004807692307692308, sits inside that band. The baseline drop, 0.1826923076923076, sits outside it. The two inputs (1 view and 6 views) are different functions, so the single-classifier band does not force them to be close. The comparison shows that the fixed-view baseline's drop is larger than sampling noise of this size, and the prefix-trained drop is not.

**Correction (2026-09-29).** That sentence does not follow. Both R̂_1 and R̂_6 are estimates, each within ε = 0.0942 of its true value with the stated probability. Separating R_1 from R_6 with these intervals alone needs a gap above 2ε = 0.188 (0.229 with the union bound over six view counts). The baseline's 0.1827 does not exceed it. The paired McNemar test on the same notes is the valid evidence (`THEOREMS.md`, Theorem 2; `STATISTICAL_ANALYSIS.md`).

## Scope

Proposition 1 holds for any multi-view task obeying A1 and A2. It does not by itself measure a drop on a dataset that has no aligned views. Those datasets are listed in `CROSS_DATASET_DOWNLOAD_LOG.md`.

A 10-class rendered subset of ModelNet40 was trained for 40 epochs with pinned initialization (`results/vcds_modelnet/subset10_seed42_e40.json`). Fixed-view and prefix training both score 0.64 at 1 view and 0.77 at 6 views on the 100-object test. The prefix run is higher at 2 views (0.73 versus 0.70) and at 3 views (0.77 versus 0.73). Shorter unseeded runs are kept as `subset10_seed42.json` and `subset10_seed42_e15.json`.

---

## Correction 2026-09-29: PRMVT numbers after the NaN fix

The PRMVT rows above read `results/qduig/prefix_ft/seed42/test_views/`, written before the
NaN-entropy fix of 2026-09-28. The re-evaluated file `test_views_20260928/views_1_to_6.json` gives,
on the same 208 test notes: 1 view 202/208 = 0.9712 (unchanged), 2-4 views 204/208 = 0.9808,
5 views 205/208 = 0.9856, 6 views 204/208 = 0.9808 (was 203). The 1-to-6-view drop is therefore
-2 notes (6 views is two notes better), not one note. Over seeds 42-44 the means are 96.47 / 98.40 /
98.08 / 97.60 / 98.56 / 98.08 % (`WEAK_RESULTS_FIX.md`).


See the 2026-09-29 update in `VCDS_UNIVERSAL.md`: with three seeds and 184 test objects, the ModelNet fixed-view model loses 10.5 points at 1 view that prefix training recovers, which supersedes the single-seed ModelNet paragraph above.
