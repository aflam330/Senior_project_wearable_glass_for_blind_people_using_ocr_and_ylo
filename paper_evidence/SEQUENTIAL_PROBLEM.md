# Sequential multi-view authentication

## Problem

An agent receives views of one object in a fixed order. After each view it may stop and predict a label, or continue, up to a maximum of N views. The deployed objective on this project is accuracy minus a cost times the number of views. The cost weight used for PRMVT was chosen on validation (`prefix_stop_policy.json`, lambda 0.02).

N = 6 on JaalTaka, so every prefix can be scored. The difficulty below is for general N, when the view set is part of the input.

## Subset choice is hard for general N

Cardinality-constrained maximum coverage is NP-hard (Feige, 1998, building on Nemhauser, Wolsey, and Fisher). If each view is an arbitrary set and the gain of a set of views is the coverage of their union, choosing k views of maximum gain is that problem. Sequential authentication with an arbitrary set function and a budget on the number of views therefore contains an NP-hard special case.

On JaalTaka, N = 6 is fixed and the prefixes were scored directly. That instance is not the hard case.

## Information gain is not submodular in general (corrected 2026-09-29)

An earlier version of this section claimed that f(S) = I(Y; X_S) is always submodular. That is false.
Counterexample: X_1, X_2 independent fair bits and Y = X_1 XOR X_2. Then I(Y; X_2) = 0, but
I(Y; X_2 | X_1) = 1 bit, so adding X_2 gains more after X_1 is known than before. The marginal gain
grew, which submodularity forbids.

What is true: if the views are conditionally independent given Y, then I(Y; X_S) is monotone and
submodular (Krause and Guestrin, 2005), and greedy selection under a cardinality budget attains
1 - 1/e of the optimum (Nemhauser, Wolsey and Fisher, 1978). Six photographs of one physical note
share lighting, wear and the note itself, so conditional independence given the genuine/counterfeit
label is an assumption, not a measured fact, on JaalTaka. The 1 - 1/e guarantee therefore applies to
this task only under that assumption, and only to mutual information, not to the 0-1 accuracy of PRMVT.

## What was measured

From `results/qduig/prefix_ft/seed42/prefix_stop_policy.json`, test n = 208:

| policy | accuracy | mean views |
| --- | ---: | ---: |
| always 1 view | 0.9711538461538461 | 1.0 |
| lambda 0.02, validation-chosen | 0.9711538461538461 | 1.0096153846153846 |
| always 6 views | 0.9759615384615384 | 6.0 |

The label-using subset oracle in `results/qduig/eval/seed42/oracle.json` scores 0.9855769230769231 at 2.0961538461538463 views. That oracle is not a policy. The gap from the learned row in that same file (0.9663461538461539 at 6 views) is 4 notes. It is a different operating point from the lambda-0.02 row.

---

## Correction 2026-09-29: which model the oracle file describes

`results/qduig/eval/seed42/oracle.json` was produced by `scripts/evaluate_all.py` for the early
`results/qduig/proposed` model (its 6-view accuracy, 0.966, is that model's), not for the deployed
PRMVT. It was also computed before three evaluation bugs were fixed on 2026-09-28 (missing view
self-gate in `run_oracle`, NaN entropy for confident notes, masked-view volume; see
`WEAK_RESULTS_FIX.md`). The oracle was re-run on the deployed checkpoints with the fixed code,
test split, n = 208 (`results/qduig/oracle_20260929/`):

| checkpoint | oracle accuracy | oracle mean views | 6-view accuracy | gap (notes) | unsolvable by any subset |
|---|---:|---:|---:|---:|---:|
| PRMVT prefix_ft seed 42 | 0.9952 (207) | 1.00 | 0.9808 (204) | 3 | 1 |
| occlusion-robust seed 42 | 1.0000 (208) | 1.00 | 0.9808 (204) | 4 | 0 |
| occlusion-robust seed 43 | 1.0000 (208) | 1.00 | 0.9904 (206) | 2 | 0 |
| occlusion-robust seed 44 | 0.9952 (207) | 1.00 | 0.9904 (206) | 1 | 1 |

The oracle reads the label, so these are analysis ceilings, not policies. For the deployed model
almost every note has at least one single view it classifies correctly; the "three notes no subset
can solve" in the section above belongs to the early model and to the pre-fix code.


---

## Correction 2026-09-29: PRMVT numbers after the NaN fix

The PRMVT rows above read `results/qduig/prefix_ft/seed42/test_views/`, written before the
NaN-entropy fix of 2026-09-28. The re-evaluated file `test_views_20260928/views_1_to_6.json` gives,
on the same 208 test notes: 1 view 202/208 = 0.9712 (unchanged), 2-4 views 204/208 = 0.9808,
5 views 205/208 = 0.9856, 6 views 204/208 = 0.9808 (was 203). The 1-to-6-view drop is therefore
-2 notes (6 views is two notes better), not one note. Over seeds 42-44 the means are 96.47 / 98.40 /
98.08 / 97.60 / 98.56 / 98.08 % (`WEAK_RESULTS_FIX.md`).
