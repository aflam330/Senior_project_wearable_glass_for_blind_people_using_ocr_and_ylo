# Safety-aware assistive decisions

For a person who cannot see the note, a wrong genuine or counterfeit verdict is the costly event. A request to reposition the note is the alternative. The score used in production is confidence = max(p, 1-p). A note is answered when confidence is at least a threshold t. Otherwise the system asks for another placement.

## Monotone threshold rule

Let W(t) be the number of notes that are both answered and wrong, and C(t) the number answered. If rejection is exactly the event confidence < t, then as t increases the answered set shrinks. Every answered mistake at the higher threshold was already an answered mistake at the lower threshold. So W(t) and C(t) are nonincreasing in t. This holds on every split, because it uses only the definition of the threshold.

## Rule used

`results/robustness/occlusion_decision.json` records the rule written before the test score: t is the largest value on a 0.50–0.99 grid that still answers at least 95% of clean validation notes. Among grid values that meet that validation constraint, the largest t minimizes W(t) on every split at once, including the test split, by the monotonicity above. It does not claim to be the best rejector among rules that are not thresholds on this confidence.

## Measured test, n = 208

| condition | answered | accuracy on answered | wrong-verdict share of all notes |
| --- | ---: | ---: | ---: |
| clean | 0.9615384615384616 | 0.99 | 0.009615384615384616 |
| 55% occlusion | 0.46634615384615385 | 0.9896907216494846 | 0.004807692307692308 |

Source: `rejection` in `occlusion_decision.json`. The threshold stored there is 0.99.

Low-light and other corruptions were measured as accuracy, not as this same rejection table. Those accuracy curves stay in `SEVERITY_CURVES.md`. A user study of whether a reposition request is acceptable is READY_FOR_DEVICE (`USER_STUDY_PROTOCOL.md`).

---

## Addendum 2026-09-29 (evening): low light, over-exposure, and an image-quality gate

**Confidence rejection alone is not enough under bad input.** PRMVT seed 42; the threshold is chosen on clean validation (largest t on 0.50–0.99 answering ≥ 95 % of clean validation notes) and applied unchanged to corrupted test notes (`results/safety/rejection_seed42.json`, `scripts/eval/eval_safety_rejection.py`). n = 208.

| condition | 1 view (t = 0.87): answered / wrong | 6 views (t = 0.97): answered / wrong |
|---|---:|---:|
| clean | 195 / 3 | 185 / 2 |
| low light 0.35 | 182 / **59** | 152 / **36** |
| low light 0.2 | 139 / **71** | 78 / **43** |
| occlusion 0.55 (black box) | 114 / **51** | 17 / 3 |
| over-exposure 2.2 | 99 / **38** | 103 / **44** |

A corrupted image can still produce a confident answer. The monotone-threshold argument above holds, but it only says that a higher t gives fewer wrong answers. It does not say how many.

**Image-quality gate** (`scripts/eval/eval_quality_gate.py` → `results/safety/quality_gate_seed42.json`). Rules were fixed on clean validation views before any test number was read:
- A view is **too dark** if its mean luma is below the 1st percentile of clean validation views: 109.3 (1-view rule), 80.5 (6-view rule).
- A view is **blown out** if its share of pixels at luma ≥ 250 exceeds the 99th percentile: 0.094 / 0.137.
- A view is **blocked** if its share of pixels at luma ≤ 5 exceeds the 99th percentile: 0.039 / 0.041.
- A note is answered only if every view passes **and** max(p, 1 − p) ≥ t, with t re-chosen under the same 95 % clean-validation rule with the gate applied (t = 0.65 at 1 view, 0.50 at 6 views).

| condition | 1 view: confidence only → gate + confidence (answered / wrong) | 6 views: confidence only → gate + confidence |
|---|---|---|
| clean | 202 / 5 → **193 / 5** | 208 / 4 → **184 / 3** |
| low light 0.35 | 197 / 65 → **0 / 0** | 208 / 73 → **0 / 0** |
| low light 0.2 | 199 / 86 → **0 / 0** | 208 / 86 → **0 / 0** |
| occlusion 0.55 | 178 / 86 → **0 / 0** | 208 / 100 → **0 / 0** |
| over-exposure 2.2 | 178 / 78 → **0 / 0** | 208 / 107 → **0 / 0** |

With the gate, no corrupted test note got a wrong verdict: 0 / 208 in each condition, 95 % Wilson upper bound 1.8 %. On clean notes it answers 193 (1 view) and 184 (6 views) of 208. Rejected notes get "improve the light / move your finger" instead of a verdict.

**Limits.**
- These corruptions are severe (brightness × 0.35 or × 0.2, × 2.2), so every corrupted note falls outside the gate. Milder corruption was not tested. Some of it will pass the gate and then depends on confidence rejection.
- The occlusion is a black box, which the near-black test finds trivially. A real finger is not black.
- The thresholds are fitted to JaalTaka photographs. On the glass camera they must be re-fitted on its own clean images before use. The gate is therefore evaluated here and not yet switched on in `savior_glass`.
