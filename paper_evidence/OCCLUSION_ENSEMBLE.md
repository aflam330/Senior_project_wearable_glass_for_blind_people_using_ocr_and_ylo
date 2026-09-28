# Occlusion: multi-seed ensemble and flip averaging (attempts G and F, 2026-09-28)

Scripts: `scripts/eval/occlusion_probs.py` saves probabilities, and `scripts/eval/occlusion_decide.py` applies these rules, fixed before looking at test:
1. Flip averaging for a model only if it raises that model's occluded validation accuracy.
2. The ensemble (mean probability of the three occlusion-robust seeds) only if it beats seed 42 on occluded validation.

Raw: `results/robustness/occlusion_decision.json`. Test is 208 notes, 55% occlusion, 6 views.

| Model | Flip used (val decision) | Val occ | Test clean | Test occ |
|---|---|---:|---:|---:|
| seed 42 | yes | 91.8 | 99.0 | 87.5 (88.5 without flip) |
| seed 43 | no | 90.9 | 99.0 | 88.9 |
| seed 44 | no | 88.9 | 99.0 | 84.6 |
| **3-seed ensemble** (chosen on val) | | **92.8** | **99.0** | **88.9** |

The ensemble won on validation (92.8% vs 91.8%). On test it equals the best single seed (88.9%), still below 90%. Flip averaging helped seed 42 on validation but cost it one point on test. Both results are reported as they came out.
