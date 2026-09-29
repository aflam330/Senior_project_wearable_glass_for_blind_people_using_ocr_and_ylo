# Quality loop, 2026-09-29

One pass. Weights already selected on validation were left in place. A new training run replaces a checkpoint only when validation improves under a rule written before the test set is scored. The runs already on disk for occlusion (median fill, test-time crops, inpainting, matched fine-tune) did not produce a validation win that replaced those checkpoints. See `OCCLUSION_FIX_RESULTS.md` and `OCCLUSION_INPAINT.md`.

## Scores read back from artifacts

| component | measured value | standing decision |
| --- | --- | --- |
| Project Python compile | 168 files, 0 errors | `final_pass_20260929.json` |
| Claim registry | 122 claims, validation exit 0 | `scripts/validate_claims.py` |
| Split leakage | 0, 0, 0 | `split_metadata.json` |
| PRMVT 1-view test | 202/208 = 0.9711538461538461 | keep prefix_ft seed 42 |
| PRMVT 6-view test | 203/208 = 0.9759615384615384 | keep |
| MTPT 1-view test | 204/208 = 0.9807692307692307 | keep the 6+3 checkpoint |
| VCIE 1-view test | 193/208 = 0.9278846153846154 | keep the k1 checkpoint |
| Occlusion ensemble test | 185/208 = 0.8894230769230769 | keep; 0.90 would be 188/208 |
| Rejection wrong-share, occluded test | 0.004807692307692308 | keep the validation threshold 0.99 |
| Synthetic box mean IoU | 0.9083, IoU ≥ 0.5 on all 2052, false alarms 0/230 | keep `best.pt` |
| D1-network McAllester | 0.11271182900151713, test unused | keep; not a bound on published PRMVT |
| External BDT images on disk | denomination and coin folders counted in `comparison_dataset_manifest.json` | authenticity not computed on those labels |
| ModelNet render, 40 epochs | test 0.64 at 1 view and 0.77 at 6 views, both trainings | `subset10_seed42_e40.json`; prefix validation peaked at epoch 34 |
| Pi 5, phone, user study | no device numbers | protocols stay READY_FOR_DEVICE |

## Stop

No validation score in this pass moved. The new outputs are the Wilson intervals, the oracle count check, the compile count, and the download log. Another training loop on the same objectives would repeat experiments that already left the checkpoints unchanged.

---

## Correction 2026-09-29: PRMVT numbers after the NaN fix

The PRMVT rows above read `results/qduig/prefix_ft/seed42/test_views/`, written before the
NaN-entropy fix of 2026-09-28. The re-evaluated file `test_views_20260928/views_1_to_6.json` gives,
on the same 208 test notes: 1 view 202/208 = 0.9712 (unchanged), 2-4 views 204/208 = 0.9808,
5 views 205/208 = 0.9856, 6 views 204/208 = 0.9808 (was 203). The 1-to-6-view drop is therefore
-2 notes (6 views is two notes better), not one note. Over seeds 42-44 the means are 96.47 / 98.40 /
98.08 / 97.60 / 98.56 / 98.08 % (`WEAK_RESULTS_FIX.md`).


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
