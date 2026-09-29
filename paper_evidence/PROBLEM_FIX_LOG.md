# Problem log, 2026-09-29

Each item keeps the existing checkpoint unless a validation rule written in advance accepts a replacement.

## ModelNet subset size

`train_modelnet_vcds.py` first asked for 64 + 16 + 16 objects in class `bowl`. That class has 84 clouds. The script stopped before any test score. The split was changed to 40 + 10 + 10, which fits `bowl`, and the run was repeated. Output: `results/vcds_modelnet/subset10_seed42.json`.

At epoch 5 the prefix validation mean was still rising. Longer runs are `subset10_seed42_e15.json`, `subset10_seed42_e25.json`, and `subset10_seed42_e40.json`. The 40-epoch run pins initialization. Its prefix validation mean peaked at 0.8066666666666666 and was lower at epoch 40, so that selection score had stopped rising. The 40-epoch test file is the standing ModelNet render result.

## Full-network PAC-Bayes

The zero-mean prior on the published PRMVT weights has McAllester penalty 57.5892990573559. Wider posterior standard deviations, an initialization prior, a linear head, and a data-dependent prior on a network trained on 487 notes were already run. The data-dependent prior gives McAllester 0.11271182900151713 on the other 487 notes. That bound stays. It is not a certificate for the published test accuracy. Detail: `PACBAYES_FIX_RESULTS.md`.

## Occlusion at 0.55

Median fill, test-time crops, a learned hole-filler, and a matched fine-tune are already on disk. The standing full-test number is the 3-seed ensemble, 185/208 = 0.8894230769230769. A rate of 0.90 on this file is 188/208. The ensemble checkpoint was not replaced. Detail: `OCCLUSION_FIX_RESULTS.md`, `occlusion_decision.json`.

## Extra Bangladeshi folders

They are denomination or coin labels, not genuine versus counterfeit. No authenticity accuracy was assigned to them. Counts: `comparison_dataset_manifest.json`.

## Bangla Money test files

`Testing` contains numbered images and no class folders. Denomination counts were taken from `Training` only.

## Claim statuses

`validate_claims.py` now accepts `NOT_MET`, `VERIFIED_SIMULATED`, and statuses that start with `NOT_MEASURED`, and still requires an artifact file for those rows. The last run reported 120 claims and exit 0.
