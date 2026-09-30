# JaalTaka counterfeit count (2026-09-30)

The whole-note counterfeit file has 87 images from about four physical notes (`LARGE_SCALE_DATASET.md`). Expanding those images to 500 by rotation, scale, blur, JPEG, occlusion, or lighting does not add prints. A 5-fold split of the augmented copies would test the same notes many times.

The expansion that does exist is note-level, and it is already published:

| Design | Notes tested once | 1 view | 6 views | Source |
|---|---:|---:|---:|---|
| Stratified note folds | 1,390 | 98.3 % | 98.9 % | `results/sota/cv_probe.json` |
| Serial-grouped folds | 1,390 | 95.8 % | 98.9 % | same |

The serial-grouped fold that holds the large shared print scores 89.2 % at one view. That drop is the print effect. It is not cured by copying the 87 whole-note photos.

No new counterfeit images were written in this pass.
