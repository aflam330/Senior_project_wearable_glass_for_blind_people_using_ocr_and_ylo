# Expanded test: every JaalTaka note tested once (2026-09-30)

**Constraint.** All 1,390 JaalTaka notes are already in the 974 / 208 / 208 split, so no unused notes exist (`NEW_TEST_SET.md`).

**Answer.** Five-fold cross-validation over all 1,390 notes, so every note is a test note exactly once for a model that never saw it (`scripts/eval/cv_probe.py` → `results/sota/cv_probe.json`).
- Model: the frozen ResNet-50 probe.
- C = 100, the value chosen on the standard split's validation set.
- k views = mean probability over the first k views.

| Fold design | 1 view | 2 | 3 | 4 | 5 | 6 views | Per-fold 1-view accuracy |
|---|---:|---:|---:|---:|---:|---:|---|
| Note folds (stratified) | 98.3 % | 98.5 % | 98.9 % | 98.9 % | 98.9 % | 98.9 % | 99.3, 98.9, 96.8, 97.5, 98.9 |
| Serial-grouped folds (a counterfeit print stays in one fold) | 95.8 % | 95.7 % | 97.6 % | 98.4 % | 98.4 % | 98.9 % | 89.2, 97.5, 97.5, 97.8, 97.1 |

- **1,390 notes** is 6.7 times the 208-note test set, so the pooled accuracy has a 95 % interval of about ±0.7 points at 98 %.
- **The serial-grouped fold holding the large 3274658 print scores 89.2 % at one view.** When a whole print is absent from training, single-view accuracy drops. Six views recover it.
- **Session-disjoint splits are not possible:** JaalTaka's `camera_id` and `session_id` fields are empty.
