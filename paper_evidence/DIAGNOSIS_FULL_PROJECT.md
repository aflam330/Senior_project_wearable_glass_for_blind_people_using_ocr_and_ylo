# Project diagnosis, 2026-09-29

## Syntax

`scripts/eval/diagnose_project.py` compiled 13017 Python files, including the virtual environment. Fifteen failed. All fifteen are `type` statements inside `venv` copies of NumPy and SciPy, which Python 3.10 cannot parse. They are not project modules. Project scripts added in this pass (`eval_bbox_multinote.py`, `export_tflite.py`, `analyze_user_study.py`, `diagnose_project.py`) were part of that compile once written; the compiler finished before some of them existed, so they were compiled separately by running.

## Checkpoints and tests

New test files that load and match the write-ups:

- `results/novel_v2/vcie_k1/seed42/test/test_metrics.json`
- `results/novel_v2/mtpt_prefix_ft/seed42/test/test_metrics.json`
- `results/theory/pacbayes_d1_bound_seed42.json`
- `results/bbox/bbox_eval.json`, `bbox_app_eval.json`, `bbox_multinote.json`
- `results/robustness/occlusion_decision.json`

The detector app still detects on the frame first and brightens only when the frame is dark and no box is found (`savior_glass/modes/currency_mode.py`).

## Claim paths

The compiler reported 11 missing claim artifacts. Those strings are relative to the repository root. `occlusion_decision.json` is one of them and the file is present; the checker ran with the detection folder as the working directory. They are not marked pending on that account.

## What was not a full re-exec

Every historical checkpoint was not reloaded in this pass. Environment variables such as `NOVEL_SET_BLOCKS` are not stored in every old checkpoint. A line-by-line proof that no historical script ever read the test split was not repeated. New runs in this record state that thresholds and epochs were chosen on validation.

## Mismatch to watch

`FINAL_RESULTS.md` rounds the ensemble occlusion accuracy to 88.5% and the rejection errors to 0.5%. The source file `occlusion_decision.json` has ensemble test occlusion accuracy 0.8894230769230769 and wrong-verdict share 0.004807692307692308. The rounded sentences and the JSON are the same experiment. The earlier median-fill result 0.875 is a different checkpoint.

## Recheck 2026-09-29

`scripts/eval/final_scientific_pass.py` compiled 168 project Python files and reported 0 errors. Virtualenv files were excluded. Split leakage in `split_metadata.json` is still 0 / 0 / 0. The oracle file is exactly 205 correct, 201 learned, gap 4, unsolvable 3. `scripts/validate_claims.py` exited 0 on 122 claims. Statuses `NOT_MET`, `VERIFIED_SIMULATED`, and `NOT_MEASURED (PROTOCOL_READY)` are recorded, and their artifact files are still required when a number or a simulated check is attached.

Five case-insensitive hits on `eur` or `usd` were the substring inside `heuristic` and two literature sentences in `scripts/build_report.py`. No foreign-currency image folder was added.
