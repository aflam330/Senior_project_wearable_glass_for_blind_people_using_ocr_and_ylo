# Final scan, 2026-09-29 (evening)

Sources: `realtime_bangla_taka_detection/results/scan/scan.json` (`scripts/eval/scan_project.py`), `results/scan/test_metrics_recheck.json` (`scripts/eval/recheck_test_metrics.py`) and `scripts/validate_claims.py`. venv/, .git/, runs/ and cache/ are skipped.

| Check | Result |
|---|---|
| Python files parsed and compiled | **243**, errors **0** |
| Model files loaded (.pt, .pts, .onnx, .torchscript, .pth) | **158**, failures **0** |
| `test_metrics.json` files | **1,025** |
| … with saved per-note predictions | 970 |
| … whose accuracy reproduces from those predictions at 0.5 | **970 / 970** (0 mismatches) |
| Claims in `CLAIM_REGISTRY.json` | **211**; missing artifact **0** |
| … value-checked against the number in their artifact | **182 / 211**, all equal (the other 29: 25 non-numeric, 4 ambiguous) |
| Datasets on disk | 16 folders; table in `SCAN_COMPLETE.md` and `docs/PROJECT_INVENTORY.md` |

The 55 `test_metrics.json` files without per-note predictions were not recomputed. These are detector and module evaluations, and the `eval_novel.py` runs of the extra seeds, which store aggregates only.

`SCAN_COMPLETE.md` was written by an earlier pass. Its counts (225 files, 144 models, 967 metrics, 136 claims) predate this pass. The scan script's docstring says it writes that page, but it writes only `scan.json`.

## Problems found and fixed in this pass

| # | Problem | Fix | Where recorded |
|---|---|---|---|
| 1 | Same-architecture runs interrupted at 5 of 9; the restart check treated a half-trained `checkpoint.pt` as finished | Restart check uses `train_summary.json`; the 4 runs were completed | `SAME_ARCH_RESULTS.md` |
| 2 | Backbone probes interrupted during feature extraction | Re-run, 5 backbones | `SOTA_COMPARISON.md` |
| 3 | PRMVT 6-view read from the pre-fix folder in `SOTA_BEAT.md` and two generators | Generators read `test_views_20260928/`; tables regenerated | `CORRECTIONS.md` rows 10–11 |
| 4 | Claim `C_CAL_PRMVT_HER` held a pre-fix value | Corrected, value-checked | row 12 |
| 5 | Claim validator checked file existence only | Value check added; 148 claims value-checked | row 14, `CLAIM_VALUE_AUDIT.md` |
| 6 | Prototype authenticator could include test notes | Rebuilt from TRAIN notes only | `docs/PROJECT_INVENTORY.md` §7 |
| 7 | `test_speech.py`, `eval_assistive_stack.py` crashed printing Bangla on Windows | stdout set to UTF-8 | same |
| 8 | Two theorem statements overclaimed | Corrected | rows 15–16 |
| 9 | "Wild 96.9 %" still presented as transfer in the thesis | Marked as a source-domain check; 91.5 % cited | row 17 |
| 10 | Deployed jaal verdict off | Safe policy E on for 500 / 1,000 Taka | `JAAL_VERDICT_FIXED.md` |
| 11 | Confidence rejection gives many wrong verdicts under bad light | Image-quality gate evaluated | `SAFETY_FRAMEWORK.md` |
| 12 | `eval_safety_rejection.py` existed but had never been run | Run | `results/safety/rejection_seed42.json` |

## Rescan 2026-09-30

Python files compiled: **262**, errors **0**. Model files: **175**, load failures **0**. Stored accuracies reproducing from saved predictions: **970 / 970**. Claims: **234**, value-checked **205**, all equal; missing artifacts **0**.

**Serial handling.** Serial numbers are read only by the research scripts (`jaaltaka_serial_audit.py`, `serial_whole_note.py`, `watermark_hybrid.py`). The watermark crop box starts below the serial, so no digits enter the watermark features (checked visually).

**Watermark code.** Eight Python files mention the watermark; all are this round's research scripts. The glass app does not yet use the watermark: it needs a back-lit capture.
