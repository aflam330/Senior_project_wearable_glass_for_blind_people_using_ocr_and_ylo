# Unused

Files retired on 2026-09-28 after the project audit. Nothing here is loaded by any
code. They are kept for reference instead of being deleted.

## `smart-glass/`

An early copy of `savior_glass/` that fell behind it. Its currency mode had only the
HSV colour matcher and an optional MobileNet classifier, and no MobileNet weights
were ever put in its `models/` folder. On 135 raw BanglaTaka photos the HSV matcher
picked the right note 10 times (7%), and it has no profile for 2 or 5 Taka.
`savior_glass/` is the maintained app.

Before the move, its classifier loader and `scripts/train_currency.py` got the same
fixes as `savior_glass` (see below), so the copy here can load a 9-class checkpoint.

## `broken_int8/`

`best_int8.onnx` from `realtime_bangla_taka_detection/models/` and
`savior_glass/models/` (the same file). It was made with the Ultralytics
`format="onnx", int8=True` export, which also quantized the Detect head. The highest
class score it ever produced was 0.007, so it detected no note in 0 of 200 test
images.

Replaced by a new `best_int8.onnx` made with
`realtime_bangla_taka_detection/scripts/quantize_int8.py`. That script keeps the
Detect head and the first conv in FP32 and calibrates on 200 validation images. The
new file gets 200/200 correct on the same test images and is 18 MB instead of 45 MB.

## `replaced_originals/`

Copies of files as they were before they were rewritten:

| File | Problem in this version |
|------|-------------------------|
| `savior_glass/scripts/train_currency.py`, `smart-glass/scripts/train_currency.py` | Hardcoded 7 classes, so it crashed on the 9-folder dataset (`IndexError: Target 7 is out of bounds`). Class indices followed string order (10, 100, 1000, 20, ...) and were not saved in the checkpoint. It overwrote `models/currency_mobilenet.pt` without asking. |
| `smart-glass/modes/currency_mode.py` | Read classifier outputs as `[10, 20, 50, 100, 200, 500, 1000][idx]`, ignoring the class order the model was trained with. |
| `realtime_bangla_taka_detection/results/camva/checkpoints/baseline_cnnvit_seed43.pt` | Baseline from a run that stopped early (saved at epoch 2, validation 0.899; no training summary was written). Retrained in full on 2026-09-28 for the three-seed CAMVA comparison. |
| `realtime_bangla_taka_detection/results/calibration/suite_seed42.json`, `paper_evidence/CALIBRATION_RESULTS.md`, `paper_evidence/figures/fig6_calibration.*` | PRMVT calibration measured while fully confident notes returned NaN and were scored as p = 0.5 (ECE 0.061; 0.015 after the fix). |
| `realtime_bangla_taka_detection/results/robustness/top_seed42.json`, `severity_seed42.json`, `paper_evidence/ROBUSTNESS_RESULTS.md`, `paper_evidence/SEVERITY_CURVES.md` | PRMVT robustness numbers measured with the same NaN fault. |
| `Thesis Report and paper/thesis report/My report/DIagram fig tab/fig_auth_*.png` | Thesis figures drawn from the numbers above; redrawn by `scripts/eval/make_thesis_auth_figures.py`. |

| `realtime_bangla_taka_detection/requirements.txt` | A `pip freeze` from a Python 3.13 machine, saved half as UTF-16 and half as UTF-8, pinned to `torch==2.6.0+cu124`; pip could not install it on this machine. Replaced by a short UTF-8 list; the venv was rebuilt from it. |

## Deleted (2026-09-28), not kept here

- `realtime_bangla_taka_detection/venv/` (1.7 GB): pointed to `C:\Users\aflam\...\Python313`, which does not exist on this machine, so no command in it could run. Rebuilt from the new `requirements.txt`.
- 30 one-line wrappers `scripts/train/train_<algo>.py` and `scripts/eval/eval_<algo>.py`, and `scripts/train/_write_wrappers.py` that generated them. Each only ran `train_novel.py --algo <algo>` or `eval_novel.py --algo <algo>`; nothing referenced them. Use those two scripts directly.
- `__pycache__/` folders (Python bytecode caches, recreated automatically).

The fixes behind these replacements, and before/after tables for every affected
number, are in `paper_evidence/WEAK_RESULTS_FIX.md`.

## Moved here during the 2026-09-29 repository clean-up

| Path | Why |
|---|---|
| `generated_mirrors/realtime_bangla_taka_detection/paper_evidence/` | A second copy of per-algorithm result pages. 10 of its 11 files were byte-identical to `paper_evidence/`, and the 11th (`NOVEL_ALGORITHMS_COMPARISON.md`) was an older version. `scripts/eval/write_algo_results.py` and `write_novel_comparison.py` now write only to `paper_evidence/`. |
| `superseded_scripts/realtime_bangla_taka_detection/scripts/capture_diag.py`, `capture_diag2.py` | One-off headless webcam diagnostics that nothing references. The live-camera check is now `savior_glass/scripts/test_live_camera.py`. |
| `replaced_originals/paper_evidence/LITERATURE_NOVELTY_MATRIX.md` | Older (2026-09-22) version. The newer copy from `research_audit/` (2026-09-26, same text plus an algorithm-label section) replaced it in `paper_evidence/`. |

Other items were moved to better places, not retired: `CURR/files/` → `docs/papers/currency_detection/`,
`ieee_figures_download/` → `docs/figures/ieee/`, `FEATURE_AUDIT_REPORT.md` and the progress report →
`docs/reports/`, `Review and paper links.xlsx` → `docs/literature/`.

## `broken_int8/.../watermark_mobilenet_int8.onnx` (2026-09-30)

INT8 versions of the watermark MobileNetV3-Small changed too many decisions against FP32 on the serial-disjoint test crops: dynamic 44.7 % agreement, static QDQ 53.3 %, Conv-only static 62.9 % (`results/watermark/mobilenet.json`). MobileNetV3's hard-swish and squeeze-excite layers quantise badly. The FP32 ONNX (6.1 MB, 100 % agreement) is the one to deploy.
