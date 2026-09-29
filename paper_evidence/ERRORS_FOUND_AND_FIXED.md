# Errors found and fixed (2026-09-29)

Scan basis: `SCAN_COMPLETE.md`. It found:
- 225 Python files, 0 syntax errors
- 144 model files, all loading
- 136 claims, none missing its artifact

The errors below were found by running the code and comparing documents with artifacts.

| # | Error | Root cause | Fix | Verified by |
|---|---|---|---|---|
| 1 | Glass spoke a genuine/jaal verdict that was wrong for 20–59 % of genuine whole-note photos | PRMVT trained on JaalTaka close-ups, applied to whole-note crops | Verdict off (`JAAL_VERDICT_ENABLED = False`); the glass says the check was not done | `JAAL_VERDICT_FIXED.md`, `results/jaal_deployed/` |
| 2 | `roboeye_live.py` crashed on the first frame (`cannot pickle '_thread.lock'`) | Permanent forward hook with a bound method, deep-copied by Ultralytics 8.4 `setup_model`; the hook would also not fire on the copy | Lazy hook on the predictor's own network (`roboeye/dual_head.py`) | 150-frame live run, exit 0; SPPF features captured (`ROBOEYE_LIVE_FIXED.md`) |
| 3 | Grad-CAM overlay did nothing | Optional `pytorch_grad_cam` not installed; exceptions swallowed; design hooked the shared live model | Self-contained Grad-CAM on a private copy, hook scoped to one call, throttled in the HUD | Heat map produced; live predictions unchanged (0.903) |
| 4 | Prototype matcher crash (576 vs 512) | CLIP-built prototypes loaded under the MobileNet fallback | `load()` rejects a mismatched backend or dimension | `test_roboeye_modules.py`: all pass |
| 5 | Desktop HUD also spoke and vibrated an unvalidated verdict | Same as #1, different model (dual head) | `AUTH_VERDICT_ENABLED = False`; the VLM template says "not checked" | `ROBOEYE_LIVE_FIXED.md` |
| 6 | No unknown-note output (1-taka announced as 5 taka) | Closed-set detector, announcement at confidence 0.25 | Validation-chosen confidence 0.60 (`CURRENCY_ANNOUNCE_CONF`) | `OPEN_SET_REJECTION.md`: 43.6 % → 22.8 % |
| 7 | "Wild" 96.9 % presented as a real-photo test | 450 photos sampled from the detector's training source | Retired; replaced by Bangla Money 91.5 % in the thesis and documents | `CORRECTIONS.md` |
| 8 | Detector README numbers (0.998 / 0.997, 2,238 images, RTX 4060 135 FPS) had no artifact | Numbers from an earlier run, never saved | Test split re-run and saved; README updated | `results/training_v2/test_eval/test_metrics.json` |
| 9 | `scripts/evaluate.py` saved no metrics | Printed only | Writes `test_metrics.json` | same file |
| 10 | McNemar p = 0.0244 attributed to PRMVT | Copied from the early Q-DUIG row | Correction note: PRMVT 6-view p = 0.0033 | `CORRECTIONS.md` |
| 11 | Pre-fix PRMVT numbers in `PAPER_FINAL.md` and `MULTI_SEED_RESULTS.md` | Written before the NaN-entropy fix | Correction notes with post-fix values | `CORRECTIONS.md` |
| 12 | READMEs and docstrings contradicted the code (YOLOv8n; HSV currency; "Fully Offline") | Documentation not updated when the code changed | Updated | `CORRECTIONS.md` items 7–9 |
| 13 | Paper-table generator read JSON keys that do not exist (empty "wild" rows) | Key names changed | Uses the external-photo result | `scripts/generate_paper_tables.py` |
| 14 | JaalTaka citation had no author, DOI or license | Source not recorded | Mendeley DOI 10.17632/2m7wk5cy4c.2, CC BY 4.0, authors added to `ref.bib` | `OPEN_SOURCE.md` |
| 15 | Three conflicting PyTorch versions documented | Different machines and dates | `requirements-lock.txt`: exact freeze of the environment behind today's results (torch 2.14.0+cu126, ultralytics 8.4.164) | file present |
| 16 | Hardcoded `E:\` paths in the dataset generator | Absolute paths | Paths relative to the repository | Import check: all three folders resolve |
| 17 | Button test misread "জাল যাচাই করা হয়নি" as a counterfeit verdict | Substring match on "জাল" | Explicit "not checked" case | Test passes |

## Known, not fixed

- **`data/data.yaml` keeps an absolute `path:`.** Ultralytics resolves a relative dataset path against its own settings directory, not against the YAML file, so changing it would break training on this machine.
- **Selection pressure.** About 110 runs have been scored on the one JaalTaka test split. This cannot be fixed without new notes (`NEW_TEST_SET.md`).
- **One-note difference between evaluation paths.** PRMVT seed 42 at 2 views scores 0.9760 through the late-fusion script and 0.9808 in `test_views_20260928`. That is one note, from a different batching and model-call path.
