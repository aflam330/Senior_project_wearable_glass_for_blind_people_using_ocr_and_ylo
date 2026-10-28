# Figure index

**Files:**
- `FIGURE_INDEX.csv` lists every figure file in this folder with its thesis caption, where the thesis uses it.
- This page gives the figures most useful for a paper, and the warnings.

## Recommended for a paper

| Topic | File | Notes |
|---|---|---|
| Prototype | `thesis/sara_v2_collage.jpg` | Second prototype: front, side, back (team member) |
| Hardware parts | `thesis/hw_*.jpg`, `thesis/v2_*.jpg` | Pi 5, camera, buttons, motor, earphone, power bank, frame, enclosure |
| Compositing | `thesis/fig_compositing.jpg` | Source note → COCO background → warped note → composite with automatic box |
| Detector training | `thesis/fig_training_curves.png`, `thesis/fig_confusion_matrix.png`, `thesis/fig_qualitative.jpg` | Synthetic test set |
| Hand-held close-ups | `thesis_extra/fig58_handheld_closeups.png`, `thesis_extra/fig57_handheld_epochs.png` | Synthetic training close-ups; epoch selection on validation |
| Live tests | `thesis/live_*.jpg` | Informal; no error rate counted |
| JaalTaka views | `thesis/jaal_real_v1.jpg` … `v6.jpg`, `thesis/jaal_fake_v1.jpg` | Six ordered views of one note |
| View-count collapse | `publication/fig13_same_arch.png` | Same network, four trainings, 3 seeds (main figure for prefix training) |
| Accuracy vs views | `thesis/fig_auth_views.png`, `publication/fig4_accuracy_vs_views.png` | All variants |
| Serial shortcut | `publication/fig24_serial_leakage.png` | Counterfeits share serials; genuine notes do not |
| Unseen prints | `publication/fig16_unseen_prints.png`, `publication/fig29_fullres_vs_hybrid.png` | Print-disjoint split |
| DMWR | `thesis/fig_dmwr_steps.png`, `publication/fig31_watermark_dmwr.png` | Quotient, portrait, residual on two test notes |
| Safety bound | `publication/fig23_safety_guarantee.png` | P(pass rate > ε) for the max-of-validation threshold |
| Robustness | `thesis/fig_auth_robustness.png`, `publication/fig8_robustness.png` | 12–13 corruptions |
| UV fibres | `thesis/fig_uv_fibres.jpg`, `publication/fig48_uv_simulated_sheets.png`, `publication/fig49_uv_colour_condition.png` | Fibre-free side simulated |
| OCR on real words | `thesis_extra/fig53_recogniser_blend.png`, `thesis_extra/fig54_read_width.png` | Blend of released and fine-tuned weights |
| Voice | `thesis_extra/fig56_voice_commands_noise.png`, `thesis_extra/fig55_bangla_commands.png` | Synthetic voices |
| Raspberry Pi 5 | `thesis/pi5_ocr_response.png`, `thesis_extra/fig52_pi_thermal_timeline.png`, `thesis/pi5_thermal.png` | 13 October run, hot board |

## Warnings

- **`thesis/fig_architecture.png`** (YOLOv8s block diagram) prints "7.3 ms/image inference (RTX 4060)". That figure has no result file; `CORRECTIONS.md` row 3 replaces it with 13.9 ms on an RTX 3050. Redraw it or crop the caption line before use.
- **`publication/fig45_ocr_real_words.png`** shows real-word OCR *before* the confidence filter and the blended recogniser (Bangla 42%, Latin 65%). The current figures are 28.7% and 50.6% (`fig53`, Table "ocr_real").
- **Superseded figure versions:** `Unused/superseded_figures/` in the research repository holds older figures and is not included. Figures 1–12 in `publication/` were redrawn from post-fix result files.
- **Credits for literature device photos:** OrCam MyEye by Ylip (CC BY-SA 4.0), Google Glass by Ted Eytan (CC BY-SA 2.0), Optacon by DrBliss (public domain), eSight 4 by eSight Digital (CC BY-SA 4.0), all from Wikimedia Commons. The AIris photo is reproduced from its paper. Cite these if used.
- **Illustrations:** `illustrations_not_evidence/` holds the slide infographics. In the cartoon, the "500 Taka" bubble shows a note with 1000/5000-style numbers, which is not a real Taka design. Mark them "Illustration" if they are used at all.
- **People in photos:** they are team members; keep their consent in mind for publication.
