# FINAL_AUDIT_REPORT

Date: 2026-09-21 (after CAMVA train+eval seed 42)

| Check | Status | Notes |
|-------|--------|-------|
| authenticity.py not overwritten | PASS | checkpoints only under results/camva/ |
| Note-ID disjoint split | PASS | 974/208/208, 6 views locked to note |
| CAMVA vs baseline 1–6 views | PASS | views_1_to_6.json; CAMVA worse 1–5, better at 6 (p=0.026) |
| Temperature on val only | PASS | T=0.865 |
| Ablations A–E | PASS | ablations.json |
| Adaptive 3 orders × 5 thresholds | PASS | adaptive.json |
| Robustness corruptions | PASS | robustness.csv |
| 3 training seeds | FAIL | only seed 42 |
| RTX 4060 | FAIL | ran on RTX 3050 Laptop |
| Raspberry Pi 5 | FAIL | NOT MEASURED |
| User study | FAIL | protocol only |
| Fabricated CAMVA SOTA | PASS | README states CAMVA loses at 1–5 views |
| Predictions exist for 1–6 views | PASS | predictions/*view.json |
