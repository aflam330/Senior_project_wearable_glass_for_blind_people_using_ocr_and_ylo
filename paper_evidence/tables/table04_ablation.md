# Table 4. Ablation study

> **Correction 2026-09-28.** Q-DUIG accuracies in this file were computed before three evaluation bugs were fixed: NaN entropy for fully confident notes (scored as p = 0.5), a volume feature that changed when views were masked, and a missing view self-gate in the policy path. Every checkpoint was re-evaluated with the fixed code; see `WEAK_RESULTS_FIX.md`. Where those numbers differ from the ones below, they supersede them, and verdicts based on the old numbers should be re-read. The original text is kept unchanged below.

Source: `results/qduig/ablations/`

| config | accuracy | macro_f1 |
|---|---|---|
| baseline | NOT_MEASURED | NOT_MEASURED |
| quality | NOT_MEASURED | NOT_MEASURED |
| uncertainty | NOT_MEASURED | NOT_MEASURED |
| diversity | NOT_MEASURED | NOT_MEASURED |
| qd | NOT_MEASURED | NOT_MEASURED |
| ud | NOT_MEASURED | NOT_MEASURED |
| full | NOT_MEASURED | NOT_MEASURED |
| no_cost | NOT_MEASURED | NOT_MEASURED |
| no_calibration | NOT_MEASURED | NOT_MEASURED |
| no_redundancy | NOT_MEASURED | NOT_MEASURED |
| no_infogain | NOT_MEASURED | NOT_MEASURED |

