# Approach A: rules fixed before any Approach A score was read

Written 2026-09-29 15:46 UTC, before training started.

- Model: PRMVT fine-tuned on synthetic whole-note composites (`scripts/train/train_whole_note_synth.py`). The epoch is chosen by VAL accuracy at 4 views on one fixed composite per VAL note.
- Policy: the same one-sided policy E. tau_A = the highest p(genuine) of any synthetic VAL counterfeit composite at 4 views; "likely genuine" iff p > tau_A.
- Reported once:
  1. Synthetic TEST composites: accuracy, counterfeit passed, genuine confirmed.
  2. Real whole-note crops (`results/jaal_whole/crops`): counterfeit-set originals, Bangla Money and BanglaTaka genuine. Same measures, plus AUC on the counterfeit-set originals.
- The app keeps the current policy E (S4, tau = 0.99959) unless Approach A both (a) passes 0 real counterfeit originals at tau_A and (b) confirms more independent genuine notes (Bangla Money + BanglaTaka) than S4 does. Both conditions were fixed now, before scoring.
