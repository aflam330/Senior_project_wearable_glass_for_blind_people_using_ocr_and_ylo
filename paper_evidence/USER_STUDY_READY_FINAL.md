# User study: READY_FOR_DEVICE, final (2026-09-30)

No participant data exists yet. This page lists what is ready; the design and IRB draft are in `USER_STUDY_READY.md`.

| Item | File | Status |
|---|---|---|
| IRB / ethics application draft | `USER_STUDY_READY.md` (section "IRB / ethics application draft") | Ready; fill in names and institution |
| Consent form (Bangla read-aloud, accessible copy) | `paper_evidence/user_study/consent_form.md` | Ready |
| Protocol, 105 participants, within-subject A/B, counterbalanced | `USER_STUDY_READY.md`, `USER_STUDY_PROTOCOL.md` | Ready |
| 30-day longitudinal plan (subset of 20) | `USER_STUDY_READY.md` | Ready |
| SUS and NASA-TLX scorer, with self-checks | `realtime_bangla_taka_detection/scripts/eval/score_sus_tlx.py` | Self-checks pass; reports Wilcoxon, paired t and dz |
| Task analysis (accuracy, time) | `realtime_bangla_taka_detection/scripts/eval/analyze_user_study.py` | Ready; prints DATA_NOT_COLLECTED |
| Power analysis (exact noncentral t) | `realtime_bangla_taka_detection/scripts/eval/user_study_power.py` → `results/user_study/power.json` | Done |

## Power (paired, two-sided, α = 0.05)

| Participants | Smallest effect for 80 % / 90 % power (dz) | Power at dz = 0.3 |
|---:|---|---:|
| 20 | 0.66 / 0.76 | 25 % |
| 50 | 0.40 / 0.47 | 55 % |
| **105** | **0.28 / 0.32** | **86 %** |

- **Needed for dz = 0.3 at 80 %:** 90 participants.
- **Wilcoxon:** needs about 5 % more under normality.
- **30-day subset of 20:** detects only large changes (dz ≥ 0.66). Report it as descriptive.

## Primary outcomes and analysis

1. Denomination accuracy.
2. Time per note.
3. SUS.
4. NASA-TLX raw score.

Each: glass vs usual method, paired t with a Wilcoxon check, Holm correction over the four, effect size dz. Report how often "check by hand" was followed.

## What the study must not do

Participants must not be asked to decide real counterfeit notes on the glass's word. The glass never says "counterfeit".
