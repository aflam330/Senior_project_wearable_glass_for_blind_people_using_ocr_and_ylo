# User study: READY_FOR_DEVICE (2026-09-30)

Everything needed to run the study is on disk. Running it needs participants, the glass and an ethics approval, so no participant data exists and no result is claimed.

## Files

| Item | Path |
|---|---|
| Protocol (conditions, measures, analysis) | `paper_evidence/USER_STUDY_PROTOCOL.md` |
| Consent form | `paper_evidence/user_study/consent_form.md` |
| Paired task analysis (accuracy, time) | `realtime_bangla_taka_detection/scripts/eval/analyze_user_study.py` |
| SUS and NASA-TLX scorer, with self-checks | `realtime_bangla_taka_detection/scripts/eval/score_sus_tlx.py` (run with no file: "self-checks passed", "DATA_NOT_COLLECTED") |

## Design for 105 participants

- **Population.** Blind or low-vision adults who handle Taka notes. Record self-reported vision level, age and phone or screen-reader experience.
- **Conditions (within-subject, counterbalanced order):**
  - (A) glass currency mode;
  - (B) the participant's usual method (touch, asking someone, phone app).
- **Tasks.** Name 18 notes (2 of each of the 9 denominations) under indoor light, then 6 under dim light. For 500 and 1,000 Taka, record what the glass said: "likely genuine" or "check by hand".
- **Measures:**
  - Denomination accuracy and time per note.
  - How often the user acted on "check by hand".
  - SUS after each condition, and NASA-TLX after each condition.
  - A 5-point trust question.
- **Counterfeit safety.** The glass never says "counterfeit". Do not ask participants to judge real counterfeit notes. If the researchers present known counterfeits, record the glass's output and do not tell the participant the note is genuine.
- **Analysis:**
  - Paired Wilcoxon (A vs B) for accuracy, time, SUS and RTLX.
  - Holm correction over the four primary outcomes.
  - Report effect sizes (matched-pairs rank-biserial).
- **Sample size.** 105 participants give about 80 % power for a paired effect of dz ≈ 0.28 at α = 0.05 (two-sided t approximation). A Wilcoxon test needs about 5 % more. That is enough for small-to-moderate effects.

## 30-day longitudinal plan (subset, e.g. 20 participants)

| Day | Activity |
|---|---|
| 0 | Training (15 min), baseline tasks, SUS / NASA-TLX |
| 1–30 | Take-home use. The glass logs only mode, button presses and spoken outputs; no images are stored |
| 7, 14 | Phone check-in: problems, trust, how often "check by hand" was followed |
| 30 | Repeat the lab tasks, SUS / NASA-TLX, semi-structured interview |

Compare day 0 with day 30 (paired Wilcoxon). Report device failures and battery life from the logs.

## IRB / ethics application draft (fill in names and the institution)

- **Title.** Evaluation of an offline assistive smart glass for banknote, text and object recognition by blind and low-vision users.
- **Risk.**
  - Minimal. The device only speaks; it never says a note is counterfeit.
  - No images of participants are stored.
  - The logs hold no personal data.
- **Consent.** Read aloud in Bangla, with an accessible copy. Verbal or thumbprint consent is witnessed. Withdrawal is possible at any time without consequence.
- **Compensation.** Travel costs plus a fixed honorarium, not tied to performance.
- **Data.** Pseudonymous IDs; the key is stored separately; data deleted after 5 years.
- **Vulnerable-population safeguards.** A sighted assistant is present; tasks can be stopped at any time; no cash changes hands.
