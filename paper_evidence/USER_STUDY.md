# User study with visually impaired participants

**Status: READY_FOR_DEVICE (protocol ready, no data collected).** No participant has been recruited. `user_study/analysis_script.py` writes NOT MEASURED until `participant_data.csv` exists. No outcome in any project document comes from participants.

## Files

| File | Content |
|---|---|
| `USER_STUDY.md` | Design, sample size, ethics application content, instruments, 30-day plan (this file) |
| `user_study/consent_form.md` | Participant information and consent template (to be translated into Bangla and read aloud) |
| `user_study/study_protocol.md` | Earlier pilot protocol (8–12 participants) |
| `user_study/participant_data_template.csv` | Per-task data layout |
| `user_study/score_questionnaires.py` | SUS and raw NASA-TLX scoring from item answers (self-check passes) |
| `user_study/analysis_script.py` | Paired Wilcoxon signed-rank tests, rank-biserial effect size, Holm correction |

## Design

- **Within-subject, two conditions:** the participant's usual method, and Savior Glass.
- **Counterbalancing:** condition order is randomised (AB/BA) with block size 4 and a sealed allocation list generated before recruitment.
- **Tasks,** repeated in both conditions with different but matched items:
  1. **Denomination:** 12 notes, all 9 denominations, shuffled. Measures: correct, wrong, "not sure", time per note.
  2. **Text:** 6 printed Bangla and 6 English signs from the OCR phrase list. Measures: words correct (WER, scored by the researcher), time.
  3. **Objects:** find 3 named objects among 8 on a table. Measures: success and time.
- **Excluded tasks:** the glass's genuine/counterfeit verdict is switched off (`JAAL_VERDICT_FIXED.md`) and is not tested with people. Asking participants to judge counterfeit notes with an unvalidated verdict would be unsafe.

## Outcomes

- **Primary** (pre-registered, tested at α = 0.05 without adjustment): the per-participant proportion of the 12 notes named correctly.
- **Secondary** (Holm-adjusted as a family):
  - wrong-denomination rate
  - task times
  - text WER
  - object success
  - SUS
  - raw NASA-TLX
  - satisfaction on a 1–7 scale
  - assistance requests

## Sample size

This is the justification for 105.

- A paired t-test detects a standardised within-person difference of d_z = 0.3 with 80 % power at α = 0.05 (two-sided) with **90** completers.
- The Wilcoxon signed-rank test has asymptotic relative efficiency 0.955 against the t-test, which gives **95**.
- Allowing 10 % dropout, recruit **106**. Computed with SciPy's noncentral t, reproduced in the table below.

| Effect d_z | Completers (Wilcoxon) | Recruit (+10 %) |
|---:|---:|---:|
| 0.3 | 95 | 106 |
| 0.4 | 55 | 62 |
| 0.5 | 36 | 40 |
| 0.8 | 16 | 18 |

**Choice of d_z = 0.3.** The detector is already 91.5 % correct on independent full-note photos (`CROSS_DATASET_ALL.md`). A person's usual method may also be accurate, so a small-to-medium difference is the realistic target. If only 40 participants can be recruited, the study can detect d_z = 0.5 and should be reported as such.

## Eligibility

**Include:**
- adults aged 18 or older
- blind or low vision, meaning they cannot read print at normal size with correction
- use or handle Taka notes in daily life
- able to consent in Bangla

**Exclude:**
- hearing loss that prevents understanding the speech output at maximum volume
- a cognitive condition that prevents informed consent
- people who took part in the development pilot

**Recruitment:** through [organisations of blind people in Bangladesh; to be named in the ethics application].

## Ethics application: content required

The template is below. The committee's own form takes precedence.

1. **Aims and background:** see `docs/reports/Project_Report_2026-09-29.md`, Chapter 1.
2. **Design, tasks, outcomes, sample size:** this file.
3. **Participant information and consent:**
   - `user_study/consent_form.md`
   - accessible process: read aloud, audio consent, sighted independent witness
4. **Risks and mitigations:**
   - fatigue: breaks, sessions of at most 90 minutes
   - device heat: surface temperature checked before each session, stop if above 40 °C
   - misplaced trust: participants are told that the device does not verify genuineness; all notes belong to the team
5. **Data protection:**
   - codes kept separate from identities
   - encrypted storage
   - no face video
   - retention for [5] years
   - withdrawal until the analysis date
6. **Compensation:** [amount] Taka per session plus travel.
7. **Adverse events:** reported to the committee within [5] working days.
8. **Conflicts of interest:** [none / declare].

## Session script (per participant)

1. Consent (about 15 min).
2. Training with the glass: 10 minutes, using items that are not in the tasks.
3. Tasks in the first allocated condition, then the second.
4. SUS and NASA-TLX after each condition, read aloud, with answers given verbally.
5. Short interview on three questions:
   - what was hard
   - when did you not trust the glass
   - would you use it
6. Debrief and payment.

## Instruments

**SUS** (Brooke 1996). Ten statements, answered 1–5 from strongly disagree to strongly agree:

1. I think that I would like to use this system frequently.
2. I found the system unnecessarily complex.
3. I thought the system was easy to use.
4. I think that I would need the support of a technical person to be able to use this system.
5. I found the various functions in this system were well integrated.
6. I thought there was too much inconsistency in this system.
7. I would imagine that most people would learn to use this system very quickly.
8. I found the system very cumbersome to use.
9. I felt very confident using the system.
10. I needed to learn a lot of things before I could get going with this system.

Scored by `score_questionnaires.py`. A validated Bangla translation must be used or produced (forward–back translation).

**Raw NASA-TLX** (Hart 2006). Six scales from 0 to 100: mental demand, physical demand, temporal demand, performance (0 = perfect), effort, frustration. The score is their unweighted mean.

## 30-day longitudinal arm (optional, participants who consent)

- **Sample:** the first 20 participants who agree.
- **Equipment:** each takes home a glass. The existing log (`savior_glass/logs/smart_glass.log`) already records mode changes and detections; it records no images or audio. Logging of "not sure" responses and battery level **must be added before this arm starts**. It is not implemented yet.
- **Weekly phone call:**
  - number of times used for notes, text and objects
  - any wrong note named
  - short SUS
  - problems encountered
- **End-point:** repeat the denomination task in the lab on day 30.
- **Outcomes:**
  - day-1 versus day-30 accuracy and time (Wilcoxon)
  - usage per week (descriptive)
  - abandonment, defined as not used for 7 days
- **Safety:** participants can phone the team at any time. Any wrong-denomination report is logged as an adverse event.

## Analysis

1. Score questionnaires: `python user_study/score_questionnaires.py items.csv scored.csv`.
2. Merge the scores into `participant_data.csv`.
3. Run `python user_study/analysis_script.py`.

`analysis_script.py` currently applies Holm to all measures. When reporting, keep the primary outcome's unadjusted p-value separate, as pre-registered above.
