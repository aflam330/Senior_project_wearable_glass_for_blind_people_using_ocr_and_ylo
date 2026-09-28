# Pilot user-study protocol (no data collected)

**Status: NOT MEASURED.** Do not invent participant numbers, p-values, or SUS scores.

## Ethics
Obtain IRB/ethics approval or documented exemption before recruitment.

## Target
8–12 visually impaired participants. Paired conditions: baseline vs RoboEye.

## Inclusion
Adults who read currency by touch or assistance; able to consent.

## Exclusion
Unable to complete familiarization; acute illness.

## Tasks
1. Denomination identification
2. Genuine vs counterfeit decision
3. Multi-view authentication
4. Feedback interaction (adaptive TTS on/off)

## Measures
Task time, accuracy, errors, NASA-TLX, SUS, satisfaction, assistance requests.

## Analysis
Participant is the independent unit. Paired tests. Save anonymized rows only.

Fill one row per trial in `participant_data_template.csv` (or a copy), with `condition`
set to `baseline` or the glass condition, then run:

```
python analysis_script.py --data participant_data_template.csv
```

For each task and measure it averages each participant's trials per condition, runs a
paired Wilcoxon signed-rank test (exact for up to 25 participants) with the matched-pairs
rank-biserial r as effect size, and applies a Holm correction across all tests. SUS,
NASA-TLX and satisfaction are tested the same way. Output: `statistical_report.json`.
With 8–12 participants only large effects can reach significance, so report the effect
sizes and medians as well as p-values.

Template: `participant_data_template.csv`
