# Pre-registration: guided back-lit capture on the smart glass (draft, 2026-10-01)

Register this, for example on OSF, before the first participant. Fill in the bracketed fields. No data has been collected.

## Question

Does capture guidance make the glass's counterfeit check faster or less often inconclusive for blind and low-vision users? Guidance means asking for the note against the light, rejecting dark or blurry frames, and only then checking.

## Participants

- **Who:** blind or low-vision adults who handle cash, recruited through [organisation].
- **Consent:** in Bangla, read aloud (`user_study/consent_form.md`).
- **Sample size:** per the exact noncentral-t calculation (`results/user_study/power.json`, `scripts/eval/user_study_power.py`), for a paired design at α = 0.05 and 80 % power:

  | Participants | Smallest detectable effect (dz) |
  |---:|---:|
  | 20 | 0.66 |
  | 50 | 0.40 |
  | 105 | 0.28 |

  About 90 participants detect dz = 0.3.
- **Planned n:** [n]. A smaller n is allowed if it is fixed here in advance; the smallest detectable effect is then reported with the result.

## Design

Within-subject, two conditions, order counterbalanced (AB / BA, alternating by participant id).

| Condition | `STUDY_CONDITION` | What the glass does |
|---|---|---|
| guided | `guided` | "Hold the note up to the light", then rejects dark frames (mean < 73.87) and blurry frames (Laplacian variance < 106.75) with a spoken prompt; checks the first frame that passes; 10 s limit |
| unguided | `unguided` | Same first prompt; checks the first frame that contains a note |

**Threshold provenance.** Both are the 2nd percentile of JaalTaka validation back-lit photos (`results/capture_guide/thresholds.json`). On test photos they accept:

| Test photos | Accepted |
|---|---:|
| Clean | 213 / 222 |
| Darkened | 0 / 222 |
| Defocused | 0 / 222 |
| Motion-blurred | 11 / 222 |

Recalibrate on the glass camera with [n] pilot photos before registration, then freeze the numbers here.

**Notes per condition:** [k] notes, 500 and 1,000 BDT, genuine and counterfeit mixed in a fixed random order recorded in the trial sheet.

**Safety.** The glass never says "counterfeit". Both conditions end in "watermark clear" or "check by hand". Participants keep the notes and can check them with the experimenter's help.

## Measures (logged automatically)

`savior_glass/logs/study_events.jsonl`, one line per check. Set `STUDY_PARTICIPANT` and `STUDY_CONDITION` before each block. The experimenter keeps `user_study/trial_sheet_template.csv`: note id, print id, true denomination and label, and the spoken denomination.

- **Primary 1:** median seconds per check, per participant and condition.
- **Primary 2:** hand-check rate on genuine notes (answer is "check by hand" or a timeout).
- **Safety (reported, not tested):** counterfeit notes answered "watermark clear".
- **Secondary:**
  - denomination accuracy;
  - frames rejected per check;
  - timeouts;
  - SUS and NASA-TLX after each condition, scored by `scripts/eval/score_sus_tlx.py`.

## Analysis (fixed now)

**Script.** `python paper_evidence/user_study/analyze_study_events.py study_events.jsonl trial_sheet.csv out.json`
- It matches checks to trials in time order.
- It stops on any count mismatch rather than guessing.

**Primary tests.**
- Two-sided Wilcoxon signed-rank tests, guided versus unguided, on the two primary outcomes.
- Holm correction over the two outcomes.
- Median paired difference and dz are reported.

**Secondary outcomes.** Descriptive, with 95 % bootstrap intervals, and no claims from them.

**Exclusions.**
- A participant who does not finish both conditions is excluded from the paired tests and reported.
- A check logged during a hardware fault that the experimenter noted at the time is dropped and reported.

## Device measurements for the same paper

Run on the Raspberry Pi 5:
```bash
bash scripts/deploy_pi5.sh
python scripts/benchmark_pi5.py --iters 100 --sustained 30
```
Publish, from the resulting JSON and CSV:
- median and 95th percentile latency;
- CPU, RAM and temperature;
- the 30-minute run;
- the note detector, both watermark paths (`watermark_int8_sift` and `watermark_int8_localizer`), and the full currency mode.
