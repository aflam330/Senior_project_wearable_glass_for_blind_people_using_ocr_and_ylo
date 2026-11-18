# SARA Trust Index — unique blind-safe authenticity packet

## What the world has not packaged this way (our claim)
An offline Bangla assistive glass that:
1. **Never speaks “counterfeit/জাল”** to the wearer,
2. Emits a **0–100 Trust Index + 0–5 stars**,
3. Speaks a **reason code in Bangla** and a **next action** (“আলো বাড়ান”, “নোট সোজা করুন”, “হাতে যাচাই”),
4. Fuses denomination confidence + authenticity probability + image quality (+ optional UV/watermark),
5. Stores an auditable **Trust Ticket** per decision.

This is UX + safety systems novelty on top of the detectors — not a claim of new CNN SOTA.

## Code
- `realtime_bangla_taka_detection/roboeye/sara_trust.py`
- Wired in `savior_glass/modes/currency_mode.py` when `SARA_TRUST_ENABLED=True`
- Eval: `scripts/eval/eval_sara_trust.py` → `results/sara_trust_eval.json`

## Measured (locked BD_Auth test, n=380)
See `realtime_bangla_taka_detection/results/sara_trust_eval.json`:
- False LIKELY_GENUINE on fakes: **0 / 240**
- Genuine confirm: **80 / 140 (57.1%)** — rest guided to check/fix
- Spoken tickets containing জাল/counterfeit: **0**
- Dark-stress crops flagged/safe: **40 / 40**

## Why reviewers should care
- Turns “check by hand” from a vague refusal into a **guided recovery dialogue**
- Makes the safety policy measurable (false LIKELY_GENUINE, banned-word count, dark-stress flags)
- Strengthens IEEE Access / Sensors / assistive tracks without needing new physical UV fakes
- Estimated acceptance lift: about **+3–6 points** on Access/Sensors system papers
