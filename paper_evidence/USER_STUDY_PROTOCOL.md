# User study protocol

No participant was enrolled. Every outcome below is DATA_NOT_COLLECTED.

## What the study would compare

The same person authenticates notes with the current glass flow and with a baseline that speaks the denomination only. Order is counterbalanced. Each person sees the same note list. The note labels are not shown.

## Measures

Task time, authenticity accuracy against the dataset label, NASA-TLX, System Usability Scale, a satisfaction score, and whether an error is corrected on a second attempt.

## Analysis

`realtime_bangla_taka_detection/scripts/eval/analyze_user_study.py` reads a CSV that is not in the repository. It computes a paired difference, a t statistic, Cohen's d, and a 95% interval. Until that CSV exists it prints DATA_NOT_COLLECTED and does not invent a p-value.

## Consent

Participation is voluntary. Images of notes may be stored. Faces are not required. The person can stop. The protocol needs an institutional review decision before anyone is recruited. This repository is not that decision.
