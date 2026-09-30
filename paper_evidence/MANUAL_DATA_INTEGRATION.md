# Where manual measurements go (2026-09-30)

Nothing in this file is a result. The scripts already refuse to invent one: `analyze_user_study.py` prints DATA_NOT_COLLECTED, and `benchmark_pi5.py` sets `raspberry_pi_5=false` when it is not running on a Pi.

## When a user-study table exists

Run `realtime_bangla_taka_detection/scripts/eval/analyze_user_study.py` and `score_sus_tlx.py` on the CSV the protocol names. Keep the paired t-test, Wilcoxon statistic, dz, and the 95 % interval that those scripts print. Do not replace them with a hand-typed p-value. The power calculation already on disk says 105 participants detect dz ≥ 0.28 at 80 % (`results/user_study/power.json`).

## When a Pi 5 log exists

Copy the script's JSON into `paper_evidence/PI5_RESULTS.md` only if `raspberry_pi_5` is true. Record median, 95th percentile, the 30-minute run, and temperature. A laptop timing is not a Pi timing.

## When glass-camera photos exist

Group files by physical note before any split. Back-lit watermark photos are a new test only if those notes were not in the JaalTaka training prints. Score the frozen INT8 watermark (`models/watermark_mobilenetv2_int8.onnx`) once. Do not refit the threshold on those photos and then quote the same photos.
