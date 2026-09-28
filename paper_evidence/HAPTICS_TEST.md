# Haptics test

**Hardware: NOT_MEASURED** (no Raspberry Pi or vibration motor here). The motor-driving code
was tested with the simulated `RPi.GPIO` in `savior_glass/scripts/test_buttons_haptics.py`,
which timestamps every write to the motor pin (BCM 13).

## Bug fixed

Before 2026-09-28 the glass's `CurrencyMode._buzz()` ignored its pattern argument and gave
one 80 ms pulse for every note, so a genuine and a counterfeit note felt the same. The
distinct patterns existed only in `roboeye/haptics.py`, which the glass did not use.
`_buzz()` now plays the verdict's pattern in a background thread (so speech is not delayed),
and `process_frame` picks it from the jaal verdict.

## Result

| Pattern | Expected pulses (on ms) | Measured pulses [start ms, on ms] | Result |
|---|---|---|---|
| genuine | 2 × 80 | [1, 80], [161, 80] | PASS |
| counterfeit | 3 × 220 | [1, 220], [301, 220], [602, 221] | PASS |
| detect (no verdict) | 1 × 50 | [1, 51] | PASS |

Timing with speech: on an ACTION press with a genuine 100-taka photo, the genuine pattern
started 0 ms after the verdict text was queued for speech (both at 989 ms after the press).

Intensity: the glass switches the motor fully on/off, so there is no user intensity control.
`roboeye/haptics.py` (desktop demo) drives PWM at 80 % duty.

Not covered: whether users can tell the patterns apart by feel. That belongs in the user study.
