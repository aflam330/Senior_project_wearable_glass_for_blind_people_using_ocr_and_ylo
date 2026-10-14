# Heat on the Raspberry Pi 5: what the log shows, and the thermal governor (14 October 2026)

Data: `benchmark_report/raw/system_timeline.csv` (Pi run of 13 October 2026, one sample every 2 s, 651 samples).
Code: `savior_glass/thermal.py`, wired into `main.py` and `preview.py`; checks in `scripts/test_thermal.py`.
Power and battery: `savior_glass/scripts/power_log.py` (new; nothing measured yet).

## 1. What the log shows

| Phase of the run | Mean temperature | Peak | CPU busy | Fan |
|---|---:|---:|---:|---:|
| Idle before anything ran | 71.1 °C | 72.7 °C | 13 % | 7,781 rpm |
| Object scans | 83.9 °C | 85.9 °C | 84 % | 10,022 rpm |
| Text reading (eight phases) | 84.8–85.5 °C | 88.7 °C | 95–99 % | ~10,100 rpm |
| Idle between phases | 74.9 °C | 79.3 °C | 9 % | 10,024 rpm |

Two readings:

1. **The board was already at 71 °C while idle, with the fan at 7,800 rpm.** A Pi 5 with its active cooler normally idles well below that. Either the room was hot, the cooler is not seated well (thermal pad), or the case traps air. No software setting changes this, and it should be checked first.
2. **Temperature follows load within seconds.** A first-order fit of the log (dT/dt = (T∞ − T)/τ) gives τ ≈ 7 s and T∞ ≈ 74.8 + 0.105 × CPU busy %. With this cooling, an average load above about 50 % holds the board above 80 °C, where the firmware starts to lower the clock. This is a rough model of one run, not a measurement of the governor.

## 2. What software can do

A text capture uses all four cores for 15–35 s. Making it lighter makes it slower, which is the opposite of what the user needs. The load that software can lower without anyone waiting longer is the work nobody asked for at that moment: the automatic object scan every 1.5 s and the live preview boxes.

The governor (`thermal.py`) reads the temperature once a second and the firmware's throttle flags every 5 s:

| State | Smoothed temperature | Object auto-scan | Live preview boxes |
|---|---|---|---|
| normal | below 72 °C | every 1.5 s | as set |
| warm | 72–77 °C | every 3 s | half as often |
| hot | 77–81 °C, or the firmware's soft limit is active | every 6 s | off |
| critical | 81 °C and above | paused | off |

- A button press or a voice command is never delayed or refused; ACTION in object mode still announces at once.
- 1.5 °C of hysteresis, and smoothing of single-sample spikes, so the state does not flicker.
- Each change of state goes to the field log; at shutdown the peak temperature and the seconds spent in each state are logged.
- Thresholds are settings (`THERMAL_WARM_C`, `THERMAL_HOT_C`, `THERMAL_CRITICAL_C`); `THERMAL_GOVERNOR=0` turns it off.

`scripts/test_thermal.py` checks the state changes, the hysteresis, the spike smoothing, the throttle-flag override and that nothing is paused off the Pi.

## 3. What is not measured

- Whether the governor keeps the board below 80 °C in real use. Run on the Pi: `python main.py` in one terminal and `python scripts/power_log.py --minutes 30 --label governor_on` in another, then the same with `THERMAL_GOVERNOR=0`.
- Battery life. `power_log.py` adds up the power of the Pi 5's own supply rails (`vcgencmd pmic_read_adc`) every 2 s and estimates hours for a given power bank. The rail sum misses the input stage, the fan and USB devices, so the script adds a stated 30 % allowance; the estimate is only as good as that allowance.
- A cooler or case change. If the idle temperature stays near 70 °C, that is the first fix.
