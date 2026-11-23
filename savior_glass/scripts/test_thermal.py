"""Checks of the thermal governor (thermal.py) with made-up sensor readings. No Pi needed.

    python scripts/test_thermal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import thermal  # noqa: E402

config.THERMAL_GOVERNOR = True
config.THERMAL_RESUME_STEP_S = 0          # checks 1-6: resume at once, as before
changes = []
g = thermal.ThermalGovernor(reader=lambda: 60.0, throttle_reader=lambda: 0, on_change=lambda o, n, t: changes.append((o, n)))

# 1. cool board: normal, scans at the configured interval
for _ in range(10):
    g.step(65.0, 0)
assert g.state == "normal" and g.scan_interval(1.5) == 1.5 and g.preview_interval(1.5) == 1.5

# 2. heating up step by step: warm, hot, critical; scans slower, then paused
for t, want in ((74.0, "warm"), (79.0, "hot"), (83.0, "critical")):
    for _ in range(15):
        g.step(t, 0)
    assert g.state == want, (t, g.state)
assert g.scan_interval(1.5) is None and g.preview_interval(1.5) is None
assert [n for _, n in changes] == ["warm", "hot", "critical"], changes

# 3. hysteresis: just under the critical threshold it stays critical; clearly under, it steps down
for _ in range(15):
    g.step(80.5, 0)
assert g.state == "critical", g.state
for _ in range(15):
    g.step(78.0, 0)
assert g.state == "hot", g.state

# 4. a single hot sample (a spike) does not change the state
g2 = thermal.ThermalGovernor(reader=lambda: 60.0, throttle_reader=lambda: 0)
for _ in range(10):
    g2.step(65.0, 0)
g2.step(90.0, 0)
assert g2.state in ("normal", "warm"), g2.state       # smoothed: 0.6 * 65 + 0.4 * 90 = 75 at most

# 5. the firmware's soft limit flag forces at least "hot", whatever the temperature
g3 = thermal.ThermalGovernor(reader=lambda: 60.0, throttle_reader=lambda: 0)
g3.step(70.0, 0x8)
assert g3.state == "hot" and g3.scan_interval(1.5) == 6.0

# 6. off the Pi (no sensor) the governor stays off and never pauses anything
g4 = thermal.ThermalGovernor(reader=lambda: None, throttle_reader=lambda: None)
assert not g4.enabled and g4.scan_interval(1.5) == 1.5
# 7. gradual resume: from critical, a cool board steps down one state at a time, each after the dwell time
config.THERMAL_RESUME_STEP_S = 30
g5 = thermal.ThermalGovernor(reader=lambda: 60.0, throttle_reader=lambda: 0)
for _ in range(15):
    g5.step(84.0, 0)
assert g5.state == "critical"
seen = []
for _ in range(120):                       # 2 minutes at 60 °C, one reading a second
    g5.step(60.0, 0)
    if not seen or seen[-1][1] != g5.state:
        seen.append((round(g5._since_change), g5.state))
states = [s for _, s in seen]
assert states == ["critical", "hot", "warm", "normal"], seen

# never faster than one state per 30 s, and heating is still immediate
g6 = thermal.ThermalGovernor(reader=lambda: 60.0, throttle_reader=lambda: 0)
for _ in range(15):
    g6.step(84.0, 0)
n = 0
while g6.state == "critical":
    g6.step(60.0, 0)
    n += 1
assert g6.seconds_in["critical"] >= 30 and n < 30, (g6.seconds_in, n)   # 30 s in critical in all, counted from entering it
for _ in range(15):
    g6.step(84.0, 0)
assert g6.state == "critical"
print("thermal governor: all checks pass")
