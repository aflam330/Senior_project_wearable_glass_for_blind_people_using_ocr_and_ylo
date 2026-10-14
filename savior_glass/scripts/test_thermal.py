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
print("thermal governor: all checks pass")
