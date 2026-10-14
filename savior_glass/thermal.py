"""Thermal governor: keeps the glass's background work light when the Raspberry Pi 5 is hot.

Evidence (benchmark_report/raw/system_timeline.csv, 13 October 2026, one sample every 2 s): the board idled at
71 °C with the fan near 7,800 rpm, ran at 85 °C during text reading (CPU 95-99 % busy) and was throttled or
clock-capped in 76 % of the samples. A first-order fit of that log, T_inf = 74.8 + 0.105 * busy %, says that with
this cooling an average load above roughly 50 % keeps the board over 80 °C. A text capture cannot be made lighter
without making it slower, so the governor works on the load that repeats without the user asking for it:

  state      temperature (smoothed)        object auto-scan        live preview boxes
  normal     below 72 °C                   every OBJECT_SCAN_INTERVAL   as set
  warm       72 to 77 °C                   2 times slower               2 times slower
  hot        77 to 81 °C, or soft limit    4 times slower               off
  critical   81 °C and above               paused (ACTION still works)  off

A button press or voice command is never delayed or refused. The thresholds sit below the firmware's soft limit
(80 °C on the Pi 5) so that the governor acts before the firmware lowers the clock. Each change of state is logged
(and written to the field log), so a field run shows how long the board spent in each state.

Reading the sensor costs a file read once a second; the throttle flags (vcgencmd) are read every 5 s.
Off the Pi there is no sensor and the state stays "normal". THERMAL_GOVERNOR=0 turns it off.
"""
from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path

import config

logger = logging.getLogger("smart_glass.thermal")

STATES = ("normal", "warm", "hot", "critical")
SCAN_FACTOR = {"normal": 1.0, "warm": 2.0, "hot": 4.0, "critical": None}      # None: paused
PREVIEW_FACTOR = {"normal": 1.0, "warm": 2.0, "hot": None, "critical": None}


def read_temp_c() -> float | None:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
    except (OSError, ValueError):
        return None


def read_throttled() -> int | None:
    """vcgencmd get_throttled as an integer; bit 3 = soft temperature limit active now, bit 2 = throttled now."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2).stdout
        return int(out.strip().split("=")[-1], 16)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def classify(temp: float, throttled: int | None, previous: str) -> str:
    """State for a smoothed temperature, with 1.5 °C of hysteresis so it does not flicker at a boundary."""
    up = (config.THERMAL_WARM_C, config.THERMAL_HOT_C, config.THERMAL_CRITICAL_C)
    h = 1.5
    level = STATES.index(previous)
    new = sum(temp >= u for u in up)
    if new < level:                                   # going down only after falling clearly below the threshold
        new = sum(temp >= u - h for u in up)
        new = min(new, level)
    if throttled is not None and throttled & 0xC:      # the firmware is already capping the clock: at least "hot"
        new = max(new, 2)
    return STATES[new]


class ThermalGovernor:
    def __init__(self, reader=read_temp_c, throttle_reader=read_throttled, on_change=None):
        self._read, self._read_thr = reader, throttle_reader
        self.on_change = on_change
        self.state = "normal"
        self.temp: float | None = None
        self.peak: float | None = None
        self.seconds_in: dict = {s: 0.0 for s in STATES}
        self._thr: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="thermal")
        self.enabled = config.THERMAL_GOVERNOR and reader() is not None

    def start(self) -> None:
        if self.enabled:
            self._thread.start()
            logger.info("Thermal governor on (warm %.0f, hot %.0f, critical %.0f °C)",
                        config.THERMAL_WARM_C, config.THERMAL_HOT_C, config.THERMAL_CRITICAL_C)

    def stop(self) -> None:
        self._stop.set()

    def step(self, temp: float | None, throttled: int | None, dt: float = 1.0) -> str:
        """One update; also used by the tests with made-up readings."""
        if temp is None:
            return self.state
        self.temp = temp if self.temp is None else 0.6 * self.temp + 0.4 * temp     # smooth single-sample spikes
        self.peak = temp if self.peak is None else max(self.peak, temp)
        new = classify(self.temp, throttled, self.state)
        self.seconds_in[self.state] += dt
        if new != self.state:
            old, self.state = self.state, new
            logger.info("Thermal state %s -> %s (%.1f °C)", old, new, self.temp)
            if self.on_change:
                try:
                    self.on_change(old, new, self.temp)
                except Exception:  # noqa: BLE001
                    pass
        return self.state

    def scan_interval(self, base: float) -> float | None:
        f = SCAN_FACTOR[self.state]
        return None if f is None else base * f

    def preview_interval(self, base: float) -> float | None:
        f = PREVIEW_FACTOR[self.state]
        return None if f is None else base * f

    def _run(self) -> None:
        k = 0
        while not self._stop.wait(1.0):
            if k % 5 == 0:
                self._thr = self._read_thr()
            k += 1
            self.step(self._read(), self._thr, 1.0)
