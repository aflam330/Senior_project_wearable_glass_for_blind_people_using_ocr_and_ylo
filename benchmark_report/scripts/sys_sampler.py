"""Log the Pi's temperature, clock, throttle flags, CPU use and memory every 2 s while a benchmark runs.

    python sys_sampler.py <out.csv> <phase.txt>

The benchmark driver writes the name of the step it is on into phase.txt; each row carries that name so the
timeline graph can show which step heated the board. Stop it with SIGTERM.
"""
from __future__ import annotations

import csv
import glob
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

out, phase_file = Path(sys.argv[1]), Path(sys.argv[2])
running = True


def stop(*_):
    global running
    running = False


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


def read_int(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def vcgencmd(*args):
    try:
        return subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=5).stdout.strip().split("=")[-1]
    except (OSError, subprocess.SubprocessError):
        return ""


fan = (glob.glob("/sys/devices/platform/cooling_fan/hwmon/*/fan1_input") or [None])[0]
psutil.cpu_percent(None)
t0 = time.time()
with out.open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["epoch", "t_s", "phase", "cpu_temp_c", "throttled_hex", "arm_freq_mhz", "cpu_busy_pct", "mem_used_mb",
                "mem_available_mb", "load_1m", "fan_rpm"])
    while running:
        time.sleep(2.0)
        temp = read_int("/sys/class/thermal/thermal_zone0/temp")
        freq = read_int("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        vm = psutil.virtual_memory()
        try:
            phase = phase_file.read_text().strip()
        except OSError:
            phase = ""
        w.writerow([round(time.time(), 2), round(time.time() - t0, 1), phase, None if temp is None else temp / 1000.0, vcgencmd("get_throttled"),
                    None if freq is None else round(freq / 1000.0), psutil.cpu_percent(None), round(vm.used / 1e6),
                    round(vm.available / 1e6), round(psutil.getloadavg()[0], 2), read_int(fan) if fan else None])
        fh.flush()
