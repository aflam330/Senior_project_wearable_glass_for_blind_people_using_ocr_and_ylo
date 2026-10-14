"""Power, temperature and clock of the Raspberry Pi 5 while the glass is used, and an estimate of battery life.

The Pi 5's power chip reports the voltage and current of each of its supply rails (vcgencmd pmic_read_adc). Their
sum of V x I is the power the board's own circuits draw. It does not include the 5 V input stage's losses, the
USB devices (camera on the ribbon cable is included, a USB microphone or sound card is not) or the fan, so it is a
LOWER bound of what the power bank delivers. The estimate below adds a stated allowance for that.

Run it in a second terminal while the glass runs, for as long as the session lasts:

    python scripts/power_log.py --minutes 30 --bank-wh 37 --label office_test
    (37 Wh is a 10,000 mAh power bank at 3.7 V)

Writes results/power_<label>_<host>.csv (one row every 2 s) and results/power_<label>_<host>.json (summary).
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RAIL = re.compile(r"^\s*(\S+)\s+(current|volt)\(\d+\)=([\d.]+)([AV])", re.M)


def pmic_watts() -> float | None:
    try:
        out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    cur, volt = {}, {}
    for name, kind, val, _unit in RAIL.findall(out):
        rail = name.rsplit("_", 1)[0]                 # e.g. VDD_CORE_A / VDD_CORE_V -> VDD_CORE
        (cur if kind == "current" else volt)[rail] = float(val)
    rails = set(cur) & set(volt)
    return sum(cur[r] * volt[r] for r in rails) if rails else None


def temp_c() -> float | None:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except (OSError, ValueError):
        return None


def arm_mhz() -> float | None:
    try:
        out = subprocess.run(["vcgencmd", "measure_clock", "arm"], capture_output=True, text=True, timeout=3).stdout
        return int(out.strip().split("=")[-1]) / 1e6
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def throttled() -> str:
    try:
        return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=3).stdout.strip().split("=")[-1]
    except (OSError, subprocess.SubprocessError):
        return ""


def cpu_times():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    return sum(v), v[3] + v[4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--bank-wh", type=float, default=37.0, help="power bank energy in watt-hours (mAh x 3.7 / 1000)")
    ap.add_argument("--overhead", type=float, default=0.30, help="allowance for what the rail sum misses (input stage, fan, USB), as a fraction")
    ap.add_argument("--bank-efficiency", type=float, default=0.85, help="share of the bank's rated energy delivered at 5 V")
    ap.add_argument("--label", default="session")
    a = ap.parse_args()
    host = platform.node() or "pi"
    out_csv = BASE / "results" / f"power_{a.label}_{host}.csv"
    rows = []
    t0 = time.time()
    last = cpu_times()
    print("logging; Ctrl+C stops early and still writes the summary", flush=True)
    try:
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "rail_watts", "temp_c", "arm_mhz", "cpu_busy_pct", "throttled"])
            while time.time() - t0 < a.minutes * 60:
                time.sleep(2)
                now = cpu_times()
                busy = 100 * (1 - (now[1] - last[1]) / max(now[0] - last[0], 1))
                last = now
                r = [round(time.time() - t0, 1), pmic_watts(), temp_c(), arm_mhz(), round(busy, 1), throttled()]
                w.writerow(r)
                f.flush()
                rows.append(r)
    except KeyboardInterrupt:
        pass
    watts = [r[1] for r in rows if r[1] is not None]
    temps = [r[2] for r in rows if r[2] is not None]
    summary = {"host": host, "label": a.label, "minutes": round((time.time() - t0) / 60, 1), "samples": len(rows),
               "rail_watts_mean": round(sum(watts) / len(watts), 2) if watts else None,
               "rail_watts_max": round(max(watts), 2) if watts else None,
               "temp_c_mean": round(sum(temps) / len(temps), 1) if temps else None, "temp_c_max": max(temps) if temps else None,
               "throttled_samples": sum(1 for r in rows if r[5] not in ("", "0x0")),
               "assumptions": {"bank_wh": a.bank_wh, "overhead": a.overhead, "bank_efficiency": a.bank_efficiency}}
    if watts:
        w_in = summary["rail_watts_mean"] * (1 + a.overhead)
        summary["estimated_input_watts"] = round(w_in, 2)
        summary["estimated_battery_hours"] = round(a.bank_wh * a.bank_efficiency / w_in, 2)
    (BASE / "results" / f"power_{a.label}_{host}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
