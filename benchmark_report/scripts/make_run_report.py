"""Build REPORT.md, REPORT.html and the graphs of one dated run of run_full_benchmark.sh.

    python make_run_report.py [benchmark_report/<date>]      # default: the newest dated folder

Every number in the report is read from the run's raw files, or from the earlier runs it is compared with
(1 October: savior_glass/results/pi5_benchmark_memo_20261001T111325Z.*; 13 October: benchmark_report/raw/).
Nothing is typed in by hand.
"""
from __future__ import annotations

import base64
import csv
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_report import (BAND, BLUE, BLUE_LIGHT, CRITICAL, GRID, INK, INK2, ORANGE, AQUA, SURFACE, WARNING,  # noqa: E402
                         Line2D, Patch, decode, header, legend_above, load, ms, plt, sec, state, top_margin)
from make_report import NAMES as OLD_NAMES  # noqa: E402

BR = Path(__file__).resolve().parents[1]
GLASS = BR.parent / "savior_glass"
RUN = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else sorted(p for p in BR.glob("20??-??-??") if p.is_dir())[-1]
RAW, GRAPHS = RUN / "raw", RUN / "graphs"
OCT1 = GLASS / "results" / "pi5_benchmark_memo_20261001T111325Z.json"
OCT13 = BR / "raw" / "pi5_benchmark_memo_20261013T173438Z.json"
OCT13_EXTRAS, OCT13_OCR = BR / "raw" / "extras.json", BR / "raw" / "ocr_response.json"
BANK_WH, BANK_EFF, OVERHEAD = 37.0, 0.85, 0.30   # the defaults of savior_glass/scripts/power_log.py
THIS = datetime.strptime(RUN.name, "%Y-%m-%d").strftime("%-d %b")

NAMES = {**OLD_NAMES,
         "taka_yolo_handheld_int8": "Taka detector, hand-held INT8 (Pi default when measured)",
         "taka_yolo_onnx_int8": "Taka detector, earlier INT8 (v1)",
         "ocr_mode": "Text reading, 3-line sign (benchmark frame)"}
GROUPS = [
    ("Currency", ["currency_mode_full", "taka_yolo_handheld_int8", "taka_yolo_onnx_int8", "taka_yolo_onnx_fp32", "taka_yolo_pt"]),
    ("Watermark", ["watermark_int8_localizer", "watermark_int8_sift"]),
    ("Counterfeit check", ["jaal_check_qduig"] + [f"prmvt_{k}view" for k in range(1, 7)]),
    ("Objects", ["object_mode", "object_yolov8s", "object_yolov8n"]),
    ("Emotion", ["emotion_with_face", "emotion"]),
]
OCR_CONFIGS = [("onnx_int8", 1200, "8-bit detector, width 1,200 (the Pi default)"),
               ("onnx", 1200, "full-precision detector, width 1,200"),
               ("onnx_int8", 880, "8-bit detector, width 880"),
               ("onnx", 880, "full-precision detector, width 880")]
CONFIG_COLOURS = [BLUE, ORANGE, AQUA, "#8a6bd1"]
STEP_LABELS = {"idle_before": "Idle", "idle_after": "Idle", "tests": "Tests", "main_benchmark": "Mode benchmark",
               "mode_switch": "Extras", "extras": "Extras", "live_camera": "Extras",
               **{f"ocr_{d}_{w}": f"OCR {'8-bit' if d == 'onnx_int8' else 'FP32'} {w}" for d, w, _ in OCR_CONFIGS}}
FIELD_NAMES = {("ocr", "result"): "Text reading: one capture", ("object", "object_announce"): "Object scan and announcement",
               ("currency", "result"): "Currency check",
               ("currency", "watermark_check"): "Watermark check (guided, several frames)",
               ("currency", "uv_check"): "UV fibre check", ("emotion", "result"): "Emotion"}


# ---------------------------------------------------------------- reading
def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def pct(part: float, whole: float) -> str:
    return f"{(part / whole - 1) * 100:+.0f}%"


def fmt_any(v_ms) -> str:
    return sec(v_ms) if v_ms is not None and v_ms >= 2000 else ms(v_ms)


def watts(v) -> str:
    return "–" if v is None else f"{v:.2f} W"


def camera_name() -> str:
    env = RAW / "environment.txt"
    text = env.read_text(encoding="utf-8", errors="ignore") if env.is_file() else ""
    m = re.search(r"^(.+?) \(usb-", text, re.M)
    return f"{m.group(1).strip()} (USB webcam)" if m else "not recorded"


def save(fig, name: str) -> str:
    GRAPHS.mkdir(exist_ok=True)
    fig.savefig(GRAPHS / name, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return f"graphs/{name}"


# ---------------------------------------------------------------- graphs
def graph_latency(lat: dict, iters: int) -> str:
    ys, labels, med, p95, heads, y = [], [], [], [], [], 0.0
    for group, keys in GROUPS:
        keys = [k for k in keys if k in lat]
        if not keys:
            continue
        heads.append((y, group))
        y += 0.9
        for k in keys:
            ys.append(y)
            labels.append(NAMES[k])
            med.append(lat[k]["median_ms"])
            p95.append(lat[k]["p95_ms"])
            y += 1.0
        y += 0.45
    fig, ax = plt.subplots(figsize=(10.5, 0.37 * y + 1.5))
    fig.subplots_adjust(left=0.37, right=0.97, top=top_margin(fig, 1.2), bottom=0.6 / fig.get_figheight() + 0.02)
    ax.barh(ys, med, height=0.56, color=BLUE)
    for yy, m, p in zip(ys, med, p95):
        ax.plot([m, p], [yy, yy], color=INK2, lw=1, solid_capstyle="butt")
        ax.plot([p, p], [yy - 0.22, yy + 0.22], color=INK, lw=1.6)
        ax.annotate(ms(m), (max(m, p), yy), xytext=(7, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(ys, labels)
    for hy, group in heads:
        ax.text(-0.012, hy, group, transform=ax.get_yaxis_transform(), ha="right", va="center", fontweight="bold",
                color=INK2, fontsize=9.5)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(p95) * 1.16)
    ax.set_xlabel("Time per call (ms) – shorter is faster")
    legend_above(fig, [Patch(color=BLUE, label="Median (the number at the end of the row)"),
                        Line2D([], [], color=INK, marker="|", markersize=10, markeredgewidth=1.6, lw=0, label="95th percentile")], 2)
    header(fig, "Latency of every mode and model on the Raspberry Pi 5",
           f"Median of {iters} calls after 3 warm-up calls, on 640x480 frames. Text reading has its own graphs (it takes seconds).")
    return save(fig, "01_latency_all_modes.png")


def graph_runs(lat: dict, runs: list[tuple[str, dict, str]]) -> str | None:
    keys = [k for _, ks in GROUPS for k in ks
            if k in lat and sum(k in r for _, r, _ in runs) >= 1 and not re.match(r"prmvt_[2345]view", k)]
    if not keys:
        return None
    every = list(runs) + [(f"{THIS} (this run)", lat, ORANGE)]
    n = len(every)
    h = 0.8 / n
    fig, ax = plt.subplots(figsize=(10.5, 0.85 * len(keys) + 1.7))
    fig.subplots_adjust(left=0.37, right=0.97, top=top_margin(fig, 1.25), bottom=0.6 / fig.get_figheight() + 0.02)
    hi = 0
    for i, (lbl, r, colour) in enumerate(every):
        for y, k in enumerate(keys):
            if k not in r:
                continue
            v = r[k]["median_ms"]
            hi = max(hi, v)
            yy = y - 0.4 + h * (i + 0.5)
            ax.barh(yy, v, height=h * 0.92, color=colour)
            ax.annotate(ms(v), (v, yy), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8,
                        color=INK if i == n - 1 else INK2, fontweight="bold" if i == n - 1 else "normal")
    ax.set_yticks(range(len(keys)), [NAMES[k] for k in keys])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, hi * 1.2)
    ax.set_xlabel("Median time per call (ms)")
    legend_above(fig, [Patch(color=c, label=lbl) for lbl, _, c in every], n)
    header(fig, "This run against the 1 and 13 October runs",
           "Same script (scripts/benchmark_pi5.py), same board. A missing bar means that row was not timed in that run.")
    return save(fig, "02_against_earlier_runs.png")


def graph_timeline(tl: list[dict]) -> str | None:
    if not tl:
        return None
    e0 = tl[0]["epoch"]
    t = [(r["epoch"] - e0) / 60 for r in tl]
    panels = [("Core temperature (°C)", [r["cpu_temp_c"] for r in tl], 2.3),
              ("Firmware throttle state", [state(r["throttled_hex"]) for r in tl], 1.0),
              ("CPU clock measured by the firmware (GHz)", [None if r["arm_clock_mhz"] is None else r["arm_clock_mhz"] / 1000 for r in tl], 1.1),
              ("Board power, sum of the power chip's rails (W)", [r["pmic_watts"] for r in tl], 1.4),
              ("CPU in use (%)", [r["cpu_busy_pct"] for r in tl], 1.1),
              ("Memory in use, whole system (GB)", [r["mem_used_mb"] / 1000 for r in tl], 1.0)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10.5, 1.5 * sum(p[2] for p in panels) + 1.5), sharex=True,
                             gridspec_kw={"height_ratios": [p[2] for p in panels], "hspace": 0.45})
    fig.subplots_adjust(left=0.08, right=0.88, top=top_margin(fig, 1.3), bottom=0.06)
    spans, start = [], 0
    for i in range(1, len(tl) + 1):
        if i == len(tl) or tl[i]["group"] != tl[start]["group"]:
            spans.append((t[start], t[i - 1], tl[start]["group"]))
            start = i
    for ax, (title, values, _) in zip(axes, panels):
        for a, b, name in spans:
            if name and name != "Idle":
                ax.axvspan(a, b, color=BAND, lw=0, zorder=0)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK2, pad=4)
        ax.grid(axis="x", visible=False)
        if title.startswith("Firmware"):
            ax.fill_between(t, values, step="post", color=BLUE, lw=0)
            ax.set_ylim(0, 2.25)
            ax.set_yticks([0, 1, 2], ["Normal", "Soft limit", "Throttled"])
        elif title.startswith("CPU clock"):
            ax.plot(t, values, color=BLUE, lw=1.2)
            ax.set_ylim(0, 2.6)
            ax.set_yticks([0, 1.2, 2.4])
        else:
            ax.plot(t, values, color=BLUE, lw=1.8 if title.startswith("Core") else 1.2)
    ax0 = axes[0]
    temps = panels[0][1]
    ax0.set_ylim(min(50, min(temps) - 2), max(90, max(temps) + 3))
    for level, colour, label in ((80, WARNING, "80 °C  soft limit"), (85, CRITICAL, "85 °C  throttling")):
        ax0.axhline(level, color=colour, lw=1.4, ls=(0, (5, 3)))
        ax0.annotate(label, (1.0, level), xycoords=ax0.get_yaxis_transform(), xytext=(6, 0), textcoords="offset points",
                     va="center", fontsize=8.5, color=INK2, annotation_clip=False)
    peak = max(range(len(temps)), key=lambda i: temps[i])
    ax0.scatter(t[peak], temps[peak], s=60, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=5)
    ax0.annotate(f"peak {temps[peak]:.1f} °C", (t[peak], temps[peak]), xytext=(0, 9), textcoords="offset points", ha="center",
                 fontsize=9, fontweight="bold")
    seen = set()
    for a, b, name in spans:
        if name and (b - a) > 1.2 and (name, round(a)) not in seen:
            seen.add((name, round(a)))
            short = b - a < 4   # a short step's label would run into the next one: end it at the step's right edge, one line up
            ax0.text(b if short else (a + b) / 2, 0.04, name + "\n" if short else name.replace("OCR ", "OCR\n"),
                     transform=ax0.get_xaxis_transform(), ha="right" if short else "center", va="bottom", fontsize=8, color=INK2)
    axes[-1].set_xlabel("Minutes since the start of logging")
    axes[-1].set_xlim(0, t[-1])
    header(fig, "The board during the whole run",
           "Sampled every 2 s. Grey bands are the measuring steps; white is idle time and the cool-downs between them.")
    return save(fig, "03_thermal_power_timeline.png")


def graph_sustained(today: list[dict], ref: list[dict]) -> str | None:
    if not today:
        return None
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.2), sharex=True, gridspec_kw={"hspace": 0.4, "height_ratios": [1.3, 1, 1]})
    fig.subplots_adjust(left=0.09, right=0.97, top=top_margin(fig, 1.35), bottom=0.07)
    series = [("1 Oct", ref, BLUE), (f"{THIS} (this run)", today, ORANGE)]
    for lbl, rows, colour in series:
        if not rows:
            continue
        axes[0].plot([r["t"] / 60 for r in rows if r["temp"]], [r["temp"] for r in rows if r["temp"]], color=colour, lw=1.8)
        for ax, mode, scale in ((axes[1], "currency_mode_full", 1.0), (axes[2], "ocr_mode", 1000.0)):
            pts = [(r["t"] / 60, r["ms"] / scale) for r in rows if r["mode"] == mode]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour, lw=0, marker="o", markersize=3.2, alpha=0.8)
    axes[0].set_title("Core temperature (°C)", loc="left", fontsize=9.5, color=INK2, pad=4)
    axes[1].set_title("Currency check, end to end: every call (ms)", loc="left", fontsize=9.5, color=INK2, pad=4)
    axes[2].set_title("Text reading, 3-line sign: every call (s)", loc="left", fontsize=9.5, color=INK2, pad=4)
    for level, colour in ((80, WARNING), (85, CRITICAL)):
        axes[0].axhline(level, color=colour, lw=1.2, ls=(0, (5, 3)))
    for ax in axes:
        ax.grid(axis="x", visible=False)
        ax.set_ylim(bottom=0 if ax is not axes[0] else None)
    axes[-1].set_xlabel("Minutes into the 30-minute run")
    legend_above(fig, [Patch(color=c, label=lbl) for lbl, rows, c in series if rows], 2)
    header(fig, "30 minutes of non-stop use", "Every model in turn, round after round, as in scripts/benchmark_pi5.py --sustained 30.")
    return save(fig, "04_sustained_30_min.png")


def graph_ocr_configs(ocr: dict) -> str | None:
    cfgs = [(d, w, lbl, c) for (d, w, lbl), c in zip(OCR_CONFIGS, CONFIG_COLOURS) if (d, w) in ocr]
    if not cfgs:
        return None
    base = ocr[cfgs[0][:2]]
    cases = [r for r in base if r["path"] == "app"]
    n = len(cfgs)
    h = 0.8 / n
    fig, ax = plt.subplots(figsize=(10.5, 0.95 * len(cases) + 1.8))
    fig.subplots_adjust(left=0.37, right=0.97, top=top_margin(fig, 1.25), bottom=0.6 / fig.get_figheight() + 0.02)
    hi = 0
    for i, (d, w, lbl, colour) in enumerate(cfgs):
        rows = {(r["case"], r["path"]): r for r in ocr[(d, w)]}
        for y, c in enumerate(cases):
            r = rows.get((c["case"], "app"))
            if not r:
                continue
            v = r["total_ms_median"] / 1000
            hi = max(hi, v)
            yy = y - 0.4 + h * (i + 0.5)
            ax.barh(yy, v, height=h * 0.92, color=colour)
            ax.annotate(f"{v:.1f} s", (v, yy), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(range(len(cases)), [c["label"] for c in cases])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, hi * 1.15)
    ax.set_xlabel("Seconds from the capture button to the full text (median), as the app runs it")
    legend_above(fig, [Patch(color=c, label=lbl) for _, _, lbl, c in cfgs], 2)
    fig.subplots_adjust(top=top_margin(fig, 1.55))
    header(fig, "Text reading: the four detector settings", "OCRMode.process_frame() as main.py calls it, 3 to 5 captures per picture.")
    return save(fig, "05_ocr_settings.png")


def graph_ocr_breakdown(rows: list[dict], title_cfg: str) -> str | None:
    rows = [r for r in rows if r["path"] == "app"]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(10.5, 0.62 * len(rows) + 1.9))
    fig.subplots_adjust(left=0.37, right=0.97, top=top_margin(fig, 1.25), bottom=0.6 / fig.get_figheight() + 0.02)
    ys = list(range(len(rows)))
    left = [0.0] * len(rows)
    for key, colour in (("detect_ms_median", BLUE), ("recognize_ms_median", ORANGE)):
        width = [max(0.0, r[key]) / 1000 for r in rows]
        ax.barh(ys, width, left=left, height=0.56, color=colour, edgecolor=SURFACE, linewidth=2)
        left = [a + b for a, b in zip(left, width)]
    for y, r, end in zip(ys, rows, left):
        ax.annotate(sec(r["total_ms_median"]), (end, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9.5,
                    fontweight="bold")
        if r.get("first_line_ms_median") is not None:
            ax.scatter(r["first_line_ms_median"] / 1000, y, marker="D", s=75, color=INK, edgecolor=SURFACE, linewidth=2, zorder=5)
    ax.set_yticks(ys, [r["label"] for r in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(r["total_ms_median"] for r in rows) / 1000 * 1.14)
    ax.set_xlabel("Seconds from the capture button to the full text (median)")
    legend_above(fig, [Patch(color=BLUE, label="Finding the text (CRAFT detector)"), Patch(color=ORANGE, label="Reading the letters (CRNN)"),
                        Line2D([], [], color=INK, marker="D", markersize=7, lw=0, label="First line handed to speech")], 3)
    header(fig, "Text reading with the Pi's default settings: where the time goes", f"{title_cfg}. Everything outside the two steps is under 0.2 s.")
    return save(fig, "06_ocr_breakdown.png")


def graph_power(groups: list[tuple[str, dict]]) -> str | None:
    rows = [(n, s) for n, s in groups if s and s.get("w_mean") is not None]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(10.5, 0.55 * len(rows) + 1.6))
    fig.subplots_adjust(left=0.25, right=0.95, top=top_margin(fig, 1.2), bottom=0.6 / fig.get_figheight() + 0.03)
    ys = list(range(len(rows)))
    ax.barh(ys, [s["w_mean"] for _, s in rows], height=0.56, color=BLUE)
    for y, (_, s) in zip(ys, rows):
        ax.plot([s["w_mean"], s["w_max"]], [y, y], color=INK2, lw=1)
        ax.plot([s["w_max"], s["w_max"]], [y - 0.22, y + 0.22], color=INK, lw=1.6)
        ax.annotate(f"{s['w_mean']:.2f} W", (s["w_max"], y), xytext=(7, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(ys, [n for n, _ in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(s["w_max"] for _, s in rows) * 1.18)
    ax.set_xlabel("Watts drawn by the board's own circuits (sum of the power chip's rails)")
    legend_above(fig, [Patch(color=BLUE, label="Mean (the number at the end of the row)"),
                        Line2D([], [], color=INK, marker="|", markersize=10, markeredgewidth=1.6, lw=0, label="Highest 2-s sample")], 2)
    header(fig, "Board power by activity", "A lower bound of what the battery delivers: the 5 V input stage, fan and USB camera are not in the sum.")
    return save(fig, "07_power_by_activity.png")


def graph_loads(items: list[tuple[str, float]]) -> str | None:
    items = [(n, v) for n, v in items if v is not None]
    if not items:
        return None
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(items) + 1.5))
    fig.subplots_adjust(left=0.37, right=0.97, top=top_margin(fig), bottom=0.6 / fig.get_figheight() + 0.03)
    ys = list(range(len(items)))
    ax.barh(ys, [v / 1000 for _, v in items], height=0.56, color=BLUE)
    for y, (_, v) in zip(ys, items):
        ax.annotate(sec(v), (v / 1000, y), xytext=(7, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(ys, [n for n, _ in items])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(v for _, v in items) / 1000 * 1.14)
    ax.set_xlabel("Seconds, once per start (or once per first use of a mode)")
    header(fig, "Loading the models", "Time to load each model into memory. One measurement each, files already in the disk cache.")
    return save(fig, "08_model_load_times.png")


# ---------------------------------------------------------------- report
def main() -> None:
    bench_path = sorted(RAW.glob("pi5_benchmark_*.json"))[-1]
    bench = load(bench_path)
    host = bench["host"]
    iters = next(iter(bench["latency"].values()))["iters"]
    oct1, oct13 = load(OCT1), load(OCT13)
    oct13_extras, oct13_ocr = load(OCT13_EXTRAS) or {"results": {}}, load(OCT13_OCR)
    extras = load(RAW / "extras.json") or {"results": {}, "skipped": {}}
    switch = load(RAW / "mode_switch_currency.json") or {}
    live = load(RAW / "live_camera_test.json")
    ocr = {}
    for d, w, _ in OCR_CONFIGS:
        r = load(RAW / f"ocr_response_{d}_{w}.json")
        if r and not r.get("partial"):
            ocr[(d, w)] = r["results"]
    ocr_meta = {(d, w): load(RAW / f"ocr_response_{d}_{w}.json") for d, w, _ in OCR_CONFIGS}
    default_cfg = ("onnx_int8", 1200)
    ocr_rows = ocr.get(default_cfg, [])

    lat = dict(bench["latency"])
    for k in ("emotion_with_face", "object_yolov8n", "object_yolov8s"):
        if k in extras["results"]:
            lat[k] = extras["results"][k]
    ref1 = dict(oct1["latency"]) if oct1 else {}
    ref13 = dict(oct13["latency"]) if oct13 else {}
    for k in ("emotion_with_face", "object_yolov8n", "object_yolov8s"):
        if k in oct13_extras["results"]:
            ref13[k] = oct13_extras["results"][k]

    steps = [{"step": r["step"], "start": float(r["start_epoch"]), "end": float(r["end_epoch"]), "rc": r["exit_code"],
              "tb": num(r["temp_before_c"]), "ta": num(r["temp_after_c"])} for r in read_csv(RAW / "steps.csv")]
    step_by = {s["step"]: s for s in steps}

    def group_of(epoch: float, phase: str) -> str:
        if phase == "main_sustained":
            return "Sustained 30 min"
        for s in steps:
            if s["start"] <= epoch <= s["end"]:
                return STEP_LABELS.get(s["step"], "")
        return ""

    tl = []
    for r in read_csv(RAW / "system_timeline.csv"):
        e, temp = num(r["epoch"]), num(r["cpu_temp_c"])
        if e is None or temp is None:
            continue
        tl.append({"epoch": e, "phase": r["phase"], "group": group_of(e, r["phase"]), "cpu_temp_c": temp,
                   "throttled_hex": r["throttled_hex"], "cpu_busy_pct": num(r["cpu_busy_pct"]) or 0.0,
                   "mem_used_mb": num(r["mem_used_mb"]) or 0.0, "fan_rpm": num(r["fan_rpm"]) or 0.0,
                   "arm_clock_mhz": num(r.get("arm_clock_mhz")), "pmic_watts": num(r.get("pmic_watts"))})

    def stats(pred) -> dict | None:
        rows = [r for r in tl if pred(r)]
        if not rows:
            return None
        n = len(rows)
        w = [r["pmic_watts"] for r in rows if r["pmic_watts"] is not None]
        clk = [r["arm_clock_mhz"] for r in rows if r["arm_clock_mhz"] is not None]
        return {"n": n, "minutes": n * 2 / 60, "max": max(r["cpu_temp_c"] for r in rows),
                "mean": statistics.fmean(r["cpu_temp_c"] for r in rows), "min": min(r["cpu_temp_c"] for r in rows),
                "soft": 100 * sum(state(r["throttled_hex"]) == 1 for r in rows) / n,
                "hard": 100 * sum(state(r["throttled_hex"]) == 2 for r in rows) / n,
                "w_mean": statistics.fmean(w) if w else None, "w_max": max(w) if w else None,
                "clk_min": min(clk) if clk else None,
                "clk_low": 100 * sum(c < 2300 for c in clk) / len(clk) if clk else None,
                "cpu": statistics.fmean(r["cpu_busy_pct"] for r in rows), "fan": max(r["fan_rpm"] for r in rows)}

    def sus_rows(path: Path) -> list[dict]:
        out = []
        for r in read_csv(path):
            t, v = num(r["elapsed_s"]), num(r["latency_ms"])
            if t is not None and v is not None:
                out.append({"t": t, "mode": r["mode"], "ms": v, "temp": num(r["cpu_temp_c"]), "thr": r["throttled"],
                            "rss": num(r["rss_mb"])})
        return out

    sus = sus_rows(bench_path.with_suffix(".csv"))
    sus_ref = sus_rows(OCT1.with_suffix(".csv"))

    def by_case(case, path="app", rows=None):
        return next((r for r in (ocr_rows if rows is None else rows) if r["case"] == case and r["path"] == path), None)

    all_s = stats(lambda r: True)
    idle_s = stats(lambda r: r["group"] == "Idle" and r["phase"] == "idle_before")
    idle_after = stats(lambda r: r["phase"] == "idle_after")
    main_s = stats(lambda r: r["group"] == "Mode benchmark")
    sus_s = stats(lambda r: r["group"] == "Sustained 30 min")
    extras_s = stats(lambda r: r["group"] == "Extras")
    tests_s = stats(lambda r: r["group"] == "Tests")
    ocr_s = {cfg: stats(lambda r, c=cfg: r["group"] == STEP_LABELS[f"ocr_{c[0]}_{c[1]}"]) for cfg in [c[:2] for c in OCR_CONFIGS]}
    started = datetime.fromtimestamp(tl[0]["epoch"]) if tl else datetime.fromisoformat(host["utc"]).astimezone()
    ended = datetime.fromtimestamp(tl[-1]["epoch"]) if tl else None

    g = {"latency": graph_latency(lat, iters),
         "runs": graph_runs(lat, [("1 Oct (cool board)", ref1, BLUE_LIGHT), ("13 Oct (board at its limit)", ref13, BLUE)]),
         "timeline": graph_timeline(tl), "sustained": graph_sustained(sus, sus_ref),
         "ocr_cfg": graph_ocr_configs(ocr),
         "ocr": graph_ocr_breakdown(ocr_rows, OCR_CONFIGS[0][2]) if ocr_rows else None}
    power_groups = [("Idle (desktop open)", idle_s), ("Tests", tests_s), ("Mode benchmark", main_s), ("Sustained 30 min", sus_s),
                    ("Extras and camera", extras_s)] + [(f"Text reading, {lbl}", ocr_s.get((d, w))) for d, w, lbl in OCR_CONFIGS]
    g["power"] = graph_power(power_groups)
    default_meta = ocr_meta.get(default_cfg) or {}
    loads = [("Text reader (EasyOCR + 8-bit detector)", default_meta.get("load_ms")),
             ("Currency mode, first switch into it", switch.get("activate_first_ms")),
             ("Currency mode, warm-up pass", switch.get("warm_up_ms")),
             ("Emotion detector", extras["results"].get("emotion_with_face", {}).get("load_ms")),
             ("Object mode (YOLOv8s)", lat.get("object_mode", {}).get("load_ms")),
             ("Watermark checker", lat.get("watermark_int8_localizer", {}).get("load_ms")),
             ("Multi-view authenticator", lat.get("prmvt_1view", {}).get("load_ms")),
             ("Camera, open to first frame", extras["results"].get("camera", {}).get("open_to_first_frame_ms"))]
    g["loads"] = graph_loads(loads)

    L: list[str] = []
    w = L.append
    duration = f"{(tl[-1]['epoch'] - tl[0]['epoch']) / 60:.0f} minutes" if tl else ""
    env = (RAW / "environment.txt").read_text(encoding="utf-8", errors="ignore").splitlines() if (RAW / "environment.txt").is_file() else []
    commit = next((env[i + 1].split()[0][:7] for i, ln in enumerate(env[:-1]) if ln == "--- git" and env[i + 1].strip()), None)
    w(f"# SARA on the Raspberry Pi 5: benchmark report, {started:%-d %B %Y}")
    w("")
    w(f"Run on {started:%d %B %Y} from {started:%H:%M} to {ended:%H:%M} local time ({duration}), on the glass's own board "
      f"({host['device_model']}, {host['cpu_count']} cores, Python {host['python']}, torch {host.get('torch')}, "
      f"ONNX Runtime {host.get('onnxruntime')}, EasyOCR {host.get('easyocr')}, Ultralytics {host.get('ultralytics')}). "
      f"No GPU; everything runs on the CPU. Camera: {camera_name()}."
      + (f" Code: commit `{commit}`, with the benchmark-script changes listed in `raw/environment.txt`." if commit else ""))
    w("")

    # ---- headline
    sign = by_case("sign_en_3_lines", "plain")
    sign_app = by_case("sign_en_3_lines")
    page = by_case("page_12_lines_640")
    w("## Headline numbers")
    w("")
    w("| What | Median | 95th percentile |")
    w("|---|---:|---:|")
    for k in ("currency_mode_full", "taka_yolo_handheld_int8", "watermark_int8_localizer", "jaal_check_qduig", "object_mode",
              "emotion_with_face"):
        if k in lat:
            w(f"| {NAMES[k]} | {ms(lat[k]['median_ms'])} | {ms(lat[k]['p95_ms'])} |")
    if "ocr_mode" in lat:
        w(f"| Text reading, 3-line sign, {iters} calls (`benchmark_pi5.py`) | {sec(lat['ocr_mode']['median_ms'])} | "
          f"{sec(lat['ocr_mode']['p95_ms'])} |")
    if sign_app:
        w(f"| Text reading, 3-line sign, as the app runs it | {sec(sign_app['total_ms_median'])} | "
          f"slowest of {sign_app['reps']}: {sec(sign_app['total_ms_max'])} |")
    if page:
        w(f"| Text reading, 12-line page: first line spoken | {sec(page['first_line_ms_median'])} | – |")
        w(f"| Text reading, 12-line page: whole page | {sec(page['total_ms_median'])} | slowest of {page['reps']}: "
          f"{sec(page['total_ms_max'])} |")
    w("")
    if all_s:
        where = ", in the 30-minute sustained run" if sus_s and sus_s["max"] == all_s["max"] else ""
        bits = [f"Core temperature {idle_s['mean']:.1f} °C at idle, peak {all_s['max']:.1f} °C{where}" if idle_s else
                f"Peak core temperature {all_s['max']:.1f} °C{where}"]
        if idle_s and idle_s["w_mean"] is not None and sus_s and sus_s["w_mean"] is not None:
            bits.append(f"board power {idle_s['w_mean']:.1f} W at idle and {sus_s['w_mean']:.1f} W under sustained load")
        line = "; ".join(bits)
        w(line[0].upper() + line[1:] + ". "
          f"Throttle flag at the end of the mode benchmark: `{host.get('throttled_end')}` ({decode(host.get('throttled_end'))}).")
        w("")
    w("<!-- FINDINGS -->")
    w("")

    # ---- temperature and power
    w("## Temperature, clock and power")
    w("")
    w(f"The board started at {host['cpu_temp_c_start']:.1f} °C with the throttle flag at `{host['throttled_start']}` "
      f"({decode(host['throttled_start'])}) before the first timing call.")
    w("")
    if g["timeline"]:
        w(f"![The board during the whole run]({g['timeline']})")
        w("")
    w("| Step | Minutes | Mean °C | Peak °C | Soft limit active | Throttled or capped | Lowest clock | Clock under 2.3 GHz | Mean power | Peak power |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, s in [("Idle before (desktop open)", idle_s), ("Tests", tests_s), ("Mode benchmark", main_s),
                    ("Sustained 30 min", sus_s), ("Extras and camera", extras_s)] + \
                   [(f"Text reading, {lbl}", ocr_s.get((d, wd))) for d, wd, lbl in OCR_CONFIGS] + [("Idle after", idle_after)]:
        if s:
            w(f"| {name} | {s['minutes']:.1f} | {s['mean']:.1f} | {s['max']:.1f} | {s['soft']:.0f}% | {s['hard']:.0f}% | "
              f"{'–' if s['clk_min'] is None else f'{s['clk_min'] / 1000:.2f} GHz'} | "
              f"{'–' if s['clk_low'] is None else f'{s['clk_low']:.0f}%'} | {watts(s['w_mean'])} | {watts(s['w_max'])} |")
    w("")
    if tl:
        uv = any(int(r["throttled_hex"], 16) & 0x10001 for r in tl if r["throttled_hex"].startswith("0x"))
        w(f"- Under-voltage was {'reported' if uv else 'never reported'}"
          f"{', so the power supply is a suspect' if uv else ', so the power supply kept up'}. "
          f"The fan ran up to {all_s['fan']:,.0f} rpm.")
        w("- The temperature is the SoC's own sensor (`thermal_zone0`), the hottest point of the board. A thermometer on the "
          "case or the board surface reads lower.")
        w("- The clock is what the firmware measures (`vcgencmd measure_clock arm`). At idle the governor drops it to 1.5 GHz by "
          "design; only a low clock under load means throttling.")
    w("")
    if g["power"]:
        w(f"![Board power by activity]({g['power']})")
        w("")
    if idle_s and sus_s and idle_s["w_mean"] and sus_s["w_mean"]:
        def hours(p):
            return BANK_WH * BANK_EFF / (p * (1 + OVERHEAD))
        w(f"**Battery estimate.** With the formula of `scripts/power_log.py` (rail sum + {OVERHEAD:.0%} for the input stage, fan "
          f"and USB camera; a {BANK_WH:.0f} Wh / 10,000 mAh power bank delivering {BANK_EFF:.0%} of its rating): about "
          f"{hours(sus_s['w_mean']):.1f} h of non-stop use at the sustained-run power ({sus_s['w_mean']:.2f} W rail sum), and about "
          f"{hours(idle_s['w_mean']):.1f} h at idle ({idle_s['w_mean']:.2f} W). The glass's real bank capacity is not recorded "
          "in the project, and the overhead is an allowance, not a measurement; a run on the bank itself with a USB power "
          "meter would replace this estimate.")
        w("")

    # ---- full latency list
    w("## Full latency list")
    w("")
    w(f"Each row is {iters} timed calls after 3 warm-up calls, from `scripts/benchmark_pi5.py` unless marked. "
      "Frames are 640x480, the glass camera size.")
    w("")
    if g["latency"]:
        w(f"![Latency of every mode and model]({g['latency']})")
        w("")
    w("| Group | Benchmark | Median | 95th pct | Mean | Calls/s | Load | Memory after | Temp after |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for group, keys in GROUPS + [("Text", ["ocr_mode"])]:
        for k in keys:
            if k not in lat:
                continue
            r = lat[k]
            mark = " †" if k in extras["results"] else ""
            w(f"| {group} | {NAMES[k]}{mark} | {fmt_any(r['median_ms'])} | {fmt_any(r['p95_ms'])} | {fmt_any(r['mean_ms'])} | "
              f"{r['fps_median']:.2f} | {fmt_any(r.get('load_ms'))} | {r['rss_mb_after'] / 1000:.2f} GB | {r['cpu_temp_c_after']:.1f} °C |")
    w("")
    w(f"† from `benchmark_report/scripts/extras_bench.py` ({extras['results'].get('object_yolov8s', {}).get('iters', 30)} calls), "
      "because the datasets the project script reads are not on the Pi. These rows ran in a separate process, so their memory "
      "column cannot be compared with the others.")
    w("")
    w("Load is the time to build the model object in that process; the ONNX detectors are read from disk on their first call, "
      "so their load is near zero. Memory is the whole benchmark process after that row, not the size of one model.")
    w("")
    notes_frames = None
    m = re.search(r"timing currency_mode_full on (\d+) notes", (RAW / "main_benchmark.txt").read_text(encoding="utf-8", errors="ignore")) \
        if (RAW / "main_benchmark.txt").is_file() else None
    if m:
        notes_frames = int(m.group(1))
    if notes_frames:
        w(f"- **Currency.** The note rows cycle through the {notes_frames} note pictures on the Pi (the 500 and 1,000 Taka templates "
          "and the two pictures in `samples/`). In the code measured here, currency mode on the Pi loads the hand-held 8-bit "
          "detector (`best_handheld_int8.onnx`, the Pi default from 15 to 29 October; since commit `a349cfe` the Pi loads "
          "`best_handheld.pt`, which this run did not time). The earlier INT8 detector is timed for comparison.")
    if "emotion_with_face" in lat and "emotion" in lat:
        e = lat["emotion_with_face"]
        w(f"- **Emotion.** `benchmark_pi5.py` has no face pictures on the Pi, so its row is a frame with no face "
          f"({ms(lat['emotion']['median_ms'])}, the face search alone). With a face in view (the astronaut test picture, classifier "
          f"`{e.get('backend')}`) one answer takes {ms(e['median_ms'])}.")
    if "object_yolov8n" in lat:
        w(f"- **Objects.** The app uses YOLOv8s ({ms(lat['object_mode']['median_ms'])} per scan as the app runs it); YOLOv8n would "
          f"take {ms(lat['object_yolov8n']['median_ms'])}. No COCO pictures are on the Pi, so the scan is timed on the note pictures; "
          "the detector does the same work on any picture.")
    w("")

    # ---- against earlier runs
    if g["runs"]:
        w("## Against the 1 and 13 October runs")
        w("")
        w(f"![This run against the 1 and 13 October runs]({g['runs']})")
        w("")
        w(f"| Benchmark | 1 Oct | 13 Oct | {THIS} | Change from 1 Oct | Change from 13 Oct |")
        w("|---|---:|---:|---:|---:|---:|")
        for _, keys in GROUPS + [("Text", ["ocr_mode"])]:
            for k in keys:
                if k not in lat:
                    continue
                n_ = lat[k]["median_ms"]
                o1, o13 = ref1.get(k, {}).get("median_ms"), ref13.get(k, {}).get("median_ms")
                w(f"| {NAMES[k]} | {fmt_any(o1)} | {fmt_any(o13)} | **{fmt_any(n_)}** | {pct(n_, o1) if o1 else '–'} | "
                  f"{pct(n_, o13) if o13 else '–'} |")
        w("")
        w(f"1 October: 100 calls a row, board cool (flag `{oct1['host']['throttled_end']}` at the end). 13 October: 30 calls a row, "
          f"board at its temperature limit (flag `{oct13['host']['throttled_end']}`). This run: {iters} calls a row. The code "
          "changed between the runs too (hand-held detector, OCR pipeline v2 with one detector pass, 8-bit text detector on the "
          "Pi, YOLOv8s), so a row whose model changed is not a like-for-like speed comparison.")
        w("")

    # ---- sustained
    if sus:
        w("## 30 minutes of non-stop use")
        w("")
        temps = [r["temp"] for r in sus if r["temp"]]
        thr = sorted({r["thr"] for r in sus if r["thr"]})
        seen = 0
        for t in thr:
            seen |= int(t, 16)
        w(f"`--sustained 30` ran every model in turn for 30 minutes: {len(sus):,} calls. Core temperature, as the benchmark "
          f"read it after each call, {min(temps):.1f} to {max(temps):.1f} °C (the 2-second sampler's peak is in the table above); "
          f"throttle flags seen: {', '.join(f'`{t}`' for t in thr)}, together: {decode(hex(seen))}. "
          + (f"On 1 October the same run made {len(sus_ref):,} calls and peaked at {max(r['temp'] for r in sus_ref if r['temp']):.1f} °C."
             if sus_ref else ""))
        w("")
        if g["sustained"]:
            w(f"![30 minutes of non-stop use]({g['sustained']})")
            w("")
        w("| Model | Calls | Median, first 5 min | Median, last 5 min | Drift | Slowest call | Median in the timing pass |")
        w("|---|---:|---:|---:|---:|---:|---:|")
        end = max(r["t"] for r in sus)
        for _, keys in GROUPS + [("Text", ["ocr_mode"])]:
            for k in keys:
                v = [r for r in sus if r["mode"] == k]
                if not v:
                    continue
                first = [r["ms"] for r in v if r["t"] <= 300]
                last = [r["ms"] for r in v if r["t"] >= end - 300]
                a = statistics.median(first) if first else None
                b = statistics.median(last) if last else None
                w(f"| {NAMES[k]} | {len(v)} | {fmt_any(a)} | {fmt_any(b)} | {pct(b, a) if a and b else '–'} | "
                  f"{fmt_any(max(r['ms'] for r in v))} | {fmt_any(lat.get(k, {}).get('median_ms'))} |")
        rss = [r["rss"] for r in sus if r["rss"]]
        w("")
        if rss:
            w(f"Process memory over the 30 minutes: {rss[0] / 1000:.2f} GB at the start, {rss[-1] / 1000:.2f} GB at the end, "
              f"highest {max(rss) / 1000:.2f} GB.")
            w("")
        sc = load(RAW / "safecheck_after_onnx.json")
        if sc:
            w(f"**Why the safe-check is slower here.** In this loop the counterfeit safe-check always runs straight after the "
              f"watermark localizer, an ONNX Runtime model, and its median is about twice its timing-pass median (the 1 October "
              f"run shows the same). `benchmark_report/scripts/safecheck_after_onnx.py` timed it, after the run, in "
              f"{sc['alone']['n']}-call blocks:")
            w("")
            w("| Order | Median | 95th pct |")
            w("|---|---:|---:|")
            for k, label in (("alone", "Safe-check alone, calls in a row"),
                             ("after_detector", "Straight after the hand-held note detector (ONNX), as in the app"),
                             ("after_detector_pause_50ms", "The same with a 50 ms pause between the two"),
                             ("after_pytorch_detector", "Straight after the PyTorch note detector"),
                             ("alone_again", "Alone again, at the end")):
                if k in sc:
                    w(f"| {label} | {ms(sc[k]['median_ms'])} | {ms(sc[k]['p95_ms'])} |")
            w("")
            w(f"Right after an ONNX Runtime call the safe-check takes {sc['after_detector']['median_ms'] - sc['alone']['median_ms']:.0f} ms "
              "longer; a short pause or a PyTorch model before it removes the extra time. The likely cause is ONNX Runtime's "
              "worker threads, which keep spinning on the four cores for a moment after a run and compete with PyTorch's. In the "
              "app the safe-check follows the ONNX note detector, so its real cost is the slower figure, not the timing-pass "
              "median. The end-to-end currency row includes it, since `process_frame` runs the safe-check on each note the detector "
              "finds.")
            w("")

    # ---- OCR
    if ocr:
        s0 = (default_meta or {}).get("settings") or {}
        w("## Text reading (OCR)")
        w("")
        w(f"`OCRMode.process_frame()`, the call `main.py` makes when ACTION is pressed in text mode, on rendered signs and pages "
          f"whose text is known; 3 to 5 captures per picture. Pipeline `{s0.get('OCR_PIPELINE')}`, decoder "
          f"`{s0.get('OCR_V2_DECODER')}`, canvas {s0.get('OCR_CANVAS_SIZE')}, {s0.get('torch_threads')} threads. "
          "`config.py` sets the 8-bit text detector as the Pi default and notes that its speed on the Pi had not been measured; "
          "this run times it against the full-precision detector, each at the two read widths "
          "(the four settings of `scripts/pi_check_all.sh`), with a cool-down before each.")
        w("")
        if g["ocr_cfg"]:
            w(f"![Text reading: the four detector settings]({g['ocr_cfg']})")
            w("")
        cfgs = [c for c in OCR_CONFIGS if c[:2] in ocr]
        w("| Capture | " + " | ".join(lbl for _, _, lbl in cfgs) + " |")
        w("|---|" + "---:|" * len(cfgs))
        for r0 in [r for r in ocr[cfgs[0][:2]]]:
            cells = []
            for d, wd, _ in cfgs:
                r = by_case(r0["case"], r0["path"], ocr[(d, wd)])
                cells.append("–" if not r else f"{sec(r['total_ms_median'])}" + ("" if r["cer"] is None else f" ({100 * r['cer']:.1f}% err)"))
            label = r0["label"] + (" — text found once, `benchmark_pi5.py` path" if r0["path"] == "plain" else "")
            w(f"| {label} | " + " | ".join(cells) + " |")
        cer_row = []
        pass_row = []
        for d, wd, _ in cfgs:
            v = [r["cer"] for r in ocr[(d, wd)] if r["path"] == "app" and r["cer"] is not None]
            cer_row.append(f"{100 * statistics.fmean(v):.2f}%" if v else "–")
            p = [r["detect_ms_median"] / max(1, r["detect_calls"]) for r in ocr[(d, wd)]]
            pass_row.append(f"{min(p) / 1000:.1f} – {max(p) / 1000:.1f} s")
        w("| **Mean character error, app path** | " + " | ".join(cer_row) + " |")
        w("| One pass of the text detector | " + " | ".join(pass_row) + " |")
        w("")
        w("The character errors are on clean rendered pictures, so they say mostly that the timed reads were real reads; the "
          "project's accuracy numbers for each setting are the validation results in `config.py`.")
        w("")
        if g["ocr"]:
            w(f"![Text reading with the Pi's default settings]({g['ocr']})")
            w("")
        w(f"With the Pi's default settings ({OCR_CONFIGS[0][2]}):")
        w("")
        w("| Capture | Runs | Total | Range | Finding text | Reading letters | Rest | First line spoken | Detector passes | Character errors | 13 Oct total |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        old_rows = (oct13_ocr or {}).get("results", [])
        for r in [r for r in ocr_rows if r["path"] == "app"]:
            o = by_case(r["case"], "app", old_rows)
            w(f"| {r['label']} | {r['reps']} | **{sec(r['total_ms_median'])}** | {sec(r['total_ms_min'])} – {sec(r['total_ms_max'])} | "
              f"{sec(r['detect_ms_median'])} | {sec(r['recognize_ms_median'])} | {sec(r['other_ms_median'])} | "
              f"{sec(r['first_line_ms_median'])} | {r['detect_calls']} | {'–' if r['cer'] is None else f'{100 * r['cer']:.1f}%'} | "
              f"{sec(o['total_ms_median']) if o else '–'} |")
        w("")
        blank = by_case("no_text")
        if blank:
            w(f"- **No text in view:** {sec(blank['total_ms_median'])}, answer "
              + (f"\"{blank['read']}\" — the reader still invents letters on a blank frame." if blank["read"] else
                 "empty (the glass says no text was found). On 13 October it returned 48 made-up letters here."))
        w(f"- Loading the text reader took {sec(default_meta.get('load_ms'))}; the first capture after loading took "
          f"{sec(default_meta.get('first_capture_after_load_ms'))}.")
        w("")
        w("What the glass read (default settings, app path, last run of each capture):")
        w("")
        w("| Capture | Text on the picture | Text the glass read |")
        w("|---|---|---|")
        for r in [r for r in ocr_rows if r["path"] == "app" and r["truth"]]:
            w(f"| {r['label']} | {r['truth']} | {r['read'] or '(nothing)'} |")
        w("")

    # ---- live camera
    if live and live.get("status") == "ok":
        lt = live["latency"]
        w("## Live camera loop")
        w("")
        w(f"`scripts/test_live_camera.py --seconds 60`: real frames from the C270 webcam ({live['camera']['resolution'][0]}x"
          f"{live['camera']['resolution'][1]}, mean brightness {live['camera']['mean_brightness_0_255']} of 255) through the "
          f"glass's own currency and object modes, and every fifth frame through emotion. {live['frames_processed']} frames in "
          f"{live['seconds']} s: {live['loop_fps']} frames a second with all of it running.")
        w("")
        w("| Stage | Calls | Median | 95th pct |")
        w("|---|---:|---:|---:|")
        for k, label in (("capture", "Camera read"), ("currency_detect_and_jaal", "Currency detection + safe check"),
                         ("object_mode", "Object scan"), ("emotion", "Emotion (every 5th frame)"), ("frame_total", "Whole frame")):
            if lt.get(k):
                w(f"| {label} | {lt[k]['n']} | {ms(lt[k]['median_ms'])} | {ms(lt[k]['p95_ms'])} |")
        w("")
        w(f"Notes seen: {live['notes_detected'] or 'none'}; objects announced: {live['object_announcements'] or 'none'}; "
          f"faces: {live['emotion_backend_counts']}. The run was unattended, so nothing was held up to the camera; this measures "
          "speed on real camera frames, not accuracy.")
        w("")
    elif live:
        w("## Live camera loop")
        w("")
        w(f"Not measured: {live.get('reason')}.")
        w("")

    # ---- field session of the same day
    run_start = tl[0]["epoch"] if tl else float("inf")
    sessions = []
    for folder in sorted((GLASS / "logs" / "field").glob(f"{started:%Y%m%d}_*")):
        meta = load(folder / "session.json") or {}
        try:
            begun = datetime.fromisoformat(meta["start"]).timestamp()
        except (KeyError, ValueError):
            continue
        if begun < run_start:   # the app as the team used it; sessions the tests started are left out
            sessions.append({"name": folder.name, "meta": meta, "events": read_csv(folder / "events.csv"),
                             "system": read_csv(folder / "system.csv")})
    if sessions:
        w("## The app itself, used earlier today")
        w("")
        w(f"`python main.py` with the real camera and buttons, before this benchmark "
          f"({', '.join('`' + s['name'] + '`, ' + s['meta']['start'][11:16] + '–' + s['meta'].get('end', '')[11:16] for s in sessions)}). "
          "These are the latencies the app's own field log recorded, from button press to the spoken answer being queued.")
        w("")
        w("| What | Times | Median | Range |")
        w("|---|---:|---:|---:|")
        grouped = {}
        for s in sessions:
            for e in s["events"]:
                if e.get("latency_ms"):
                    grouped.setdefault((e["mode"], e["event"]), []).append(float(e["latency_ms"]))
        for key, label in FIELD_NAMES.items():
            v = grouped.get(key)
            if v:
                fmt = sec if max(v) >= 2000 else ms
                w(f"| {label} | {len(v)} | {fmt(statistics.median(v))} | {'–' if len(v) == 1 else f'{fmt(min(v))} – {fmt(max(v))}'} |")
        w("")
        temps = [float(r["cpu_temp_c"]) for s in sessions for r in s["system"] if r.get("cpu_temp_c")]
        rss = [float(r["app_rss_mb"]) for s in sessions for r in s["system"] if r.get("app_rss_mb")]
        verdicts = {}
        for s in sessions:
            for e in s["events"]:
                if e["event"] == "watermark_check":
                    verdicts[e["verdict"]] = verdicts.get(e["verdict"], 0) + 1
        bits = []
        if temps:
            bits.append(f"The app logged {min(temps):.1f} to {max(temps):.1f} °C.")
        if rss:
            bits.append(f"It held up to {max(rss) / 1000:.2f} GB of memory with all modes loaded and the camera open.")
        if verdicts:
            bits.append(f"The guided watermark check ended {', '.join(f'{n}× {v}' for v, n in verdicts.items())}: the note was "
                        "not held up to a light long enough (its frames were rejected as blurry or as having no note), so these "
                        "are the time-out, not a watermark reading.")
        w(" ".join(bits))
        w("")
        reads = []
        for s in sessions:
            for e in s["events"]:
                if (e["mode"], e["event"]) == ("ocr", "result") and e.get("detail"):
                    try:
                        text = json.loads(e["detail"]).get("ocr_text", "")
                    except ValueError:
                        continue
                    if text:
                        reads.append((e.get("lang") or "–", float(e["latency_ms"]), text))
        if reads:
            w("What those text captures read. They were photos of printed pages, longer than the 3-line sign, so their times "
              "compare with the 6-line label and the 12-line page in the text-reading section:")
            w("")
            w("| Language | Characters read | Time | Text the glass read |")
            w("|---|---:|---:|---|")
            for lang, t, text in reads:
                w(f"| {({'en': 'English', 'bn': 'Bangla'}).get(lang, lang)} | {len(text)} | {sec(t)} | {text.replace('|', '/')} |")
            w("")

    # ---- loading, speech, camera
    w("## Loading, speech and camera")
    w("")
    if g["loads"]:
        w(f"![Loading the models]({g['loads']})")
        w("")
    w("| What | Time |")
    w("|---|---:|")
    for name, v in loads:
        if v is not None:
            w(f"| {name} | {sec(v)} |")
    if switch:
        w(f"| Currency mode, first check after switching | {ms(switch.get('first_check_ms'))} |")
        w(f"| Currency mode, later checks | {ms(switch.get('steady_check_ms'))} |")
        w(f"| Currency mode, switching back into it | {ms(switch.get('activate_again_ms'))} |")
    for k, label in (("speech_bn_short", "Speech: short Bangla sentence"), ("speech_en_short", "Speech: short English sentence"),
                     ("speech_bn_page", "Speech: six Bangla lines")):
        r = extras["results"].get(k)
        if r:
            w(f"| {label} ({r['audio_seconds']} s of audio) | {ms(r['median_ms'])} |")
    cam = extras["results"].get("camera")
    if cam:
        w(f"| Camera: time between frames ({cam['size']}) | {ms(cam['frame_interval_ms_median'])} ({cam['fps_median']} frames/s) |")
    w("")
    w("Speech is espeak-ng rendering the audio to a pipe, not playing it: rendering adds almost nothing to the response time; "
      "the listener still has to hear the sentence out.")
    w("")

    # ---- tests
    tests = read_csv(RAW / "tests.csv")
    if tests:
        w("## Tests")
        w("")
        w("The project's hardware-free test scripts, run on the Pi before the benchmark (`raw/tests/<name>.txt` holds each output).")
        w("")
        w("| Test | Result | Time | Last line of output |")
        w("|---|---|---:|---|")
        for t in tests:
            path = RAW / "tests" / f"{t['test']}.txt"
            lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()] if path.is_file() else []
            last = (lines[-1] if lines else "").replace("|", "\\|")
            last = last if len(last) <= 110 else last[:107] + "…"
            w(f"| `{t['test']}` | {'pass' if t['exit_code'] == '0' else 'FAIL (exit ' + t['exit_code'] + ')'} | "
              f"{float(t['seconds']):.0f} s | {last} |")
        w("")
        w("<!-- TESTS -->")
        w("")
    pre = RAW / "preflight.txt"
    if pre.is_file():
        rc = step_by.get("preflight", {}).get("rc")
        w(f"Preflight (`scripts/pi5_preflight.py`): exit {rc}"
          + (", every model file present and the ONNX graphs ran a dummy input." if rc == "0" else "."))
        w("")

    # ---- method
    w("## How this was measured")
    w("")
    w("All steps ran from one script, `benchmark_report/scripts/run_full_benchmark.sh`, while "
      "`benchmark_report/scripts/sys_sampler.py` logged temperature, throttle flags, measured clock, CPU, memory, fan and "
      "board power every 2 s (`raw/system_timeline.csv`). Between the heavy steps the script waited until the board was back "
      "under 63 °C (at most 10 minutes).")
    w("")
    w("1. 2 minutes idle, then `scripts/pi5_preflight.py` and the test scripts.")
    w(f"2. `scripts/benchmark_pi5.py --iters {iters} --warmup 3 --sustained 30`, the protocol in `HOW_TO_RUN.md` "
      f"(`raw/{bench_path.name}` and the `.csv` of the sustained run; the script also keeps a copy in `savior_glass/results/`).")
    w("3. `scripts/measure_mode_switch.py`, `benchmark_report/scripts/extras_bench.py` (emotion with a face, both object "
      "detectors, speech, camera) and `scripts/test_live_camera.py --seconds 60` (real webcam frames).")
    w("4. `benchmark_report/scripts/ocr_response_bench.py` once for each of the four text-detector settings.")
    w("5. 3 minutes idle.")
    if (RAW / "safecheck_after_onnx.json").is_file():
        w("6. Afterwards, from a cool board: `benchmark_report/scripts/safecheck_after_onnx.py` (the safe-check after the ONNX "
          "detector, `raw/safecheck_after_onnx.json`) and the reruns of the tests in the Tests section.")
    w("")
    w("Limits of this run:")
    w("")
    w("- VS Code (with this Claude Code session), its language servers and a Chromium window stayed open on the Pi during the "
      "run; they use some CPU and memory (`raw/environment.txt` lists the top processes at the start).")
    w("- Only four note pictures are on the Pi; OCR and emotion pictures are rendered or stock pictures, not glass-camera photos.")
    w("- The live-camera loop was unattended: it measures speed on real frames, not what the glass says about a real note.")
    w("- Board power is the sum of the power chip's rails, a lower bound; the battery figure is an estimate with stated assumptions.")
    for name, why in {**bench.get("skipped", {}), **extras.get("skipped", {})}.items():
        w(f"- Skipped: `{name}`: {why}")
    w("")
    w(f"To rebuild this report from the raw files: `~/.venvs/savior_glass/bin/python benchmark_report/scripts/make_run_report.py "
      f"benchmark_report/{RUN.name}`.")
    w("")

    md = "\n".join(L)
    findings = RUN / "findings.md"
    md = md.replace("<!-- FINDINGS -->", findings.read_text(encoding="utf-8").strip() if findings.is_file() else "")
    tests_note = RUN / "tests_note.md"
    md = md.replace("<!-- TESTS -->", tests_note.read_text(encoding="utf-8").strip() if tests_note.is_file() else "")
    md = re.sub(r"\n{3,}", "\n\n", md)
    (RUN / "REPORT.md").write_text(md, encoding="utf-8")

    import markdown
    body = markdown.markdown(md, extensions=["tables"])

    def inline(match):
        data = base64.b64encode((RUN / match.group(1)).read_bytes()).decode()
        return f'src="data:image/png;base64,{data}"'

    body = re.sub(r'src="(graphs/[^"]+)"', inline, body)
    css = ("body{font:16px/1.55 system-ui,'Noto Sans','Noto Sans Bengali',sans-serif;color:#0b0b0b;background:#fcfcfb;"
           "max-width:1000px;margin:0 auto;padding:24px 16px 64px}h1{font-size:28px;margin:0 0 8px}"
           "h2{font-size:21px;margin:40px 0 8px;padding-top:16px;border-top:1px solid #e6e5e0}img{max-width:100%;height:auto}"
           "table{border-collapse:collapse;margin:12px 0;font-size:14px;display:block;overflow-x:auto}"
           "th,td{padding:6px 10px;border-bottom:1px solid #e6e5e0;text-align:left;vertical-align:top}"
           "th{color:#52514e;font-weight:600;border-bottom:2px solid #c9c8c2}td[style*=right],th[style*=right]{text-align:right;"
           "font-variant-numeric:tabular-nums;white-space:nowrap}code{background:#f1f0ec;padding:1px 5px;border-radius:4px;"
           "font-size:.9em}li{margin:6px 0}")
    (RUN / "REPORT.html").write_text(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>SARA Pi 5 benchmark, {THIS}</title><style>{css}</style></head><body>{body}</body></html>", encoding="utf-8")
    print("wrote", RUN / "REPORT.md", "and REPORT.html;", len(list(GRAPHS.glob("*.png"))), "graphs")


if __name__ == "__main__":
    main()
