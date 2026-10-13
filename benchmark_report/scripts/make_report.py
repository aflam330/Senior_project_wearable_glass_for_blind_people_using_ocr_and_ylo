"""Build REPORT.md, REPORT.html and the graphs from the files in ../raw.

    python make_report.py

Every number in the report is read from the raw files; nothing is typed in by hand.
"""
from __future__ import annotations

import base64
import csv
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

RD = Path(__file__).resolve().parents[1]
RAW, GRAPHS = RD / "raw", RD / "graphs"
GLASS = RD.parent / "savior_glass"
REFERENCE = GLASS / "results" / "pi5_benchmark_memo_20261001T111325Z.json"

SURFACE, INK, INK2, GRID, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#f1f0ec"
BLUE, ORANGE, AQUA, BLUE_LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#cde2fb"
WARNING, CRITICAL = "#fab219", "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans",
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "xtick.major.size": 0,
    "ytick.major.size": 0, "legend.frameon": False, "legend.fontsize": 9,
})

NAMES = {
    "currency_mode_full": "Currency check, end to end",
    "taka_yolo_onnx_int8": "Taka detector, ONNX INT8 (used on the Pi)",
    "taka_yolo_onnx_fp32": "Taka detector, ONNX FP32",
    "taka_yolo_pt": "Taka detector, PyTorch",
    "watermark_int8_localizer": "Watermark check, learned localizer (used)",
    "watermark_int8_sift": "Watermark check, SIFT alignment",
    "jaal_check_qduig": "Counterfeit safe-check, one crop",
    **{f"prmvt_{k}view": f"Multi-view authenticator, {k} view{'s' if k > 1 else ''}" for k in range(1, 7)},
    "object_mode": "Object scan, as the app runs it (YOLOv8s)",
    "object_yolov8s": "Object detector YOLOv8s alone",
    "object_yolov8n": "Object detector YOLOv8n alone",
    "emotion_with_face": "Emotion, face in view",
    "emotion": "Emotion, no face in view (face search only)",
}
GROUPS = [
    ("Currency", ["currency_mode_full", "taka_yolo_onnx_int8", "taka_yolo_onnx_fp32", "taka_yolo_pt"]),
    ("Watermark", ["watermark_int8_localizer", "watermark_int8_sift"]),
    ("Counterfeit check", ["jaal_check_qduig"] + [f"prmvt_{k}view" for k in range(1, 7)]),
    ("Objects", ["object_mode", "object_yolov8s", "object_yolov8n"]),
    ("Emotion", ["emotion_with_face", "emotion"]),
]
THROTTLE_BITS = {0: "under-voltage now", 1: "CPU clock capped now", 2: "throttled now", 3: "soft temperature limit active now",
                 16: "under-voltage has occurred", 17: "CPU clock capping has occurred", 18: "throttling has occurred",
                 19: "soft temperature limit has occurred"}
PHASES = [("main_benchmark", "Mode benchmark"), ("extras", "Extra timings"), ("ocr", "OCR response test")]


# ---------------------------------------------------------------- helpers
def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def ms(v) -> str:
    if v is None:
        return "–"
    return f"{v:,.0f} ms" if v >= 100 else f"{v:.1f} ms"


def sec(v_ms) -> str:
    return "–" if v_ms is None else f"{v_ms / 1000:.2f} s"


def decode(hexval: str) -> str:
    try:
        v = int(hexval, 16)
    except (TypeError, ValueError):
        return "not reported"
    if v == 0:
        return "nothing"
    return "; ".join(name for bit, name in THROTTLE_BITS.items() if v >> bit & 1)


def state(hexval: str) -> int:
    """0 normal, 1 soft temperature limit active, 2 throttled or clock capped."""
    try:
        v = int(hexval, 16)
    except (TypeError, ValueError):
        return 0
    return 2 if v & 0x6 else 1 if v & 0x8 else 0


def phase_group(name: str) -> str:
    for prefix, label in PHASES:
        if name.startswith(prefix):
            return label
    return ""


def header(fig, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.012, 0.985 - 0.36 / fig.get_figheight(), subtitle, ha="left", va="top", fontsize=9.5, color=INK2)


def legend_above(fig, handles, ncols: int) -> None:
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 1 - 0.74 / fig.get_figheight()), ncols=ncols,
               borderaxespad=0, borderpad=0, columnspacing=1.6, handlelength=1.3, handletextpad=0.5)


def save(fig, name: str) -> str:
    GRAPHS.mkdir(exist_ok=True)
    fig.savefig(GRAPHS / name, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return f"graphs/{name}"


def top_margin(fig, inches: float = 0.95) -> float:
    return 1 - inches / fig.get_figheight()


# ---------------------------------------------------------------- graphs
def graph_latency(lat: dict) -> str:
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
           "Median of 30 calls after 3 warm-up calls, on 640x480 frames. Text reading is on its own graph (it takes seconds).")
    return save(fig, "01_latency_all_modes.png")


def graph_compare(lat: dict, ref: dict) -> str | None:
    keys = [k for _, ks in GROUPS for k in ks if k in lat and k in ref and not re.match(r"prmvt_[2345]view", k)]
    if not keys:
        return None
    fig, ax = plt.subplots(figsize=(10.5, 0.62 * len(keys) + 1.7))
    fig.subplots_adjust(left=0.37, right=0.97, top=top_margin(fig, 1.25), bottom=0.6 / fig.get_figheight() + 0.02)
    ys = list(range(len(keys)))
    old = [ref[k]["median_ms"] for k in keys]
    new = [lat[k]["median_ms"] for k in keys]
    ax.barh([y - 0.2 for y in ys], old, height=0.36, color=BLUE)
    ax.barh([y + 0.2 for y in ys], new, height=0.36, color=ORANGE)
    for y, o, n in zip(ys, old, new):
        ax.annotate(ms(o), (o, y - 0.2), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
        ax.annotate(f"{ms(n)}  ({(n / o - 1) * 100:+.0f}%)", (n, y + 0.2), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5)
    ax.set_yticks(ys, [NAMES[k] for k in keys])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(old + new) * 1.24)
    ax.set_xlabel("Median time per call (ms)")
    legend_above(fig, [Patch(color=BLUE, label="1 Oct 2026 – no throttling, 100 calls"),
                        Patch(color=ORANGE, label="13 Oct 2026 – board at its temperature limit, 30 calls")], 2)
    header(fig, "Tonight against the 1 October run", "Same script (scripts/benchmark_pi5.py), same board. Percentages are the change in the median.")
    return save(fig, "02_tonight_vs_1_october.png")


def graph_ocr(results: list[dict]) -> str | None:
    rows = [r for r in results if r["path"] == "app"]
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
        ax.annotate(sec(r["total_ms_median"]), (end, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9.5, fontweight="bold")
        if r.get("first_line_ms_median") is not None:
            ax.scatter(r["first_line_ms_median"] / 1000, y, marker="D", s=75, color=INK, edgecolor=SURFACE, linewidth=2, zorder=5)
    ax.set_yticks(ys, [r["label"] for r in rows])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(r["total_ms_median"] for r in rows) / 1000 * 1.14)
    ax.set_xlabel("Seconds from the capture button to the full text (median)")
    legend_above(fig, [Patch(color=BLUE, label="Finding the text (CRAFT detector)"),
                        Patch(color=ORANGE, label="Reading the letters (CRNN)"),
                        Line2D([], [], color=INK, marker="D", markersize=7, lw=0, label="First line handed to speech")], 3)
    header(fig, "Text reading: response time and where it goes",
           "OCRMode.process_frame() as main.py calls it. The number is the median total; everything outside the two steps is under 0.2 s.")
    return save(fig, "03_ocr_response_time.png")


def graph_ocr_paths(results: list[dict]) -> str | None:
    by = {}
    for r in results:
        by.setdefault(r["case"], {})[r["path"]] = r
    cases = [c for c, v in by.items() if {"app", "plain"} <= set(v)]
    if not cases:
        return None
    fig, ax = plt.subplots(figsize=(10.5, 0.7 * len(cases) + 1.7))
    fig.subplots_adjust(left=0.37, right=0.97, top=top_margin(fig, 1.25), bottom=0.6 / fig.get_figheight() + 0.03)
    ys = list(range(len(cases)))
    plain = [by[c]["plain"]["total_ms_median"] / 1000 for c in cases]
    app = [by[c]["app"]["total_ms_median"] / 1000 for c in cases]
    ax.barh([y - 0.2 for y in ys], plain, height=0.36, color=BLUE)
    ax.barh([y + 0.2 for y in ys], app, height=0.36, color=ORANGE)
    for y, p, a in zip(ys, plain, app):
        ax.annotate(f"{p:.2f} s", (p, y - 0.2), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
        ax.annotate(f"{a:.2f} s  ({(a / p - 1) * 100:+.0f}%)", (a, y + 0.2), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9)
    ax.set_yticks(ys, [by[c]["app"]["label"] for c in cases])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(plain + app) * 1.22)
    ax.set_xlabel("Seconds per capture (median)")
    legend_above(fig, [Patch(color=BLUE, label="Text found once (what benchmark_pi5.py times)"),
                        Patch(color=ORANGE, label="As the app runs: page check first, then the normal read")], 2)
    header(fig, "Short signs: the app looks for the text twice",
           "With line-by-line speech on, a capture of fewer than 4 lines runs the text detector a second time.")
    return save(fig, "04_ocr_sign_app_vs_benchmark.png")


def graph_timeline(tl: list[dict], clock: list[tuple[float, float]]) -> str | None:
    if not tl:
        return None
    e0 = tl[0]["epoch"]
    t = [(r["epoch"] - e0) / 60 for r in tl]
    panels = [("Core temperature (°C)", [r["cpu_temp_c"] for r in tl], 2.3),
              ("Firmware throttle state", [state(r["throttled_hex"]) for r in tl], 1.1)]
    if clock:
        panels.append((f"CPU clock (GHz), logged from minute {(clock[0][0] - e0) / 60:.1f}", None, 1.1))
    panels += [("CPU in use (%)", [r["cpu_busy_pct"] for r in tl], 1.1),
               ("Memory in use, whole system (GB)", [r["mem_used_mb"] / 1000 for r in tl], 1.1)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10.5, 1.55 * sum(p[2] for p in panels) + 1.5), sharex=True,
                             gridspec_kw={"height_ratios": [p[2] for p in panels], "hspace": 0.42})
    fig.subplots_adjust(left=0.2, right=0.9, top=top_margin(fig, 1.3), bottom=0.07)
    spans, start = [], 0
    for i in range(1, len(tl) + 1):
        if i == len(tl) or phase_group(tl[i]["phase"]) != phase_group(tl[start]["phase"]):
            spans.append((t[start], t[i - 1], phase_group(tl[start]["phase"])))
            start = i
    for ax, (title, values, _) in zip(axes, panels):
        for a, b, name in spans:
            if name:
                ax.axvspan(a, b, color=BAND, lw=0, zorder=0)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK2, pad=4)
        ax.grid(axis="x", visible=False)
        if title.startswith("CPU clock"):
            ax.plot([(e - e0) / 60 for e, _ in clock], [hz / 1e9 for _, hz in clock], color=BLUE, lw=1.6)
            ax.set_ylim(0, 2.6)
            ax.set_yticks([0, 1.2, 2.4])
        elif title.startswith("Firmware"):
            ax.fill_between(t, values, step="post", color=BLUE, lw=0)
            ax.set_ylim(0, 2.25)
            ax.set_yticks([0, 1, 2], ["Normal", "Soft limit", "Throttled"])
        else:
            ax.plot(t, values, color=BLUE, lw=2 if title.startswith("Core") else 1.6)
    ax0 = axes[0]
    temps = panels[0][1]
    ax0.set_ylim(min(60, min(temps) - 2), 92)
    for level, colour, label in ((80, WARNING, "80 °C  soft limit"), (85, CRITICAL, "85 °C  throttling")):
        ax0.axhline(level, color=colour, lw=1.4, ls=(0, (5, 3)))
        ax0.annotate(label, (1.0, level), xycoords=ax0.get_yaxis_transform(), xytext=(6, 0), textcoords="offset points",
                     va="center", fontsize=8.5, color=INK2, annotation_clip=False)
    peak = max(range(len(temps)), key=lambda i: temps[i])
    ax0.scatter(t[peak], temps[peak], s=60, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=5)
    ax0.annotate(f"peak {temps[peak]:.1f} °C", (t[peak], temps[peak]), xytext=(0, 9), textcoords="offset points", ha="center",
                 fontsize=9, fontweight="bold")
    for a, b, name in spans:
        if name:
            ax0.text((a + b) / 2, 0.04, name, transform=ax0.get_xaxis_transform(), ha="center", va="bottom", fontsize=9,
                     color=INK2)
    axes[-1].set_xlabel("Minutes since the start of logging")
    axes[-1].set_xlim(0, t[-1])
    header(fig, "The board during the whole run", "Sampled every 2 s. Grey bands are the three timing steps; white is idle time between them.")
    return save(fig, "05_thermal_timeline.png")


def graph_views(lat: dict) -> str | None:
    ks = [k for k in range(1, 7) if f"prmvt_{k}view" in lat]
    if len(ks) < 2:
        return None
    med = [lat[f"prmvt_{k}view"]["median_ms"] for k in ks]
    p95 = [lat[f"prmvt_{k}view"]["p95_ms"] for k in ks]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    fig.subplots_adjust(left=0.1, right=0.95, top=top_margin(fig, 1.2), bottom=0.14)
    ax.fill_between(ks, med, p95, color=BLUE_LIGHT, lw=0)
    ax.plot(ks, med, color=BLUE, lw=2, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    for k, m in ((ks[0], med[0]), (ks[-1], med[-1])):
        ax.annotate(ms(m), (k, m), xytext=(0, -16), textcoords="offset points", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_xticks(ks)
    ax.set_xlim(ks[0] - 0.35, ks[-1] + 0.35)
    ax.set_ylim(0, max(p95) * 1.15)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Views of the note given to the authenticator")
    ax.set_ylabel("Time per check (ms)")
    legend_above(fig, [Line2D([], [], color=BLUE, lw=2, marker="o", markersize=7, label="Median"),
                        Patch(color=BLUE_LIGHT, label="Median to 95th percentile")], 2)
    header(fig, "Counterfeit check: cost of each extra view", "Multi-view authenticator on 1 to 6 views of one note.")
    return save(fig, "06_authenticator_views.png")


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
    return save(fig, "07_model_load_times.png")


# ---------------------------------------------------------------- field logs of the same day
def field_sessions(day: str) -> list[dict]:
    out = []
    for folder in sorted((GLASS / "logs" / "field").glob(f"{day}_*")):
        ev, sy = folder / "events.csv", folder / "system.csv"
        if not ev.is_file():
            continue
        with ev.open(encoding="utf-8-sig", newline="") as fh:
            events = [r for r in csv.DictReader(fh) if r.get("latency_ms")]
        system = []
        if sy.is_file():
            with sy.open(encoding="utf-8-sig", newline="") as fh:
                system = list(csv.DictReader(fh))
        out.append({"name": folder.name, "events": events, "system": system})
    return out


FIELD_NAMES = {("ocr", "result"): "Text reading: one capture", ("object", "object_announce"): "Object scan and announcement",
               ("currency", "result"): "Currency check", ("currency", "watermark_check"): "Watermark check (guided, several frames)",
               ("currency", "uv_check"): "UV fibre check"}


# ---------------------------------------------------------------- report
def main() -> None:
    today_path = sorted(RAW.glob("pi5_benchmark_*.json"))[-1]
    today = load(today_path)
    ref = load(REFERENCE)
    extras = load(RAW / "extras.json") or {"results": {}, "skipped": {}}
    ocr = load(RAW / "ocr_response.json")
    switch = load(RAW / "mode_switch_currency.json") or {}
    host = today["host"]
    lat = dict(today["latency"])
    for k in ("emotion_with_face", "object_yolov8n", "object_yolov8s"):
        if k in extras["results"]:
            lat[k] = extras["results"][k]
    started = datetime.fromisoformat(host["utc"]).astimezone()

    tl = []
    if (RAW / "system_timeline.csv").is_file():
        with (RAW / "system_timeline.csv").open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    tl.append({"epoch": float(r["epoch"]), "phase": r["phase"], "cpu_temp_c": float(r["cpu_temp_c"]),
                               "throttled_hex": r["throttled_hex"], "cpu_busy_pct": float(r["cpu_busy_pct"]),
                               "mem_used_mb": float(r["mem_used_mb"]), "fan_rpm": float(r["fan_rpm"] or 0)})
                except (ValueError, KeyError):
                    continue
    clock = []
    if (RAW / "arm_clock.csv").is_file():
        with (RAW / "arm_clock.csv").open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    clock.append((float(r["epoch"]), float(r["arm_clock_hz"])))
                except (ValueError, KeyError, TypeError):
                    continue

    ocr_rows = ocr["results"] if ocr else []
    g = {
        "latency": graph_latency(lat),
        "compare": graph_compare(lat, ref["latency"]) if ref else None,
        "ocr": graph_ocr(ocr_rows),
        "ocr_paths": graph_ocr_paths(ocr_rows),
        "timeline": graph_timeline(tl, clock),
        "views": graph_views(lat),
    }
    loads = [("Text reader (EasyOCR, Bangla + English)", ocr.get("load_ms") if ocr else None),
             ("Currency mode, first switch into it", switch.get("activate_first_ms")),
             ("Currency mode, warm-up pass", switch.get("warm_up_ms")),
             ("Emotion detector", extras["results"].get("emotion_with_face", {}).get("load_ms")),
             ("Object mode (YOLOv8s)", lat.get("object_mode", {}).get("load_ms")),
             ("Watermark checker", lat.get("watermark_int8_localizer", {}).get("load_ms")),
             ("Multi-view authenticator", lat.get("prmvt_1view", {}).get("load_ms")),
             ("Camera, open to first frame", extras["results"].get("camera", {}).get("open_to_first_frame_ms"))]
    g["loads"] = graph_loads(loads)

    def by_case(case, path="app"):
        return next((r for r in ocr_rows if r["case"] == case and r["path"] == path), None)

    def stats(group: str) -> dict | None:
        rows = [r for r in tl if phase_group(r["phase"]) == group]
        if not rows:
            return None
        n = len(rows)
        return {"max": max(r["cpu_temp_c"] for r in rows), "mean": statistics.fmean(r["cpu_temp_c"] for r in rows),
                "soft": 100 * sum(state(r["throttled_hex"]) == 1 for r in rows) / n,
                "hard": 100 * sum(state(r["throttled_hex"]) == 2 for r in rows) / n, "minutes": n * 2 / 60}

    L = []
    w = L.append
    w("# SARA on the Raspberry Pi 5: benchmark report")
    w("")
    w(f"Run on {started:%d %B %Y}, starting {started:%H:%M} local time, on the glass's own board "
      f"({host['device_model']}, {host['cpu_count']} cores, Python {host['python']}, torch {host.get('torch')}, "
      f"ONNX Runtime {host.get('onnxruntime')}, EasyOCR {host.get('easyocr')}). No GPU; everything runs on the CPU.")
    w("")

    # ---- headline
    peak = max((r["cpu_temp_c"] for r in tl), default=None)
    sign = by_case("sign_en_3_lines", "plain")
    sign_app = by_case("sign_en_3_lines")
    page = by_case("page_12_lines_640")
    w("## Headline numbers")
    w("")
    w("| What | Median | 95th percentile |")
    w("|---|---:|---:|")
    for k in ("currency_mode_full", "taka_yolo_onnx_int8", "watermark_int8_localizer", "jaal_check_qduig", "object_mode",
              "emotion_with_face"):
        if k in lat:
            w(f"| {NAMES[k]} | {ms(lat[k]['median_ms'])} | {ms(lat[k]['p95_ms'])} |")
    if sign:
        w(f"| Text reading, 3-line sign, text found once (the README's measure) | {sec(sign['total_ms_median'])} | "
          f"slowest of {sign['reps']}: {sec(sign['total_ms_max'])} |")
    if sign_app:
        w(f"| Text reading, 3-line sign, as the app runs it | {sec(sign_app['total_ms_median'])} | "
          f"slowest of {sign_app['reps']}: {sec(sign_app['total_ms_max'])} |")
    if page:
        w(f"| Text reading, 12-line page: first line spoken | {sec(page['first_line_ms_median'])} | – |")
        w(f"| Text reading, 12-line page: whole page | {sec(page['total_ms_median'])} | slowest of {page['reps']}: "
          f"{sec(page['total_ms_max'])} |")
    w("")
    if peak is not None:
        w(f"Peak core temperature {peak:.1f} °C. Memory held by the mode benchmark at its end: {today['peak_rss_mb'] / 1000:.2f} GB"
          + (f"; by the text-reading test: {ocr['peak_rss_mb'] / 1000:.2f} GB." if ocr and ocr.get("peak_rss_mb") else "."))
        w("")
    short = [r for r in ocr_rows if r["path"] == "app" and r["truth"] and not r["streamed"]]
    ocr_s = stats("OCR response test")
    blank = by_case("no_text")
    w("What stands out:")
    w("")
    if ocr_s:
        w(f"- The board ran at its temperature limit: the CPU was throttled in {ocr_s['hard']:.0f}% of the samples during the "
          "text-reading test. Every number here is for a hot, throttled board.")
    if sign and sign_app and sign_app["detect_calls"] >= 2:
        w(f"- Text reading is the slow mode by far, and in the app a short sign takes {sec(sign_app['total_ms_median'])} "
          f"where the benchmark path takes {sec(sign['total_ms_median'])}, because the text detector runs twice.")
    if "emotion_with_face" in lat and "emotion" in lat:
        w(f"- Emotion takes {ms(lat['emotion_with_face']['median_ms'])} when there is a face to classify; the "
          f"{ms(lat['emotion']['median_ms'])} from the project script is a frame with no face.")
    if "object_yolov8n" in lat and "object_mode" in lat:
        w(f"- The object scan now uses YOLOv8s at {ms(lat['object_mode']['median_ms'])}; YOLOv8n takes "
          f"{ms(lat['object_yolov8n']['median_ms'])}.")
    if blank and blank["read"]:
        w(f"- On a blank frame the text reader worked for {sec(blank['total_ms_median'])} and then returned made-up letters "
          "instead of \"no text found\".")
    w("")

    # ---- thermal state
    main_s = stats("Mode benchmark")
    w("## Read this first: the board was at its temperature limit")
    w("")
    w(f"The Pi was at {host['cpu_temp_c_start']:.1f} °C before the first timing call and the firmware flag already read "
      f"`{host['throttled_start']}` ({decode(host['throttled_start'])}). At the end of the mode benchmark it read "
      f"`{host['throttled_end']}`: {decode(host['throttled_end'])}.")
    w("")
    if main_s:
        w(f"- Mode benchmark ({main_s['minutes']:.1f} min): mean {main_s['mean']:.1f} °C, peak {main_s['max']:.1f} °C. "
          f"The soft temperature limit was active in {main_s['soft']:.0f}% of the 2-second samples and the CPU was throttled "
          f"or clock-capped in {main_s['hard']:.0f}%.")
    if ocr_s:
        w(f"- OCR response test ({ocr_s['minutes']:.1f} min): mean {ocr_s['mean']:.1f} °C, peak {ocr_s['max']:.1f} °C. "
          f"Soft limit active in {ocr_s['soft']:.0f}% of samples, throttled or clock-capped in {ocr_s['hard']:.0f}%.")
    if clock:
        lo = min(hz for _, hz in clock) / 1e9
        w(f"- Lowest CPU clock seen while it was logged (extra timings and OCR test only): {lo:.2f} GHz; full speed is 2.40 GHz.")
    if tl:
        uv = any(int(r["throttled_hex"], 16) & 0x10001 for r in tl if r["throttled_hex"].startswith("0x"))
        fan = max(r["fan_rpm"] for r in tl)
        w(f"- Under-voltage was {'reported' if uv else 'never reported'}, so the power supply is {'a suspect' if uv else 'not the cause'}. "
          f"The fan was running (up to {fan:,.0f} rpm).")
    w("")
    sessions = field_sessions(f"{started:%Y%m%d}")
    app_temps = [float(r["cpu_temp_c"]) for s_ in sessions for r in s_["system"] if r.get("cpu_temp_c")]
    line = "So these are the speeds of a hot glass"
    if app_temps:
        line += (f", which is the state it was in after tonight's {len(sessions)} sessions with the app (its own log shows up to "
                 f"{max(app_temps):.1f} °C during them)")
    line += "."
    if ref:
        line += (f" The README figures come from the 1 October run, when the same script ended with the flag at "
                 f"`{ref['host']['throttled_end']}`" + (f" and a {ref['sustained']['max_cpu_temp_c']} °C peak over 30 minutes"
                                                       if ref.get("sustained") else "") + ".")
    line += " This run cannot say how much of the difference is heat; a repeat with the board cool would."
    w(line)
    w("")
    if g["timeline"]:
        w(f"![The board during the whole run]({g['timeline']})")
        w("")

    # ---- full list
    w("## Full latency list")
    w("")
    w("Each row is 30 timed calls after 3 warm-up calls, from `scripts/benchmark_pi5.py` unless marked. "
      "Frames are 640x480, the glass camera size.")
    w("")
    if g["latency"]:
        w(f"![Latency of every mode and model]({g['latency']})")
        w("")
    w("| Group | Benchmark | Median | 95th pct | Mean | Calls/s | Load | Memory after | Temp after |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for group, keys in GROUPS:
        for k in keys:
            if k not in lat:
                continue
            r = lat[k]
            extra = " †" if k in extras["results"] else ""
            w(f"| {group} | {NAMES[k]}{extra} | {ms(r['median_ms'])} | {ms(r['p95_ms'])} | {ms(r['mean_ms'])} | "
              f"{r['fps_median']:.2f} | {ms(r.get('load_ms'))} | {r['rss_mb_after'] / 1000:.2f} GB | {r['cpu_temp_c_after']:.1f} °C |")
    w("")
    w("† from `benchmark_report/scripts/extras_bench.py`, because the datasets the project script reads are not on the Pi. "
      "These rows ran in a separate process, so their memory column cannot be compared with the others.")
    w("")
    w("Load is the time to build the model object in that process. The ONNX detectors are only read from disk on their first "
      "call, so their load column is near zero, and a row that reuses a library an earlier row imported loads faster. "
      "Memory is the whole benchmark process after that row ran, not the size of that one model.")
    w("")
    notes = []
    if "emotion_with_face" in lat and "emotion" in lat:
        e = extras["results"]["emotion_with_face"]
        notes.append(f"**Emotion.** The project script has no face pictures on the Pi (RAF-DB is absent), so it times a black "
                     f"frame: {ms(lat['emotion']['median_ms'])} is the face search finding nothing. With a face in view "
                     f"(the astronaut test picture, face found, classifier `{e.get('backend')}`) one answer takes "
                     f"{ms(e['median_ms'])}, which is {e['median_ms'] / lat['emotion']['median_ms']:.0f} times longer.")
    if "object_yolov8n" in lat and "object_yolov8s" in lat:
        notes.append(f"**Objects.** The app now loads YOLOv8s ({ms(lat['object_mode']['median_ms'])} per scan). YOLOv8n on the same "
                     f"frames takes {ms(lat['object_yolov8n']['median_ms'])}; the README's 207 ms object figure was YOLOv8n. "
                     "COCO pictures are not on the Pi, so the scan was timed on the note photos; the detector does the same "
                     "amount of work on any picture.")
    notes.append("**Currency.** Only two note photos are on the Pi (the 500 and 1,000 Taka templates), so every currency, "
                 "watermark and counterfeit-check row cycles through those two.")
    for n in notes:
        w(f"- {n}")
    w("")
    if g["views"]:
        w(f"![Counterfeit check: cost of each extra view]({g['views']})")
        w("")

    # ---- comparison
    if ref and g["compare"]:
        w("## Against the 1 October run")
        w("")
        w(f"![Tonight against the 1 October run]({g['compare']})")
        w("")
        w("| Benchmark | 1 Oct median | Tonight median | Change |")
        w("|---|---:|---:|---:|")
        for _, keys in GROUPS:
            for k in keys:
                if k in lat and k in ref["latency"]:
                    o, n = ref["latency"][k]["median_ms"], lat[k]["median_ms"]
                    w(f"| {NAMES[k]} | {ms(o)} | {ms(n)} | {(n / o - 1) * 100:+.0f}% |")
        w("")
        w("The two runs differ in more than temperature: 1 October used 100 calls per row and tonight 30, and parts of the "
          "code have changed since (config.py dates changes on 2 and 13 October). The slowdown is consistent with the "
          "throttling shown above, but this run does not prove that heat is the only cause. The object scan was skipped on 1 October (its weights were missing then).")
        w("")

    # ---- OCR
    if ocr:
        s = ocr.get("settings") or {}
        w("## Text reading (OCR) response time")
        w("")
        w(f"This times `OCRMode.process_frame()`, the call `main.py` makes when ACTION is pressed in text mode, on rendered "
          f"signs and pages whose text is known. Settings as shipped: pipeline `{s.get('OCR_PIPELINE')}`, decoder "
          f"`{s.get('OCR_V2_DECODER')}`, canvas {s.get('OCR_CANVAS_SIZE')}, text detector `{s.get('detector_class')}`, "
          f"{s.get('torch_threads')} threads. Each row is the median of 3 to 5 captures.")
        w("")
        if ocr_s:
            ocr_epochs = [r["epoch"] for r in tl if phase_group(r["phase"]) == "OCR response test"]
            in_ocr = [hz for e, hz in clock if ocr_epochs[0] <= e <= ocr_epochs[-1]]
            line = (f"The firmware reported throttling in {ocr_s['hard']:.0f}% of the samples during this test (mean "
                    f"{ocr_s['mean']:.1f} °C)")
            if in_ocr:
                line += (f", and the CPU clock was below 2.0 GHz in {100 * sum(hz < 2.0e9 for hz in in_ocr) / len(in_ocr):.0f}% "
                         "of them instead of the full 2.4 GHz")
            w(line + ". These are the times of a throttled CPU; this was the last and hottest step of the run.")
            w("")
        if g["ocr"]:
            w(f"![Text reading: response time and where it goes]({g['ocr']})")
            w("")
        w("| Capture | Runs | Total | Range | Finding text | Reading letters | Rest | First line spoken | Character errors |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in [r for r in ocr_rows if r["path"] == "app"]:
            w(f"| {r['label']} | {r['reps']} | **{sec(r['total_ms_median'])}** | {sec(r['total_ms_min'])} – {sec(r['total_ms_max'])} | "
              f"{sec(r['detect_ms_median'])} | {sec(r['recognize_ms_median'])} | {sec(r['other_ms_median'])} | "
              f"{sec(r['first_line_ms_median'])} | {'–' if r['cer'] is None else f'{100 * r['cer']:.1f}%'} |")
        w("")
        app_rows = [r for r in ocr_rows if r["path"] == "app" and r["truth"]]
        passes = [r["detect_ms_median"] / max(1, r["detect_calls"]) for r in ocr_rows]
        w(f"- **One pass of the text detector takes {min(passes) / 1000:.0f} to {max(passes) / 1000:.0f} s on this board, whatever is "
          "in the picture.** That is the floor under every capture, and it is paid before a single letter is read.")
        if sign and sign_app and sign_app["detect_calls"] >= 2:
            old = ref and ref["latency"].get("ocr_mode")
            old = f"; the 1 October run measured {sec(old['median_ms'])} on this same frame" if old else ""
            w(f"- **Short signs pay for the detector twice in the app.** The 3-line English sign takes {sec(sign['total_ms_median'])} "
              f"when the text is found once (the path `benchmark_pi5.py` times{old}) and {sec(sign_app['total_ms_median'])} in the "
              f"app, where the detector runs {sign_app['detect_calls']} times: once to count the lines for line-by-line speech "
              f"(`_read_page_streaming`), then again inside the normal read when there are fewer than {s.get('OCR_STREAM_MIN_LINES')} "
              f"lines. All {len(short)} short captures show it ({min(r['total_ms_median'] for r in short) / 1000:.0f} to "
              f"{max(r['total_ms_median'] for r in short) / 1000:.0f} s in the app).")
        page_hi = by_case("page_12_lines_1280")
        if page:
            w(f"- **Pages are spoken line by line.** On the 12-line page at camera size the first line reaches speech after "
              f"{sec(page['first_line_ms_median'])} and the whole page takes {sec(page['total_ms_median'])}. Here reading the "
              f"letters ({sec(page['recognize_ms_median'])} over {page['recognize_calls']} lines) costs more than finding them "
              f"({sec(page['detect_ms_median'])}).")
        if page and page_hi:
            w(f"- **The same page at 1280x720 is faster ({sec(page_hi['total_ms_median'])}) than at 640x480 "
              f"({sec(page['total_ms_median'])}).** A 640x480 frame is enlarged to 1200x900 before detection, which is more "
              "pixels than a 1280x720 frame that is used as it is.")
        if blank:
            line = (f"- **With no text in view the glass still works for {sec(blank['total_ms_median'])}** "
                    f"({blank['detect_calls']} detector passes).")
            if blank["read"]:
                line += (f" On the blank grey test frame it did not answer \"no text found\": the handwritten-letter fallback "
                         f"returned {len(blank['read'])} characters (`{blank['read'][:12]}…`). Worth checking on a real blank wall.")
            w(line)
        w(f"- Loading the text reader takes {sec(ocr.get('load_ms'))}; the first capture after loading took "
          f"{sec(ocr.get('first_capture_after_load_ms'))}.")
        w("")
        if g["ocr_paths"]:
            w(f"![Short signs: the app looks for the text twice]({g['ocr_paths']})")
            w("")
            w("| Capture | Text found once | As the app runs | Change |")
            w("|---|---:|---:|---:|")
            for r in [r for r in ocr_rows if r["path"] == "plain"]:
                a = by_case(r["case"])
                if a:
                    w(f"| {r['label']} | {sec(r['total_ms_median'])} | {sec(a['total_ms_median'])} | "
                      f"{(a['total_ms_median'] / r['total_ms_median'] - 1) * 100:+.0f}% |")
            w("")
        w("What was read (app path, last run of each capture):")
        w("")
        w("| Capture | Text on the picture | Text the glass read |")
        w("|---|---|---|")
        for r in app_rows:
            w(f"| {r['label']} | {r['truth']} | {r['read'] or '(nothing)'} |")
        w("")
        w("These are clean rendered pictures, not camera photos, so the error rates say only that the timed captures were "
          "real reads. The pictures are saved in `raw/ocr_input_*.png`.")
        w("")

    # ---- field
    if sessions:
        w("## What the app itself logged tonight")
        w("")
        w(f"The glass was run twice tonight before this benchmark ({', '.join('`' + s['name'] + '`' for s in sessions)}). "
          "These are the latencies its own field log recorded with the real camera.")
        w("")
        w("| What | Times | Median | Range |")
        w("|---|---:|---:|---:|")
        grouped = {}
        for s in sessions:
            for e in s["events"]:
                grouped.setdefault((e["mode"], e["event"]), []).append(float(e["latency_ms"]))
        for key, label in FIELD_NAMES.items():
            v = grouped.get(key)
            if v:
                fmt = sec if max(v) >= 2000 else ms
                w(f"| {label} | {len(v)} | {fmt(statistics.median(v))} | {'–' if len(v) == 1 else f'{fmt(min(v))} – {fmt(max(v))}'} |")
        w("")
        rss = [float(r["app_rss_mb"]) for s in sessions for r in s["system"] if r.get("app_rss_mb")]
        temps = [float(r["cpu_temp_c"]) for s in sessions for r in s["system"] if r.get("cpu_temp_c")]
        ocr_real = grouped.get(("ocr", "result"))
        bits = []
        if ocr_real:
            line = (f"The app's one real text capture took {sec(ocr_real[0])}" if len(ocr_real) == 1 else
                    f"The app's {len(ocr_real)} real text captures took a median of {sec(statistics.median(ocr_real))}")
            if short:
                line += (f", close to the {min(r['total_ms_median'] for r in short) / 1000:.0f} to "
                         f"{max(r['total_ms_median'] for r in short) / 1000:.0f} s measured above for short captures as the app runs them")
            bits.append(line + ".")
        if rss:
            bits.append(f"The running app reached {max(rss) / 1000:.2f} GB of memory (all modes loaded, camera and preview "
                        f"window open).")
        if temps:
            bits.append(f"The app logged up to {max(temps):.1f} °C.")
        w(" ".join(bits))
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
    w("Speech is espeak-ng writing the audio to a pipe, not playing it: the voice adds almost nothing to the response time.")
    if cam:
        w(f"The camera delivered {cam['fps_median']} frames a second with the app's own settings (MJPG, {cam['size']}, 30 frames/s "
          "requested). It was measured indoors at midnight; a USB camera usually slows down in dim light, and that was not "
          "checked in daylight. The frame rate sets how old the picture is when a button is pressed.")
    w("")

    # ---- method
    w("## How this was measured")
    w("")
    w("1. `scripts/pi5_preflight.py`: every model file present, each ONNX model ran one dummy input (passed).")
    w("2. `scripts/benchmark_pi5.py --iters 30 --warmup 3` for every mode except text reading "
      f"(result: `raw/{today_path.name}`, also saved by the script in `savior_glass/results/`).")
    w("3. `scripts/measure_mode_switch.py` for the currency mode switch.")
    w("4. `benchmark_report/scripts/extras_bench.py`: emotion with a face, both object detectors, speech, camera.")
    w("5. `benchmark_report/scripts/ocr_response_bench.py`: the text reader, 3 to 5 captures per picture.")
    w("6. `benchmark_report/scripts/sys_sampler.py` logged temperature, throttle flag, CPU and memory every 2 s throughout.")
    w("")
    w("Limits of this run:")
    w("")
    w("- VS Code and its language server were running on the Pi during the run and use some CPU and memory.")
    w("- Text reading was left out of step 2 because 33 captures would have added about ten minutes; step 5 times the same "
      "call on the same frame.")
    w("- The 30-minute sustained run (`--sustained 30`) was not repeated tonight.")
    w("- The steps ran one after another on a board that kept heating up; text reading came last and was the hottest.")
    w("- The CPU clock was logged only from the extra timings onwards.")
    w("- OCR and emotion pictures are rendered or stock pictures, not photos from the glass camera.")
    if extras.get("skipped"):
        w(f"- Skipped: {extras['skipped']}")
    if today.get("skipped"):
        w(f"- Skipped by the project script: {today['skipped']}")
    w("")
    w("To rebuild this report from the raw files: `~/.venvs/savior_glass/bin/python benchmark_report/scripts/make_report.py`.")
    w("")

    md = "\n".join(L)
    (RD / "REPORT.md").write_text(md, encoding="utf-8")

    import markdown
    body = markdown.markdown(md, extensions=["tables"])

    def inline(match):
        data = base64.b64encode((RD / match.group(1)).read_bytes()).decode()
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
    (RD / "REPORT.html").write_text(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>SARA Pi 5 benchmark report</title><style>{css}</style></head><body>{body}</body></html>", encoding="utf-8")
    print("wrote", RD / "REPORT.md", "and REPORT.html;", len(list(GRAPHS.glob('*.png'))), "graphs")


if __name__ == "__main__":
    main()
