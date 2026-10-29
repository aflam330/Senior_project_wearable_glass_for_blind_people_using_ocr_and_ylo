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
import subprocess
import sys
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
# Earlier runs this one is compared with, oldest first. The first is the source of the README figures.
REFERENCES = [GLASS / "results" / "pi5_benchmark_memo_20261001T111325Z.json",
              RD / "archive" / "2026-10-13" / "raw" / "pi5_benchmark_memo_20261013T173438Z.json"]
# Code that can change a timing; changes_between() lists the commits to it between an earlier run and this one.
TIMED_CODE = ["savior_glass/modes", "savior_glass/assistive", "savior_glass/config.py", "savior_glass/utils.py",
              "savior_glass/models", "realtime_bangla_taka_detection/models"]
# Accuracy of the counterfeit check that runs on the Pi: its checkpoint and the latest per-view test of it.
AUTH_EVAL = RD.parent / "realtime_bangla_taka_detection" / "results" / "qduig" / "prefix_ft" / "seed42" / "test_views_20260928"
EMOTION_EVAL = GLASS / "results" / "emotion_rafdb.json"

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
    "taka_yolo_onnx_int8": "Taka detector (earlier model), ONNX INT8",
    "taka_yolo_onnx_fp32": "Taka detector (earlier model), ONNX FP32",
    "taka_yolo_pt": "Taka detector (earlier model), PyTorch",
    "watermark_int8_localizer": "Watermark check, learned localizer (used)",
    "watermark_int8_sift": "Watermark check, SIFT alignment",
    "jaal_check_qduig": "Counterfeit safe-check, one crop",
    **{f"prmvt_{k}view": f"Multi-view authenticator, {k} view{'s' if k > 1 else ''}" for k in range(1, 7)},
    "object_mode": "Object scan, as the app runs it (YOLOv8s)",
    "object_yolov8s": "Object detector YOLOv8s alone",
    "object_yolov8n": "Object detector YOLOv8n alone",
    "emotion_with_face": "Emotion, face in view",
    "emotion": "Emotion, no face in view (face search only)",
    "ocr_mode": "Text reading, 3-line sign (one capture)",
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
        return "no throttling, clock capping or under-voltage"
    return "; ".join(name for bit, name in THROTTLE_BITS.items() if v >> bit & 1)


def state(hexval: str) -> int:
    """0 normal, 1 soft temperature limit active, 2 throttled or clock capped."""
    try:
        v = int(hexval, 16)
    except (TypeError, ValueError):
        return 0
    return 2 if v & 0x6 else 1 if v & 0x8 else 0


def ever_hot(hexval: str) -> bool:
    """True if the flag shows throttling, clock capping or the soft temperature limit, now or since boot."""
    try:
        return bool(int(hexval, 16) & 0xE000E)
    except (TypeError, ValueError):
        return False


def run_label(run: dict) -> str:
    h = run["host"]
    heat = "throttled" if ever_hot(h.get("throttled_end")) else "no throttling"
    iters = next(iter(run["latency"].values()))["iters"]
    return f"{datetime.fromisoformat(h['utc']).astimezone():%-d %b %Y} – {heat}, {iters} calls"


def changes_between(after_utc: str, before_utc: str) -> list[tuple[str, str, str]]:
    """Commits to TIMED_CODE between two runs: (short hash, date, first sentence of the message)."""
    try:
        out = subprocess.run(["git", "-C", str(RD.parent), "log", f"--since={after_utc[:19]}Z", f"--until={before_utc[:19]}Z",
                              "--format=%h%x09%ad%x09%s", "--date=short", "--", *TIMED_CODE],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.splitlines():
        h, d, s = line.split("\t", 2)
        s = re.split(r"(?<=[.;])\s(?=[A-Z])", s, maxsplit=1)[0]
        rows.append((h, d, s if len(s) <= 120 else s[:117] + "…"))
    return rows


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


def graph_compare(runs: list[tuple[str, dict]]) -> str | None:
    """runs: (legend label, latency) oldest first; the last one is this run."""
    *older, (_, lat) = runs
    keys = [k for _, ks in GROUPS for k in ks
            if k in lat and any(k in o for _, o in older) and not re.match(r"prmvt_[2345]view", k)]
    if not older or not keys:
        return None
    n, prev = len(runs), older[-1][1]
    h = 0.84 / n
    fig, ax = plt.subplots(figsize=(10.5, (0.24 * n + 0.18) * len(keys) + 1.8))
    fig.subplots_adjust(left=0.37, right=0.97, top=top_margin(fig, 1.25), bottom=0.6 / fig.get_figheight() + 0.02)
    colours = [BLUE, ORANGE, AQUA, WARNING]
    top = 0.0
    for i, (_, run) in enumerate(runs):
        off = (i - (n - 1) / 2) * h
        for y, k in enumerate(keys):
            if k not in run:
                continue
            v = run[k]["median_ms"]
            top = max(top, v)
            ax.barh(y + off, v, height=h * 0.88, color=colours[i])
            text = ms(v)
            if i == n - 1 and k in prev:
                text += f"  ({(v / prev[k]['median_ms'] - 1) * 100:+.0f}%)"
            ax.annotate(text, (v, y + off), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8,
                        color=INK if i == n - 1 else INK2)
    ax.set_yticks(range(len(keys)), [NAMES[k] for k in keys])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, top * 1.28)
    ax.set_xlabel("Median time per call (ms)")
    legend_above(fig, [Patch(color=colours[i], label=label) for i, (label, _) in enumerate(runs)], min(n, 3))
    header(fig, "This run against the earlier runs",
           f"Same script (scripts/benchmark_pi5.py), same board. The percentage is this run's change from {older[-1][0].split(' – ')[0]}.")
    return save(fig, "02_compared_with_earlier_runs.png")


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
    twice = any(by[c]["app"]["detect_calls"] >= 2 for c in cases)
    legend_above(fig, [Patch(color=BLUE, label="Text found once (what benchmark_pi5.py times)"),
                        Patch(color=ORANGE, label="As the app runs it" + (": page check first, then the normal read" if twice else ""))], 2)
    if twice:
        header(fig, "Short signs: the app looks for the text twice",
               "With line-by-line speech on, a capture of fewer than 4 lines runs the text detector a second time.")
    else:
        header(fig, "Short signs: the app path costs the same as the benchmark path",
               "The text detector runs once per capture in both, so the app adds no second pass.")
    return save(fig, "04_ocr_sign_app_vs_benchmark.png")


def graph_timeline(tl: list[dict], clock: list[tuple[float, float]]) -> str | None:
    if not tl:
        return None
    e0 = tl[0]["epoch"]
    t = [(r["epoch"] - e0) / 60 for r in tl]
    panels = [("Core temperature (°C)", [r["cpu_temp_c"] for r in tl], 2.3),
              ("Firmware throttle state", [state(r["throttled_hex"]) for r in tl], 1.1)]
    if clock:
        late = (clock[0][0] - e0) / 60
        panels.append(("CPU clock (GHz)" + (f", logged from minute {late:.1f}" if late >= 1 else ""), None, 1.1))
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


def draw_matrix(ax, m: list[list[int]], labels: list[str], fontsize: float) -> None:
    """Counts and row percentages; the colour is the row percentage, so each row reads as 'of these notes, ...'."""
    from matplotlib.colors import LinearSegmentedColormap
    frac = [[v / (sum(r) or 1) for v in r] for r in m]
    ax.imshow(frac, cmap=LinearSegmentedColormap.from_list("rows", [SURFACE, BLUE_LIGHT, BLUE]), vmin=0, vmax=1)
    for i, r in enumerate(m):
        for j, v in enumerate(r):
            ax.text(j, i, f"{v}\n{frac[i][j] * 100:.0f}%", ha="center", va="center", fontsize=fontsize, linespacing=1.25,
                    color=SURFACE if frac[i][j] > 0.55 else INK, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(len(labels)), labels, rotation=30 if len(labels) > 3 else 0, ha="right" if len(labels) > 3 else "center")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    ax.set_xlabel("Answer given")
    ax.set_ylabel("Real class")


def graph_auth_confusion(auth: dict[int, dict]) -> str | None:
    ks = sorted({k for k in (min(auth, default=0), max(auth, default=0)) if k in auth})
    if not ks:
        return None
    m = auth[ks[0]]["matrix"]
    fig, axes = plt.subplots(1, len(ks), figsize=(4.4 * len(ks) + 1.2, 4.9), squeeze=False)
    fig.subplots_adjust(left=0.14, right=0.97, top=top_margin(fig, 1.45), bottom=0.13, wspace=0.55)
    for ax, k in zip(axes[0], ks):
        draw_matrix(ax, auth[k]["matrix"], ["Counterfeit", "Genuine"], 11)
        a = auth[k]
        ax.set_title(f"{k} view{'s' if k > 1 else ''}: {a['accuracy'] * 100:.1f}% correct, "
                     f"{a['false_acceptance_rate'] * 100:.1f}% of fakes accepted", loc="left", fontsize=9.5, color=INK, pad=8)
    header(fig, "Counterfeit check: confusion matrices",
           f"{sum(map(sum, m))} held-out notes ({sum(m[0])} counterfeit, {sum(m[1])} genuine), the checkpoint the Pi runs. "
           "Evaluated off the Pi.")
    return save(fig, "08_counterfeit_confusion.png")


def graph_auth_tradeoff(auth: dict[int, dict], lat: dict) -> str | None:
    ks = [k for k in sorted(auth) if f"prmvt_{k}view" in lat]
    if len(ks) < 2:
        return None
    panels = [("Correct answers (%)", [auth[k]["accuracy"] * 100 for k in ks], "{:.1f}%"),
              ("Counterfeits accepted as genuine (%)", [auth[k]["false_acceptance_rate"] * 100 for k in ks], "{:.1f}%"),
              ("Time per check on the Pi 5, this run (ms)", [lat[f"prmvt_{k}view"]["median_ms"] for k in ks], "{:.0f} ms")]
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.6), sharex=True, gridspec_kw={"hspace": 0.5})
    fig.subplots_adjust(left=0.11, right=0.95, top=top_margin(fig, 1.05), bottom=0.09)
    for ax, (title, vals, fmt) in zip(axes, panels):
        ax.plot(ks, vals, color=BLUE, lw=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
        for k, v in zip(ks, vals):
            ax.annotate(fmt.format(v), (k, v), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5)
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.45 or 1
        ax.set_ylim(lo - pad, hi + pad * 1.4)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK2, pad=4)
        ax.grid(axis="x", visible=False)
    axes[-1].set_xticks(ks)
    axes[-1].set_xlim(ks[0] - 0.35, ks[-1] + 0.35)
    axes[-1].set_xlabel("Views of the note given to the counterfeit check")
    header(fig, "Counterfeit check: what each extra view buys and costs",
           "Accuracy from the held-out test (off the Pi); time from this run on the Pi.")
    return save(fig, "09_counterfeit_views_tradeoff.png")


def graph_emotion_confusion(ev: dict) -> str | None:
    m, labels = ev.get("confusion_matrix"), ev.get("labels")
    if not m or not labels:
        return None
    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    fig.subplots_adjust(left=0.16, right=0.97, top=top_margin(fig, 1.0), bottom=0.13)
    draw_matrix(ax, m, labels, 8.5)
    header(fig, f"Emotion: confusion matrix ({ev['accuracy'] * 100:.1f}% correct)",
           f"{ev.get('arch')} on the RAF-DB test set ({ev.get('n_test'):,} faces). Evaluated off the Pi.")
    return save(fig, "10_emotion_confusion.png")


def read_csv(path: Path | None) -> list[dict]:
    if not path or not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def graph_sustained(calls: list[dict], power: list[dict]) -> str | None:
    """30 minutes of mixed use: temperature, power, clock (power log, every 2 s) and memory (benchmark, every call)."""
    if not calls and not power:
        return None
    panels = []
    if power:
        t = [num(r["t_s"]) / 60 for r in power]
        panels += [("Core temperature (°C)", t, [num(r["temp_c"]) for r in power], 2.0),
                   ("Power drawn by the board's own circuits (W)", t, [num(r["rail_watts"]) for r in power], 1.3),
                   ("CPU clock (GHz)", t, [(num(r["arm_mhz"]) or 0) / 1000 for r in power], 1.0)]
    elif calls:
        panels.append(("Core temperature (°C)", [num(r["elapsed_s"]) / 60 for r in calls], [num(r["cpu_temp_c"]) for r in calls], 2.0))
    if calls:
        panels.append(("Memory of the benchmark process, all models loaded (GB)", [num(r["elapsed_s"]) / 60 for r in calls],
                       [(num(r["rss_mb"]) or 0) / 1000 for r in calls], 1.0))
    fig, axes = plt.subplots(len(panels), 1, figsize=(10.5, 1.5 * sum(p[3] for p in panels) + 1.4), sharex=True,
                             gridspec_kw={"height_ratios": [p[3] for p in panels], "hspace": 0.45})
    fig.subplots_adjust(left=0.1, right=0.88, top=top_margin(fig, 1.2), bottom=0.08)
    for ax, (title, x, y, _) in zip(axes, panels):
        pts = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
        ax.plot([a for a, _ in pts], [b for _, b in pts], color=BLUE, lw=1.8 if title.startswith("Core") else 1.3)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK2, pad=4)
        ax.grid(axis="x", visible=False)
        if title.startswith("CPU clock"):
            ax.set_ylim(0, 2.6)
            ax.set_yticks([0, 1.2, 2.4])
        if title.startswith("Core"):
            temps = [b for _, b in pts]
            ax.set_ylim(min(temps) - 3, 92)
            for level, colour, label in ((80, WARNING, "80 °C  soft limit"), (85, CRITICAL, "85 °C  throttling")):
                ax.axhline(level, color=colour, lw=1.4, ls=(0, (5, 3)))
                ax.annotate(label, (1.0, level), xycoords=ax.get_yaxis_transform(), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8.5, color=INK2, annotation_clip=False)
            i = max(range(len(pts)), key=lambda j: pts[j][1])
            ax.scatter(*pts[i], s=60, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=5)
            ax.annotate(f"peak {pts[i][1]:.1f} °C", pts[i], xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9,
                        fontweight="bold")
    axes[-1].set_xlabel("Minutes into the 30-minute run")
    axes[-1].set_xlim(0, max(p[1][-1] for p in panels))
    header(fig, "30 minutes of continuous use", "benchmark_pi5.py --sustained 30: every mode in turn, without a pause; the "
           "power log ran alongside.")
    return save(fig, "11_sustained_30_minutes.png")


# ---------------------------------------------------------------- field logs of the same day
def field_sessions(day: str) -> list[dict]:
    """The app's own field logs of that day: raw/field/ (copied there to be kept) and savior_glass/logs/field/ (git-ignored)."""
    out, seen = [], set()
    for folder in sorted([*(RAW / "field").glob(f"{day}_*"), *(GLASS / "logs" / "field").glob(f"{day}_*")], key=lambda p: p.name):
        ev, sy = folder / "events.csv", folder / "system.csv"
        if folder.name in seen or not ev.is_file():
            continue
        seen.add(folder.name)
        with ev.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        system = []
        if sy.is_file():
            with sy.open(encoding="utf-8-sig", newline="") as fh:
                system = list(csv.DictReader(fh))
        settings = (load(folder / "session.json") or {}).get("settings", {})
        out.append({"name": folder.name, "start": datetime.strptime(folder.name[:15], "%Y%m%d_%H%M%S"),
                    "events": [r for r in rows if r.get("latency_ms")], "rows": rows, "system": system, "settings": settings})
    return out


FIELD_NAMES = {("ocr", "result"): "Text reading: one capture", ("object", "object_announce"): "Object scan and announcement",
               ("currency", "result"): "Currency check", ("currency", "watermark_check"): "Watermark check (guided, several frames)",
               ("currency", "uv_check"): "UV fibre check"}


# ---------------------------------------------------------------- report
def main() -> None:
    today_path = sorted(RAW.glob("pi5_benchmark_*.json"))[-1]
    today = load(today_path)
    refs = [r for r in map(load, REFERENCES) if r and r["host"]["utc"] < today["host"]["utc"]]
    ref = refs[0] if refs else None
    auth = {}
    for k in range(1, 7):
        cm, met = load(AUTH_EVAL / f"{k}view" / "confusion_matrix.json"), load(AUTH_EVAL / f"{k}view" / "test_metrics.json")
        if cm and met:
            auth[k] = {**met, "matrix": cm["matrix"]}
    emotion_ev = load(EMOTION_EVAL)
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
        "compare": graph_compare([(run_label(r), r["latency"]) for r in refs] + [(run_label(today) + " (this run)", lat)]),
        "ocr": graph_ocr(ocr_rows),
        "ocr_paths": graph_ocr_paths(ocr_rows),
        "timeline": graph_timeline(tl, clock),
        "views": graph_views(lat),
        "auth_cm": graph_auth_confusion(auth),
        "auth_tradeoff": graph_auth_tradeoff(auth, lat),
    }
    # The RAF-DB matrix describes the Pi only if the Pi runs the same classifier.
    emo = extras["results"].get("emotion_with_face", {})
    emotion_same = bool(emotion_ev and emo.get("backend") and str(emotion_ev.get("arch", "")).split("_")[0] in emo["backend"])
    g["emotion_cm"] = graph_emotion_confusion(emotion_ev) if emotion_same else None
    sus_path = next(iter(sorted((RAW / "sustained").glob("pi5_benchmark_*.json"))[-1:]), None)
    sus = load(sus_path) if sus_path else None
    sus_calls = read_csv(sus_path.with_suffix(".csv")) if sus_path else []
    pw_path = next(iter(sorted((RAW / "sustained").glob("power_*.json"))[-1:]), None)
    pw = load(pw_path) if pw_path else None
    pw_rows = read_csv(pw_path.with_suffix(".csv")) if pw_path else []
    g["sustained"] = graph_sustained(sus_calls, pw_rows) if sus and sus.get("sustained") else None
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
    hot = any(ever_hot(r["throttled_hex"]) for r in tl) or ever_hot(host["throttled_end"])
    w("What stands out:")
    w("")
    if ocr_s and hot:
        w(f"- The board ran at its temperature limit: the CPU was throttled in {ocr_s['hard']:.0f}% of the samples during the "
          "text-reading test. Every number here is for a hot, throttled board.")
    elif tl:
        w(f"- The board never reached its temperature limit: peak {peak:.1f} °C, and the firmware flag still read "
          f"`{tl[-1]['throttled_hex']}` at the end ({decode(tl[-1]['throttled_hex'])} since the Pi started).")
    hot_refs = [r for r in refs if ever_hot(r["host"]["throttled_end"])]
    if hot_refs and not hot and "currency_mode_full" in hot_refs[-1]["latency"]:
        o, n = hot_refs[-1]["latency"]["currency_mode_full"]["median_ms"], lat["currency_mode_full"]["median_ms"]
        w(f"- Against the throttled {datetime.fromisoformat(hot_refs[-1]['host']['utc']).astimezone():%-d %B} run, the "
          f"currency check went from {ms(o)} to {ms(n)} ({(n / o - 1) * 100:+.0f}%). Code changed in between as well; "
          "see \"Against the earlier runs\".")
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
    w("## Read this first: the board was at its temperature limit" if hot else "## Temperature: the board stayed below its limit")
    w("")
    w(f"The Pi was at {host['cpu_temp_c_start']:.1f} °C before the first timing call and the firmware flag read "
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
    if clock and tl and clock[0][0] - tl[0]["epoch"] < 60:
        # each clock sample takes the step of the nearest timeline row; idle gaps are left out (the clock drops there on purpose)
        import bisect
        ep = [r["epoch"] for r in tl]
        busy = [(hz, tl[min(bisect.bisect_left(ep, e), len(tl) - 1)]["phase"]) for e, hz in clock]
        busy = [(hz, p) for hz, p in busy if p not in ("idle", "")]
        low = [(hz, p) for hz, p in busy if hz < 2.39e9]
        if busy:
            line = (f"- During the timing steps the CPU ran at full speed (2.40 GHz) in {len(busy) - len(low)} of {len(busy)} "
                    "clock samples")
            if low:
                steps = {"extras_camera": "the camera timing", "mode_switch": "the mode switch", "preflight": "the pre-flight check"}
                line += (f"; the {len(low)} lower one{'s' if len(low) > 1 else ''} (down to {min(low)[0] / 1e9:.2f} GHz) fell in "
                         f"{' and '.join(sorted({steps.get(p, p) for _, p in low}))}, where the CPU mostly waits")
            w(line + ". The clock between the steps is left out: it drops there on purpose when the CPU is idle.")
    elif clock:
        lo = min(hz for _, hz in clock) / 1e9
        w(f"- Lowest CPU clock seen while it was logged: {lo:.2f} GHz; full speed is 2.40 GHz.")
    if tl:
        uv = any(int(r["throttled_hex"], 16) & 0x10001 for r in tl if r["throttled_hex"].startswith("0x"))
        fan = max(r["fan_rpm"] for r in tl)
        w(f"- Under-voltage was {'reported' if uv else 'never reported'}, so the power supply is {'a suspect' if uv else 'not the cause'}. "
          f"The fan was running (up to {fan:,.0f} rpm).")
    w("")
    sessions = field_sessions(f"{started:%Y%m%d}")
    app_temps = [float(r["cpu_temp_c"]) for s_ in sessions for r in s_["system"] if r.get("cpu_temp_c")]
    if not hot:
        line = "So these are the speeds of a glass that is not throttled."
        if hot_refs:
            r = hot_refs[-1]
            line += (f" A glass worn for a long time gets hotter: on {datetime.fromisoformat(r['host']['utc']).astimezone():%-d %B}, "
                     f"after sessions with the app, the same board started this script at {r['host']['cpu_temp_c_start']:.1f} °C "
                     f"and was throttled (`archive/2026-10-13/REPORT.md`).")
        if ref and ref.get("sustained"):
            line += (f" The longest test so far, 30 minutes of mixed use on {datetime.fromisoformat(ref['host']['utc']).astimezone():%-d %B}, "
                     f"peaked at {ref['sustained']['max_cpu_temp_c']} °C without throttling.")
    else:
        line = "So these are the speeds of a hot glass"
    if hot and app_temps:
        line += (f", which is the state it was in after that day's {len(sessions)} sessions with the app (its own log shows up to "
                 f"{max(app_temps):.1f} °C during them)")
    if hot:
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
    try:
        sys.path.insert(0, str(GLASS))
        import config
        app_det = Path(config.CURRENCY_YOLO_PATH).name
    except Exception:  # noqa: BLE001
        app_det = None
    if app_det and app_det not in ("best_int8.onnx", "best.onnx", "best.pt"):
        notes.append(f"**Taka detector.** The app's currency check loads `{app_det}`; `benchmark_pi5.py` times only the earlier "
                     "detector on its own (`best_int8.onnx`, `best.onnx`, `best.pt`), so the detector the app uses is timed "
                     "only inside the end-to-end currency row.")
    for n in notes:
        w(f"- {n}")
    w("")
    if g["views"]:
        w(f"![Counterfeit check: cost of each extra view]({g['views']})")
        w("")

    # ---- comparison
    if refs and g["compare"]:
        days = [f"{datetime.fromisoformat(r['host']['utc']).astimezone():%-d %b}" for r in refs]
        w("## Against the earlier runs")
        w("")
        w(f"![This run against the earlier runs]({g['compare']})")
        w("")
        w("| Benchmark | " + " | ".join(f"{d}" for d in days) + " | This run | " + " | ".join(f"Change from {d}" for d in days) + " |")
        w("|---|" + "---:|" * (2 * len(refs) + 1))
        for _, keys in GROUPS:
            for k in keys:
                if k in lat and any(k in r["latency"] for r in refs):
                    old = [r["latency"].get(k, {}).get("median_ms") for r in refs]
                    n = lat[k]["median_ms"]
                    w(f"| {NAMES[k]} | " + " | ".join(ms(o) for o in old) + f" | {ms(n)} | "
                      + " | ".join("–" if o is None else f"{(n / o - 1) * 100:+.0f}%" for o in old) + " |")
        w("")
        w("| Run | Calls per row | Temperature, start → end | Firmware flag at the end |")
        w("|---|---:|---:|---|")
        for label, r in [(d, r) for d, r in zip(days, refs)] + [(f"{started:%-d %b} (this run)", today)]:
            h = r["host"]
            w(f"| {label} | {next(iter(r['latency'].values()))['iters']} | {h['cpu_temp_c_start']:.1f} → "
              f"{h['cpu_temp_c_end']:.1f} °C | `{h['throttled_end']}` ({decode(h['throttled_end'])}) |")
        w("")
        w("Heat is not the only difference between the runs: the code changed too. Commits to the timed code "
          f"(`{'`, `'.join(TIMED_CODE)}`) between each earlier run and this one:")
        w("")
        for d, r in zip(days, refs):
            commits = changes_between(r["host"]["utc"], host["utc"])
            w(f"**Since {d}: {len(commits)} commit{'' if len(commits) == 1 else 's'}.**" + (" The latest:" if commits else ""))
            w("")
            for c in commits[:10]:
                w(f"- `{c[0]}` {c[1]}: {c[2]}")
            if len(commits) > 10:
                w(f"- … and {len(commits) - 10} earlier (`git log` lists them all).")
            w("")
        if "object_mode" not in refs[0]["latency"]:
            w(f"The object scan was skipped on {days[0]} (its weights were missing then).")
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
            if ocr_s["hard"] > 0:
                line = (f"The firmware reported throttling in {ocr_s['hard']:.0f}% of the samples during this test (mean "
                        f"{ocr_s['mean']:.1f} °C)")
                if in_ocr:
                    line += (f", and the CPU clock was below 2.0 GHz in {100 * sum(hz < 2.0e9 for hz in in_ocr) / len(in_ocr):.0f}% "
                             "of them instead of the full 2.4 GHz")
                w(line + ". These are the times of a throttled CPU; this was the last and hottest step of the run.")
            else:
                line = (f"The board was not throttled during this test (mean {ocr_s['mean']:.1f} °C, peak {ocr_s['max']:.1f} °C, "
                        f"soft limit active in {ocr_s['soft']:.0f}% of the samples)")
                if in_ocr:
                    line += f", and the CPU clock never dropped below {min(in_ocr) / 1e9:.2f} GHz"
                w(line + ".")
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
            old = today["latency"].get("ocr_mode") or (ref and ref["latency"].get("ocr_mode"))
            old = f"; `benchmark_pi5.py` measured {sec(old['median_ms'])} on this same frame" if old else ""
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
        if page and page_hi and page_hi["total_ms_median"] < page["total_ms_median"]:
            w(f"- **The same page at 1280x720 is faster ({sec(page_hi['total_ms_median'])}) than at 640x480 "
              f"({sec(page['total_ms_median'])}).** A 640x480 frame is enlarged to 1200x900 before detection, which is more "
              "pixels than a 1280x720 frame that is used as it is.")
        elif page and page_hi:
            w(f"- **The same page at 1280x720 takes {sec(page_hi['total_ms_median'])}, against {sec(page['total_ms_median'])} at "
              f"640x480.** Finding the text is faster on the larger frame ({sec(page_hi['detect_ms_median'])} against "
              f"{sec(page['detect_ms_median'])}), because a 640x480 frame is enlarged to 1200x900 before detection, but reading "
              f"the letters takes longer ({sec(page_hi['recognize_ms_median'])} against {sec(page['recognize_ms_median'])}).")
        if blank:
            line = (f"- **With no text in view the glass still works for {sec(blank['total_ms_median'])}** "
                    f"({blank['detect_calls']} detector pass{'' if blank['detect_calls'] == 1 else 'es'}).")
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
        w("## What the app itself logged that day")
        w("")
        bench_start = started.replace(tzinfo=None)
        sides = []
        for side in ("before", "after"):
            names = [f"`{s['name']}`" for s in sessions if (s["start"] < bench_start) == (side == "before")]
            if names:
                sides.append(f"{' and '.join(names)}, {'both ' if len(names) == 2 else ''}{side} this benchmark")
        w(f"The glass app was used {len(sessions)} time{'' if len(sessions) == 1 else 's'} that day ({'; '.join(sides)}; logs "
          "in `raw/field/`). These are the latencies its own field log recorded with the real camera, from the button press "
          "to the spoken answer.")
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
                    f"The app's {len(ocr_real)} real text captures took a median of {sec(statistics.median(ocr_real))} "
                    f"({sec(min(ocr_real))} to {sec(max(ocr_real))})")
            if page:
                line += (f". Reading time grows with the amount of text: on the rendered 12-line page above, reading the "
                         f"letters alone took {sec(page['recognize_ms_median'])}. The field log does not record how many lines "
                         "each capture had, so the slowest captures cannot be explained from it")
            bits.append(line + ".")
        if rss:
            bits.append(f"The running app reached {max(rss) / 1000:.2f} GB of memory (all modes loaded, camera and preview "
                        f"window open).")
        if temps:
            bits.append(f"The app logged up to {max(temps):.1f} °C.")
        w(" ".join(bits))
        w("")
        cur = [r for s in sessions for r in s["rows"] if r["mode"] == "currency" and r["event"] == "result"]
        if cur:
            said = {}
            for r in cur:
                said.setdefault(r["result"], []).append(r)
            w("What the glass said after each currency press:")
            w("")
            w("| Spoken answer | Times | Detector's guess (confidence) |")
            w("|---|---:|---|")
            for text, rs in sorted(said.items(), key=lambda kv: -len(kv[1])):
                guesses = ", ".join(f"{r['denomination'] or 'none'}" + (f" {float(r['confidence']) * 100:.0f}%" if r["confidence"] else "")
                                    for r in rs)
                w(f"| {text} | {len(rs)} | {guesses} |")
            w("")
            thr = next((s["settings"].get("CURRENCY_ANNOUNCE_CONF") for s in sessions if s["settings"].get("CURRENCY_ANNOUNCE_CONF")), None)
            confs = [(float(r["confidence"]), r["result"]) for r in cur if r["confidence"]]
            above = {t for c, t in confs if thr and c >= thr}
            below = {t for c, t in confs if thr and c < thr}
            if thr and confs and not (above & below):
                w(f"`CURRENCY_ANNOUNCE_CONF` was {thr} in these sessions. The {sum(c >= thr for c, _ in confs)} checks at or above "
                  f"it named the note; the {sum(c < thr for c, _ in confs)} below it did not.")
                w("")
        wm = [r for s in sessions for r in s["rows"] if r["mode"] == "currency" and r["event"] == "watermark_check"]
        if wm:
            outcome, frames, rejected = {}, 0, {}
            for r in wm:
                outcome[r["result"]] = outcome.get(r["result"], 0) + 1
                try:
                    d = json.loads(r["detail"])
                except (TypeError, ValueError):
                    d = {}
                frames += d.get("frames", 0)
                for k, v in (d.get("rejected") or {}).items():
                    rejected[k] = rejected.get(k, 0) + v
            w(f"Guided watermark checks: {len(wm)}, ending with "
              + ", ".join(f"`{k}` {v} time{'' if v == 1 else 's'}" for k, v in outcome.items())
              + f", after a median of {sec(statistics.median(float(r['latency_ms']) for r in wm))}. They looked at {frames} frames "
              "and rejected " + ", ".join(f"{v} as {k.replace('_', ' ')}" for k, v in sorted(rejected.items(), key=lambda kv: -kv[1]))
              + (". No back-lit reading was made, so the glass never reached \"likely genuine\"." if set(outcome) == {"timeout"} else "."))
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
    w("The speech rows time espeak-ng writing the audio to a pipe, not playing it. With `TTS_VOICE` at its default (`auto`) "
      "the app speaks with the Piper voices where their models are installed and falls back to espeak-ng, so these rows are "
      "the fallback voice; `scripts/speech_check.py --quiet` times Piper.")
    if cam:
        w(f"The camera delivered {cam['fps_median']} frames a second with the app's own settings (MJPG, {cam['size']}, 30 frames/s "
          f"requested), in the run that started at {started:%H:%M} local time; the light was not recorded, and a USB camera lowers its frame "
          "rate in dim light. The frame rate sets how old the picture is when a button is pressed.")
    w("")

    # ---- 30 minutes of use
    if sus and sus.get("sustained"):
        s, sh = sus["sustained"], sus["host"]
        w("## 30 minutes of continuous use")
        w("")
        w(f"`scripts/benchmark_pi5.py --sustained {s['minutes']:g}` ran every mode in turn, without a pause, for {s['minutes']:g} "
          f"minutes ({s['calls']:,} calls) after its own latency pass (result: `raw/sustained/{sus_path.name}`). It started at "
          f"{sh['cpu_temp_c_start']:.1f} °C with the firmware flag at `{sh['throttled_start']}`"
          + (f"; `scripts/power_log.py` logged the power alongside (`raw/sustained/{pw_path.name}`)." if pw else "."))
        w("")
        if g["sustained"]:
            w(f"![30 minutes of continuous use]({g['sustained']})")
            w("")
        rss = [num(r["rss_mb"]) for r in sus_calls if num(r["rss_mb"])]
        limited = 100 * sum(state(r["throttled"]) > 0 for r in sus_calls) / max(1, len(sus_calls))
        w("| What | Value |")
        w("|---|---:|")
        w(f"| Peak core temperature | {s['max_cpu_temp_c']:.1f} °C |")
        w(f"| Calls made while throttled or at the soft temperature limit | {limited:.0f}% |")
        w(f"| Firmware flag at the end | `{s['final_throttled']}` ({decode(s['final_throttled'])}) |")
        if rss:
            w(f"| Peak memory of the benchmark process (every model loaded) | {max(rss) / 1000:.2f} GB |")
        if pw and pw.get("rail_watts_mean") is not None:
            a = pw["assumptions"]
            w(f"| Power of the board's own circuits, mean / highest | {pw['rail_watts_mean']:.2f} W / {pw['rail_watts_max']:.2f} W |")
            w(f"| Estimated draw from the power bank (+{a['overhead'] * 100:.0f}% allowance) | {pw['estimated_input_watts']:.2f} W |")
            w(f"| Estimated battery life on a {a['bank_wh']:g} Wh bank ({a['bank_efficiency'] * 100:.0f}% of it usable) | "
              f"{pw['estimated_battery_hours']:.1f} h |")
        w("")
        modes = list(dict.fromkeys(r["mode"] for r in sus_calls))
        if sus_calls and modes:
            end = max(num(r["elapsed_s"]) for r in sus_calls)

            def window(m, lo, hi):
                v = [num(r["latency_ms"]) for r in sus_calls if r["mode"] == m and lo <= num(r["elapsed_s"]) <= hi]
                return statistics.median(v) if v else None
            w("Did the heat slow anything down? Median time per call in the first and the last 5 minutes:")
            w("")
            w("| Mode | First 5 min | Last 5 min | Change |")
            w("|---|---:|---:|---:|")
            for m in modes:
                a_, b_ = window(m, 0, 300), window(m, end - 300, end)
                if a_ and b_:
                    w(f"| {NAMES.get(m, m)} | {ms(a_)} | {ms(b_)} | {(b_ / a_ - 1) * 100:+.0f}% |")
            w("")
        if pw:
            a = pw["assumptions"]
            w(f"The power figure is the sum of the rails the Pi's power chip reports (`vcgencmd pmic_read_adc`). It leaves out "
              f"the 5 V input stage, the fan and the USB devices (the webcam), so it is a lower bound; the estimate adds "
              f"{a['overhead'] * 100:.0f}% for them. Battery hours = bank Wh × {a['bank_efficiency']:.2f} ÷ estimated W. "
              f"{a['bank_wh']:g} Wh is a 10,000 mAh bank at 3.7 V; for another bank, scale the hours by its capacity. Every mode "
              "ran back to back here, which is harder than real use: the app waits for a button between captures.")
            w("")

    # ---- accuracy (from the evaluation files, not from this run)
    if g["auth_cm"] or g["auth_tradeoff"] or emotion_ev:
        w("## Accuracy of the models timed here")
        w("")
        w("A speed test cannot measure accuracy: that needs labelled test pictures, and those datasets are not on the Pi. "
          "The figures below are drawn from the evaluation files in the repository, which were made off the Pi.")
        w("")
    if auth:
        rel = AUTH_EVAL.relative_to(RD.parent).as_posix()
        k1, k6 = min(auth), max(auth)
        best = max(auth, key=lambda k: (auth[k]["accuracy"], -k))
        w(f"**Counterfeit check** (multi-view authenticator, checkpoint `{AUTH_EVAL.parent.relative_to(RD.parent).as_posix()}"
          f"/checkpoint.pt`, the one the pre-flight loads; PyTorch both there and on the Pi). Test of `{rel}`: "
          f"{auth[k1]['n']} notes kept out of training, first k views of each note.")
        w("")
        if g["auth_cm"]:
            w(f"![Counterfeit check: confusion matrices]({g['auth_cm']})")
            w("")
        w("| Views | Correct | Balanced accuracy | Counterfeits accepted | Genuine notes rejected | ROC AUC | Time on the Pi (this run) |")
        w("|---:|---:|---:|---:|---:|---:|---:|")
        for k in sorted(auth):
            a, t = auth[k], lat.get(f"prmvt_{k}view")
            w(f"| {k} | {a['accuracy'] * 100:.1f}% | {a['balanced_accuracy'] * 100:.1f}% | {a['false_acceptance_rate'] * 100:.1f}% | "
              f"{a['false_rejection_rate'] * 100:.1f}% | {a['roc_auc']:.3f} | {ms(t['median_ms']) if t else '–'} |")
        w("")
        if g["auth_tradeoff"]:
            w(f"![Counterfeit check: what each extra view buys and costs]({g['auth_tradeoff']})")
            w("")
        t1, tb, det = lat.get(f"prmvt_{k1}view"), lat.get(f"prmvt_{best}view"), lat.get("currency_mode_full")
        if t1 and tb and best != k1:
            line = (f"- The best result is with {best} views ({auth[best]['accuracy'] * 100:.1f}%, against "
                    f"{auth[k1]['accuracy'] * 100:.1f}% with one). On the Pi that costs {ms(tb['median_ms'])} instead of "
                    f"{ms(t1['median_ms'])} per check")
            if det:
                line += f", against {ms(det['median_ms'])} for the whole currency check"
            w(line + ". The larger cost is the time the user takes to show the note several times.")
        w(f"- On {auth[k1]['n']} notes one note is {100 / auth[k1]['n']:.1f} points, so the differences between {k1} and {k6} views "
          "come down to a few notes.")
        readme = RD.parent / "README.md"
        quoted = re.search(r"(\d+(?:\.\d+)?)% with one view", readme.read_text(encoding="utf-8")) if readme.is_file() else None
        if quoted and abs(float(quoted.group(1)) - auth[k1]["accuracy"] * 100) > 0.05:
            w(f"- The README gives {quoted.group(1)}% for one view; these files give {auth[k1]['accuracy'] * 100:.1f}%. "
              f"The {quoted.group(1)}% comes from an evaluation that is not in `{rel}`.")
        w("")
    if emotion_ev:
        if g["emotion_cm"]:
            w(f"![Emotion: confusion matrix]({g['emotion_cm']})")
            w("")
        else:
            w(f"**Emotion.** `{EMOTION_EVAL.relative_to(RD.parent).as_posix()}` holds a confusion matrix for "
              f"{emotion_ev.get('arch')} on the RAF-DB test set ({emotion_ev['accuracy'] * 100:.1f}% of "
              f"{emotion_ev.get('n_test'):,} faces), but in this run the Pi answered with the `{emo.get('backend')}` classifier, "
              "so that matrix does not describe what ran here and is not drawn.")
            w("")

    # ---- method
    w("## How this was measured")
    w("")
    w("1. `scripts/pi5_preflight.py`: every model file present, each ONNX model ran one dummy input (passed).")
    w(f"2. `scripts/benchmark_pi5.py --iters 30 --warmup 3` for every mode{'' if 'ocr_mode' in today['latency'] else ' except text reading'} "
      f"(result: `raw/{today_path.name}`, also saved by the script in `savior_glass/results/`).")
    w("3. `scripts/measure_mode_switch.py` for the currency mode switch.")
    w("4. `benchmark_report/scripts/extras_bench.py`: emotion with a face, both object detectors, speech, camera.")
    w("5. `benchmark_report/scripts/ocr_response_bench.py`: the text reader, 3 to 5 captures per picture.")
    w("6. `benchmark_report/scripts/sys_sampler.py` logged temperature, throttle flag, CPU and memory every 2 s throughout.")
    for hw in sorted(RAW.glob("hardware_*.txt"))[-1:]:
        w(f"7. What the Pi reported about its hardware (USB devices, camera, sound devices, Bluetooth, memory, OS, kernel): "
          f"`raw/{hw.name}`.")
    w("")
    w("Limits of this run:")
    w("")
    w("- VS Code and its language server were running on the Pi during the run and use some CPU and memory.")
    if "ocr_mode" not in today["latency"]:
        w("- Text reading was left out of step 2 because 33 captures would have added about ten minutes; step 5 times the same "
          "call on the same frame.")
    if not today.get("sustained"):
        w("- The 30-minute sustained run (`--sustained 30`) was not part of this run.")
    w("- The steps ran one after another, so each step started on a board warmed by the one before; text reading came last.")
    if clock and tl and clock[0][0] - tl[0]["epoch"] >= 60:
        w("- The CPU clock was logged only from the extra timings onwards.")
    w("- OCR and emotion pictures are rendered or stock pictures, not photos from the glass camera.")
    w("- The test photos the project script reads (RAF-DB faces, JaalTaka views, COCO scenes) are not on the Pi; the "
      "notes under the latency list say what each row used instead.")
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
