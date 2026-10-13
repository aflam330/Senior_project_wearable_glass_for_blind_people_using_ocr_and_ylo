"""Figures from the result files of 2026-10-11 to 2026-10-13. Every bar is a number in a saved result file;
a figure whose file is missing is skipped, not drawn from memory.

    python scripts/make_new_figures.py
Output: <project>/paper_evidence/figures/fig4x_*.png
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
R = HERE.parent / "results"
OUT = HERE.parents[1] / "paper_evidence" / "figures"
# categorical slots 1-3 of the validated default palette, fixed order; text in ink, never in the series colour
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})


def load(rel):
    p = R / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", name)


def labels(ax, bars, fmt="{:.1f}", dy=0.0):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height() + dy), ha="center", va="bottom",
                    fontsize=9, color=INK, xytext=(0, 2), textcoords="offset points")


def uv_protocols():
    a, b = load("uv_check/uv_fibre_check.json"), load("uv_check/uv_blocked_cv.json")
    if not (a and b):
        return
    vals = ["200", "500", "1000"]
    first = [100 * a["test"]["per_value"][v]["accepted"] / a["test"]["per_value"][v]["n"] for v in vals] + [100 * a["test"]["accepted_rate"]]
    cv = b["A_one_threshold"]
    second = [100 * cv["per_value"][v]["rate"] for v in vals] + [100 * cv["rate"]]
    lo = [100 * cv["per_value"][v]["wilson95"][0] for v in vals] + [100 * cv["wilson95"][0]]
    hi = [100 * cv["per_value"][v]["wilson95"][1] for v in vals] + [100 * cv["wilson95"][1]]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    b1 = ax.bar(x - 0.19, first, 0.36, color=C1, label="First test: one split (threshold from front-side photos)")
    b2 = ax.bar(x + 0.19, second, 0.36, color=C2, label="Blocked 5-fold (threshold from the other blocks)")
    ax.errorbar(x + 0.19, second, yerr=[np.array(second) - lo, np.array(hi) - second], fmt="none", ecolor=INK, lw=1, capsize=3)
    labels(ax, b1, "{:.0f}%"); labels(ax, b2, "{:.0f}%", dy=6)
    ax.set_xticks(x, ["200 Taka", "500 Taka", "1,000 Taka", "All"])
    ax.set_ylim(0, 118); ax.set_ylabel("Genuine notes accepted (%)"); ax.grid(axis="x", visible=False)
    ax.set_title("UV fibre check: genuine notes recognised, two ways of testing", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
    save(fig, "fig41_uv_protocols.png")


def uv_counts():
    from modes.uv_check import fibre_mask, inner
    from scripts.uv_dataset import DATA, read
    if not (DATA / "boxes.csv").exists():
        return
    rows = [r for r in csv.DictReader(open(DATA / "boxes.csv", encoding="utf-8")) if r["x1"]]
    per = {"200": [], "500": [], "1000": []}
    for r in rows:
        img = read(DATA / r["value"] / r["file"])
        _, n, ab = fibre_mask(inner(img[int(r["y1"]):int(r["y2"]), int(r["x1"]):int(r["x2"])]))
        if ab is not None:
            per[r["value"]].append(n)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    rng = np.random.default_rng(0)
    for i, (v, c) in enumerate(zip(per, (C1, C2, C3))):
        y = np.array(per[v])
        ax.scatter(i + rng.uniform(-0.22, 0.22, len(y)), np.maximum(y, 0.8), s=22, color=c, edgecolor=SURFACE, linewidth=0.6)
        ax.hlines(np.median(y), i - 0.3, i + 0.3, color=INK, lw=1.5)
        ax.annotate(f"median {np.median(y):.0f}", (i + 0.32, np.median(y)), va="center", fontsize=9, color=INK)
    ax.axhline(6, color=MUTED, lw=1, ls="--")
    ax.annotate("threshold: 6 fibres", (-0.45, 6), va="bottom", fontsize=9, color=MUTED, xytext=(0, 2), textcoords="offset points")
    ax.set_yscale("log"); ax.set_xticks(range(3), ["200 Taka", "500 Taka", "1,000 Taka"]); ax.set_xlim(-0.5, 2.9)
    ax.set_ylabel("Fibres found in the note (log scale)"); ax.grid(axis="x", visible=False)
    ax.set_title("UV fibres found on 111 photos of genuine notes (plain surface: 0)", loc="left", fontsize=11)
    save(fig, "fig42_uv_fibre_counts.png")


def uv_conditions():
    a = load("uv_check/uv_fibre_check.json")
    if not a:
        return
    c = a["test_under_other_conditions"]
    names = list(c)
    vals = [100 * c[k]["rate"] for k in names]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    bars = ax.barh(range(len(names))[::-1], vals, 0.62, color=C1)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.0f}%", (v, b.get_y() + b.get_height() / 2), va="center", xytext=(4, 0), textcoords="offset points", fontsize=9)
    ax.set_yticks(range(len(names))[::-1], names); ax.set_xlim(0, 105); ax.set_xlabel("Genuine notes accepted (%), first test, 43 photos")
    ax.grid(axis="y", visible=False)
    ax.set_title("UV fibre check: blur hides fibres; brightness matters little", loc="left", fontsize=11)
    save(fig, "fig43_uv_conditions.png")


def ocr_page_speed():
    a, s = load("ocr_bench/page_speed_compare.json"), load("ocr_bench/page_streaming.json")
    if not a:
        return
    names = list(a["results"])
    t = [a["results"][k]["page_s_median"] for k in names]
    cer = [100 * a["results"][k]["val_cer"] for k in names]
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.5))
    short = ["beam, batch 1\n(before)", "batch 16", "batch 16 +\ngreedy (now)", "960 px", "camera's\nown size"]
    b = axs[0].bar(range(len(names)), t, 0.62, color=C1); labels(axs[0], b, "{:.1f}")
    axs[0].set_ylabel("Seconds for a 12-line page"); axs[0].set_title("Time", loc="left", fontsize=11)
    b = axs[1].bar(range(len(names)), cer, 0.62, color=C2); labels(axs[1], b, "{:.2f}")
    axs[1].axhline(cer[0] + 0.5, color=MUTED, ls="--", lw=1)
    axs[1].annotate("limit: +0.5 points", (-0.45, cer[0] + 0.5), ha="left", va="bottom", fontsize=9, color=MUTED)
    axs[1].set_ylabel("Validation letter error (%)"); axs[1].set_title("Accuracy", loc="left", fontsize=11)
    for ax in axs:
        ax.set_xticks(range(len(names)), short, fontsize=8); ax.grid(axis="x", visible=False)
    fig.suptitle("Reading a whole page on four CPU threads" + (f" (line by line: first line after {s['page_first_line_s_median']:.1f} s)" if s else ""), x=0.02, ha="left", fontsize=11)
    save(fig, "fig44_ocr_page_speed.png")


def ocr_real():
    a = load("ocr_bench/real_words_mlt19.json")
    if not a:
        return
    cats = [("all", "All words"), ("upright_only", "Upright words"), ("rotated_only", "Rotated words")]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    x = np.arange(len(cats))
    for off, lang, c, name in ((-0.19, "bn", C1, "Bangla"), (0.19, "en", C2, "Latin script")):
        v = [100 * a[lang][k]["cer"] if a[lang][k] else 0 for k, _ in cats]
        b = ax.bar(x + off, v, 0.36, color=c, label=name); labels(ax, b, "{:.0f}%")
    ax.set_xticks(x, [f"{n}\n(n = {a['bn'][k]['n'] if a['bn'][k] else 0} / {a['en'][k]['n'] if a['en'][k] else 0})" for k, n in cats])
    ax.set_ylabel("Letter error (CER, %)"); ax.grid(axis="x", visible=False); ax.legend(frameon=False, fontsize=9)
    ax.set_title("Text reading on real photographs (ICDAR 2019 MLT word images)", loc="left", fontsize=11)
    save(fig, "fig45_ocr_real_words.png")


def objects():
    a = load("object_coco_val2017_full.json")
    if not a:
        return
    runs = {(r["weights"], r["imgsz"]): r for r in a["runs"]}
    ws = [w for w in ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt") if (w, 640) in runs and (w, 320) in runs]
    x = np.arange(len(ws))
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    b1 = ax.bar(x - 0.19, [runs[(w, 320)]["map50_all_80"] for w in ws], 0.36, color=C1, label="Picture size 320 (as in the glass)")
    b2 = ax.bar(x + 0.19, [runs[(w, 640)]["map50_all_80"] for w in ws], 0.36, color=C2, label="Picture size 640")
    labels(ax, b1, "{:.2f}"); labels(ax, b2, "{:.2f}")
    ax.set_xticks(x, [w.replace(".pt", "").replace("yolov8", "YOLOv8") for w in ws]); ax.set_ylim(0, 0.8)
    ax.set_ylabel("mAP@0.5, 80 object types"); ax.grid(axis="x", visible=False); ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title(f"Object detection on all {a['images']:,} COCO validation photographs", loc="left", fontsize=11)
    save(fig, "fig46_objects_models.png")


def detector():
    a = load("ocr_bench/detector_onnx.json")
    if not a:
        return
    names = list(a["variants"])
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.4))
    for ax, key, title, fmt, col in ((axs[0], "detector_s_median", "Text detector alone (s)", "{:.2f}", C1),
                                     (axs[1], "page_s_median", "12-line page (s)", "{:.1f}", C2),
                                     (axs[2], "val_cer", "Validation letter error (%)", "{:.2f}", C3)):
        v = [a["variants"][n][key] * (100 if key == "val_cer" else 1) for n in names]
        b = ax.bar(range(len(names)), v, 0.62, color=col); labels(ax, b, fmt)
        ax.set_xticks(range(len(names)), ["PyTorch\n(before)", "ONNX full\nprecision (now)", "ONNX INT8\n(opt-in)"], fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=10.5); ax.grid(axis="x", visible=False)
    fig.suptitle("Text detector through ONNX Runtime, four CPU threads (laptop, not the Pi)", x=0.02, ha="left", fontsize=11)
    save(fig, "fig47_detector_onnx.png")


if __name__ == "__main__":
    for f in (uv_protocols, uv_counts, uv_conditions, ocr_page_speed, ocr_real, objects, detector):
        try:
            f()
        except Exception as exc:
            print("skipped", f.__name__, type(exc).__name__, exc)
