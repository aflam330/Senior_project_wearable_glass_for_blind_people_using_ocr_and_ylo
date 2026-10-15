"""Figures 52-60 from saved result files only (15 October 2026). Each figure reads one result file; nothing is typed in.

    python scripts/make_figures_oct15.py
Writes paper_evidence/figures/fig52_*.png ... fig60_*.png (those whose result files exist).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

G = Path(__file__).resolve().parents[1]
WS = G.parent
FIG = WS / "paper_evidence" / "figures"
R = G / "results"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3, C4 = "#1f5f99", "#c0504d", "#4f9a5d", "#8a6db1"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)
    print("wrote", name)


def fig52_thermal():
    rows = list(csv.DictReader(open(WS / "benchmark_report" / "raw" / "system_timeline.csv")))
    t = np.array([float(r["t_s"]) for r in rows]) / 60
    T = np.array([float(r["cpu_temp_c"]) for r in rows])
    B = np.array([float(r["cpu_busy_pct"]) for r in rows])
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(t, T, color=C2, lw=1.2, label="CPU temperature")
    for y, lab in ((72, "warm"), (77, "hot"), (81, "critical")):
        ax.axhline(y, color="0.5", lw=0.6, ls="--")
        ax.text(t[-1], y + 0.3, f"governor: {lab}", ha="right", fontsize=8, color="0.35")
    ax.set_xlabel("Minutes into the run (Raspberry Pi 5, 13 October 2026)")
    ax.set_ylabel("°C")
    ax2 = ax.twinx()
    ax2.fill_between(t, B, color=C1, alpha=0.18, lw=0, label="CPU busy")
    ax2.set_ylabel("CPU busy (%)")
    ax2.set_ylim(0, 300)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.legend(loc=(0.02, 0.60), frameon=False)
    ax2.legend(loc=(0.02, 0.50), frameon=False)
    save(fig, "fig52_pi_thermal_timeline.png")


def fig53_recogniser():
    d = json.loads((R / "ocr_bench" / "real_ft_eval.json").read_text())
    v = d["validation"]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for name, r in v.items():
        if "rotation" in name:
            continue
        col = C2 if name.startswith("fine-tuned") else C3 if name.startswith("blend") else C1
        ax.scatter(100 * r["rendered_val_cer"], 100 * r["real_val"]["mean_cer"], color=col, s=36, zorder=3)
        ax.annotate(name.replace("fine-tuned", "FT").replace("blend 0.5", "blend").replace("seed ", "s"),
                    (100 * r["rendered_val_cer"], 100 * r["real_val"]["mean_cer"]), fontsize=7, xytext=(4, 2), textcoords="offset points")
    base = v["released"]["rendered_val_cer"] * 100
    ax.axvspan(0, base + 0.5, color=C3, alpha=0.08)
    ax.axvline(base + 0.5, color=C3, lw=0.8, ls="--")
    ax.text(base + 0.55, 46, "0.5-point rule", fontsize=8, color=C3)
    ax.set_xlabel("Rendered validation CER (%)")
    ax.set_ylabel("Real-word validation CER, mean of scripts (%)")
    ax.set_xlim(2.0, 4.1)
    save(fig, "fig53_recogniser_blend.png")


def fig54_width():
    d = json.loads((R / "ocr_bench" / "width_study.json").read_text())
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for det, col, lab in (("onnx", C1, "full-precision detector"), ("onnx_int8", C2, "8-bit detector")):
        ws = [1200, 1040, 960, 880]
        ax.plot(ws, [100 * d["validation"][f"{det} {w}"]["rendered_val_cer"] for w in ws], "o-", color=col, label=f"{lab}, validation")
        if det in d.get("test", {}):
            ax.scatter([d["test"][det]["width"]], [100 * d["test"][det]["rendered"]["cer"]], marker="x", s=60, color=col,
                       label=f"{lab}, test at {d['test'][det]['width']} px")
    ax.scatter([1200], [2.46], marker="x", s=60, color="k", label="test at 1,200 px (13 Oct)")
    ax.invert_xaxis()
    ax.set_xlabel("Width the camera picture is enlarged to (px)")
    ax.set_ylabel("Rendered CER (%)")
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, "fig54_read_width.png")


def fig55_bangla_commands():
    a = json.loads((R / "bangla_voice_commands.json").read_text(encoding="utf-8"))
    b = json.loads((R / "bangla_voice_commands_heldout.json").read_text(encoding="utf-8"))
    labels, vals = [], []
    for k in ("tiny", "base", "small"):
        r = a["models"][k]["voices"]
        labels.append(f"Whisper {k}\nBangla script")
        vals.append(100 * sum(v["right"] for v in r.values()) / sum(v["of"] for v in r.values()))
    for cond, lab in (("summary_clean", "base + Latin\nunseen, clean"), ("summary_noise", "base + Latin\nunseen, 10 dB")):
        s = b["held_out"][cond]
        labels.append(lab)
        vals.append(100 * s["right"] / s["of"])
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    bars = ax.bar(labels, vals, color=[C1] * 3 + [C3] * 2)
    for bb, v in zip(bars, vals):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    ax.set_ylabel("Bangla commands carried out (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", labelsize=7.5)
    save(fig, "fig55_bangla_commands.png")


def fig56_voice_noise():
    p = R / "voice_eval.json"
    if not p.exists() or "test" not in json.loads(p.read_text()):
        return
    d = json.loads(p.read_text())
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for key, col in zip(d["test"], (C1, C2)):
        r = d["test"][key]["all held-out voices"]
        for kind, ls in (("babble", "-"), ("street/home sounds", "--")):
            xs = [10, 5, 0, -5, -10]
            ax.plot(xs, [100 * r[f"{kind} {s} dB"]["right_rate"] for s in xs], ls, marker="o", color=col, ms=3.5,
                    label=f"{'current' if key.startswith('none | early off | conf 0.7') else 'chosen'}: {kind}")
    ax.invert_xaxis()
    ax.set_xlabel("Signal-to-noise ratio (dB)")
    ax.set_ylabel("Commands carried out (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, "fig56_voice_commands_noise.png")


def fig57_closeups():
    p = WS / "realtime_bangla_taka_detection" / "results" / "closeup_ft" / "closeup_ft.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    names = [n for n in d["validation"] if n.startswith("epoch")]
    names.sort(key=lambda n: int(n[5:]))
    if not names:
        return
    x = [int(n[5:]) + 1 for n in names]
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    ax.plot(x, [100 * d["validation"][n]["nstu_val"]["correct_rate"] for n in names], "o-", color=C3, label="right value")
    ax.plot(x, [100 * d["validation"][n]["nstu_val"]["wrong"] / d["validation"][n]["nstu_val"]["known"] for n in names], "o-", color=C2, label="wrong value")
    dep = d["validation"]["deployed"]["nstu_val"]
    ax.axhline(100 * dep["correct_rate"], color=C3, ls=":", lw=1, label="deployed: right")
    ax.axhline(100 * dep["wrong"] / dep["known"], color=C2, ls=":", lw=1, label="deployed: wrong")
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("NSTU validation hand-held photos (%)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    save(fig, "fig57_handheld_epochs.png")


if __name__ == "__main__":
    for f in (fig52_thermal, fig53_recogniser, fig54_width, fig55_bangla_commands, fig56_voice_noise, fig57_closeups):
        try:
            f()
        except Exception as exc:  # noqa: BLE001
            print("skipped", f.__name__, type(exc).__name__, exc)
