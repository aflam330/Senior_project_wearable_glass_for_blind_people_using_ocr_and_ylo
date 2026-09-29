"""Figures 19-27 (PNG + PDF, 300 dpi), drawn only from saved result files.

19 standard vs print-disjoint accuracy          20 watermark vs serial vs hybrid (seed-42 split)
21 missed fakes and false alarms, unseen prints 22 5-fold CV: note vs serial-grouped folds
23 safety guarantee of the max-of-VAL threshold 24 serial leakage: shared counterfeit serials
25 all three-seed methods, per seed             26 system pipeline (diagram)
27 architecture with the watermark branch (diagram)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
import write_benchmark_final as bench  # noqa: E402
from make_final_figures import COLORS, GRID, INK, MUTED, style  # noqa: E402

R = ROOT / "results"
FIG = ROOT.parent / "paper_evidence" / "figures"
J = lambda p: json.loads((R / p).read_text(encoding="utf-8"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=300)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def f19():
    same = J("same_arch/prefix_shbn/s2/seed42/test_views/views_1_to_6.json")
    runs = [bench.THREE_SEED["Same network, prefix, shared BN"](s) for s in bench.SEEDS]
    std_net = [np.mean([r[1] for r in runs]) * 100, np.mean([r[6] for r in runs]) * 100]
    probes = J("sota/backbone_probes.json")["resnet50"]["test_acc"]
    hf = J("watermark/hybrid_final_seeds.json")["summary"]
    ss = J("watermark/serial_split_eval.json")["PROBE"]
    rows = [("ResNet-50 probe", [probes["1"] * 100, probes["6"] * 100], [ss["k1"]["accuracy"] * 100, ss["k6"]["accuracy"] * 100]),
            ("prefix network", std_net, [hf["k1/network"]["mean"] * 100, hf["k6/network"]["mean"] * 100])]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, i, t in zip(axes, (0, 1), ("1 view", "6 views")):
        x = np.arange(len(rows))
        ax.bar(x - 0.18, [r[1][i] for r in rows], 0.34, color=COLORS[5], label="note-disjoint (standard)")
        ax.bar(x + 0.18, [r[2][i] for r in rows], 0.34, color=COLORS[3], label="print-disjoint (unseen prints)")
        ax.set_xticks(x, [r[0] for r in rows], fontsize=8)
        ax.set_title(t, fontsize=9, color=INK)
        ax.set_ylim(75, 100)
        style(ax)
    axes[0].set_ylabel("test accuracy, %")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("JaalTaka accuracy drops when counterfeit prints are unseen", fontsize=10, color=INK)
    save(fig, "fig19_split_comparison")


def f20():
    h = J("watermark/hybrid.json")
    items = [("serial list", h["S_list"]["test"]["accuracy"]), ("serial duplicate", h["S_dup"]["test"]["accuracy"]),
             ("PRMVT, 1 view", h["PRMVT_k1"]["test"]["accuracy"]), ("watermark window", h["W_deep"]["test"]["accuracy"]),
             ("view 1 + watermark", h["HYBRID_k1"]["test"]["accuracy"])]
    unseen = [h["S_list"]["test"], h["S_dup"]["test"], h["PRMVT_k1"]["test"], h["W_deep"]["test"], h["HYBRID_k1"]["test"]]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(items))
    ax.bar(x, [v * 100 for _, v in items], color=[COLORS[5], COLORS[5], COLORS[1], COLORS[2], COLORS[0]])
    for i, u in enumerate(unseen):
        ax.text(i, items[i][1] * 100 + 0.6, f"unseen-serial fakes\ncaught {u['unseen_serial_counterfeit_caught']}/{u['unseen_serial_counterfeit_n']}",
                ha="center", fontsize=6.5, color=MUTED)
    ax.set_xticks(x, [n for n, _ in items], fontsize=8)
    ax.set_ylim(80, 104)
    ax.set_ylabel("test accuracy, % (seed-42 split)")
    ax.set_title("Serial number vs watermark vs hybrid", fontsize=10, color=INK)
    style(ax)
    save(fig, "fig20_watermark_serial_hybrid")


def f21():
    hf = J("watermark/hybrid_final_seeds.json")["per_seed"]
    ss = J("watermark/serial_split_eval.json")
    meth = [("ResNet-50 probe", [ss["PROBE"]["k6"]["counterfeit_missed"]] * 3, [ss["PROBE"]["k6"]["false_counterfeit_on_genuine"]] * 3),
            ("prefix network", [hf[s]["k6"]["network"]["counterfeit_missed"] for s in hf], [hf[s]["k6"]["network"]["false_counterfeit_on_genuine"] for s in hf]),
            ("network + watermark", [hf[s]["k6"]["hybrid"]["counterfeit_missed"] for s in hf], [hf[s]["k6"]["hybrid"]["false_counterfeit_on_genuine"] for s in hf])]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
    for ax, idx, title, n in ((axes[0], 1, "counterfeits missed (of 101)", 101), (axes[1], 2, "genuine called counterfeit (of 121)", 121)):
        x = np.arange(len(meth))
        vals = [np.mean(m[idx]) for m in meth]
        ax.bar(x, vals, color=[COLORS[5], COLORS[1], COLORS[0]])
        for i, m in enumerate(meth):
            ax.scatter([i] * 3, m[idx], color=INK, s=10, zorder=3)
        ax.set_xticks(x, [m[0] for m in meth], fontsize=7.5)
        ax.set_title(title, fontsize=9, color=INK)
        style(ax)
    axes[1].axhline(0.05 * 121, color=COLORS[3], ls="--", lw=1)
    axes[1].text(2.4, 0.05 * 121 + 0.2, "5 % target", fontsize=7, color=COLORS[3], ha="right")
    fig.suptitle("Unseen prints, 6 views: bars = mean over 3 seeds, dots = seeds", fontsize=10, color=INK)
    save(fig, "fig21_missed_and_false_alarms")


def f22():
    cv = J("sota/cv_probe.json")
    fig, ax = plt.subplots(figsize=(6, 3.3))
    for i, (d, c) in enumerate((("note", COLORS[5]), ("serial", COLORS[3]))):
        ax.plot(range(1, 6), np.array(cv[d]["fold_k1"]) * 100, "o-", color=c, label=f"{d} folds, 1 view (pooled {cv[d]['pooled_accuracy_all_1390_notes']['1'] * 100:.1f} %)")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("fold")
    ax.set_ylabel("test accuracy, %")
    ax.set_title("5-fold CV over all 1,390 notes (ResNet-50 probe)", fontsize=10, color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig22_cv_folds")


def f23():
    eps = np.linspace(0.001, 0.15, 200)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for n, c in ((30, COLORS[5]), (88, COLORS[0]), (200, COLORS[2])):
        ax.plot(eps * 100, (1 - eps) ** n, color=c, label=f"n = {n} validation counterfeits")
    ax.axvline(5, color=MUTED, ls=":")
    ax.scatter([5], [0.95 ** 88], color=COLORS[0], zorder=3)
    ax.text(5.4, 0.95 ** 88 + 0.03, f"n = 88: P(pass rate > 5 %) = {0.95 ** 88:.3f}", fontsize=7.5, color=INK)
    ax.set_xlabel("true pass rate ε of the threshold, %")
    ax.set_ylabel("P(pass rate > ε) = (1 − ε)^n")
    ax.set_title("Theorem 8: safety of the max-of-validation threshold", fontsize=10, color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig23_safety_guarantee")


def f24():
    s = J("jaal_whole/serial_audit.json")["summary"]
    keys = ["500/counterfeit", "1000/counterfeit", "500/genuine", "1000/genuine"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(4)
    ax.bar(x - 0.2, [s[k]["serial_read"] for k in keys], 0.38, color=COLORS[5], label="notes with a readable serial")
    ax.bar(x + 0.2, [s[k]["top3"][0][1] for k in keys], 0.38, color=COLORS[3], label="notes carrying the most common serial")
    for i, k in enumerate(keys):
        ax.text(i + 0.2, s[k]["top3"][0][1] + 6, s[k]["top3"][0][0], ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, [k.replace("/", " BDT\n") for k in keys], fontsize=8)
    ax.set_ylabel("notes")
    ax.set_title("JaalTaka counterfeits share printed serials; genuine notes do not", fontsize=10, color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig24_serial_leakage")


def f25():
    rows = []
    for name, fn in bench.THREE_SEED.items():
        runs = [fn(s) for s in bench.SEEDS]
        if all(runs):
            rows.append((name, [r[1] * 100 for r in runs]))
    rows.sort(key=lambda r: np.mean(r[1]))
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(rows) + 1.2))
    for i, (n, v) in enumerate(rows):
        ax.scatter(v, [i] * 3, color=COLORS[0], s=14, zorder=3)
        ax.plot([min(v), max(v)], [i, i], color=GRID, lw=3)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7)
    ax.set_xlabel("1-view test accuracy, % (208 JaalTaka notes; one dot per seed)")
    ax.set_title("All methods with three seeds (standard split)", fontsize=10, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig25_methods_3seeds")


def boxes(ax, items, y=0.5, h=0.34):
    n = len(items)
    w = 0.9 / n
    for i, (t, c) in enumerate(items):
        x = 0.05 + i * w
        ax.add_patch(FancyBboxPatch((x, y - h / 2), w * 0.82, h, boxstyle="round,pad=0.01", fc=c, ec=MUTED, alpha=0.9))
        ax.text(x + w * 0.41, y, t, ha="center", va="center", fontsize=7.2, color=INK, wrap=True)
        if i < n - 1:
            ax.annotate("", xy=(x + w, y), xytext=(x + w * 0.82, y), arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def f26():
    fig, ax = plt.subplots(figsize=(9, 2.2))
    boxes(ax, [("camera\nframe", "#e8eef9"), ("Taka YOLOv8s\n(denomination)", "#e8eef9"), ("crop +\n4 view windows", "#e8eef9"),
               ("PRMVT\n(prefix-trained)", "#e3f4ec"), ("policy E\np > τ ?", "#fdf0e3"), ("speech:\n'likely genuine' /\n'check by hand'", "#f9e6ee")])
    ax.set_title("Glass pipeline for 500 / 1,000 Taka (never says 'counterfeit')", fontsize=10, color=INK)
    save(fig, "fig26_pipeline")


def f27():
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    top = [("views 1..k\n(front, any k)", "#e8eef9"), ("prefix-trained\nnetwork", "#e3f4ec"), ("network\nlogit", "#e3f4ec")]
    bot = [("back-lit view\n(hold to light)", "#e8eef9"), ("register to template,\ncrop watermark window", "#fdf0e3"), ("MobileNetV2\n(2.6 MB INT8)", "#fdf0e3")]
    for row, yy in ((top, 0.72), (bot, 0.28)):
        for i, (t, c) in enumerate(row):
            x = 0.03 + i * 0.22
            ax.add_patch(FancyBboxPatch((x, yy - 0.12), 0.18, 0.24, boxstyle="round,pad=0.01", fc=c, ec=MUTED))
            ax.text(x + 0.09, yy, t, ha="center", va="center", fontsize=7.2, color=INK)
            if i < 2:
                ax.annotate("", xy=(x + 0.22, yy), xytext=(x + 0.18, yy), arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.add_patch(FancyBboxPatch((0.70, 0.38), 0.12, 0.24, boxstyle="round,pad=0.01", fc="#f9e6ee", ec=MUTED))
    ax.text(0.76, 0.5, "combiner\n(fitted on\nvalidation)", ha="center", va="center", fontsize=7.2, color=INK)
    for yy in (0.72, 0.28):
        ax.annotate("", xy=(0.70, 0.5), xytext=(0.65, yy), arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.add_patch(FancyBboxPatch((0.86, 0.38), 0.12, 0.24, boxstyle="round,pad=0.01", fc="#eeeeee", ec=MUTED))
    ax.text(0.92, 0.5, "p(genuine)\n+ policy E", ha="center", va="center", fontsize=7.2, color=INK)
    ax.annotate("", xy=(0.86, 0.5), xytext=(0.82, 0.5), arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_title("Watermark-aware counterfeit check", fontsize=10, color=INK)
    save(fig, "fig27_architecture_watermark")


if __name__ == "__main__":
    for f in (f19, f20, f21, f22, f23, f24, f25, f26, f27):
        f()
        print("drew", f.__name__, flush=True)
