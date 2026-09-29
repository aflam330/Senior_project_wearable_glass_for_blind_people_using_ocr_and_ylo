"""Figures for the 2026-09-29 results, drawn only from saved result files.

fig13_same_arch.png     accuracy vs views for the four same-network arms (mean ± std, 3 seeds)
fig14_leaderboard.png   1-view and 6-view accuracy of the 3-seed methods and the frozen probes
fig15_jaal_whole.png    p(genuine) on whole-note photos per set, deployed vs cut-view checkers
                        (only if results/jaal_whole/scores.json exists)
Output folder: paper_evidence/figures/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
import write_benchmark_final as bench  # noqa: E402

FIG = ROOT.parent / "paper_evidence" / "figures"
KS = list(range(1, 7))
INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"
COLORS = ["#2a6fdb", "#e0782f", "#1a9e77", "#c4417a", "#7a5bd6", "#8a8f98"]


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)


def fig_same_arch():
    arms = [("Same network, prefix, shared BN", "prefix, shared BN"),
            ("PRMVT (prefix + fine-tune)", "prefix, per-count BN (PRMVT)"),
            ("Same network, fixed 6-view, per-count BN", "fixed 6-view, per-count BN"),
            ("Same network, fixed 6-view, shared BN", "fixed 6-view, shared BN")]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for (key, label), c, ls in zip(arms, COLORS, ["-", "-", "--", "--"]):
        runs = [bench.THREE_SEED[key](s) for s in bench.SEEDS]
        a = np.array([[r[k] for k in KS] for r in runs]) * 100
        m, sd = a.mean(0), a.std(0, ddof=1)
        ax.plot(KS, m, ls, color=c, lw=2, marker="o", ms=4, label=label)
        ax.fill_between(KS, m - sd, m + sd, color=c, alpha=0.12, lw=0)
    ax.set_xlabel("views given at test time (first k)")
    ax.set_ylabel("test accuracy, % (208 notes)")
    ax.set_title("Same network, fixed-view vs prefix training (3 seeds, ±1 sd)", color=INK, fontsize=10)
    ax.set_ylim(55, 101)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig13_same_arch.png", dpi=200)
    plt.close(fig)


def fig_leaderboard():
    rows = []
    for key, fn in bench.THREE_SEED.items():
        runs = [fn(s) for s in bench.SEEDS]
        if all(runs):
            rows.append((key, np.mean([r[1] for r in runs]) * 100, np.mean([r[6] for r in runs]) * 100, "3 seeds"))
    pr = bench.load(bench.R / "sota" / "backbone_probes.json") or {}
    for name, r in pr.items():
        rows.append((f"{name} probe", r["test_acc"]["1"] * 100, r["test_acc"]["6"] * 100, "probe"))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7, 0.34 * len(rows) + 1.2))
    y = np.arange(len(rows))
    ax.barh(y + 0.2, [r[1] for r in rows], 0.38, color=COLORS[0], label="1 view")
    ax.barh(y - 0.2, [r[2] for r in rows], 0.38, color=COLORS[1], label="6 views")
    ax.set_yticks(y, [f"{r[0]}{'' if r[3] == '3 seeds' else ' *'}" for r in rows], fontsize=7.5)
    ax.set_xlim(45, 101)
    ax.set_xlabel("test accuracy, % (208 JaalTaka notes; 3-seed mean; * = frozen probe, deterministic)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig14_leaderboard.png", dpi=200)
    plt.close(fig)


def fig_jaal():
    p = ROOT / "results" / "jaal_whole" / "scores.json"
    pol = ROOT / "results" / "jaal_whole" / "policy.json"
    if not p.is_file():
        return
    rows = json.loads(p.read_text(encoding="utf-8"))
    chosen = json.loads(pol.read_text(encoding="utf-8"))["chosen_checker_on_val"] if pol.is_file() else "S4"
    groups = [("cf", "counterfeit", "counterfeit set\ncounterfeit"), ("cf", "genuine", "counterfeit set\ngenuine"),
              ("bm", "genuine", "Bangla Money\ngenuine"), ("bt", "genuine", "BanglaTaka\ngenuine")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, key, title in zip(axes, ["S0", chosen], ["deployed: whole crop as 1 view (S0)", f"cut views, chosen on VAL ({chosen})"]):
        data = [[r[key] for r in rows if r["set"] == s and r["truth"] == t and r.get("yolo") and not r["augmented"]]
                for s, t, _ in groups]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True,
                        flierprops={"markersize": 2, "markerfacecolor": MUTED, "markeredgecolor": MUTED})
        for patch, c in zip(bp["boxes"], [COLORS[3], COLORS[0], COLORS[0], COLORS[0]]):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
        ax.set_xticks(range(1, 5), [g[2] for g in groups], fontsize=7.5)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
        ax.set_title(title, fontsize=9, color=INK)
        style(ax)
    axes[0].set_ylabel("p(genuine)")
    fig.tight_layout()
    fig.savefig(FIG / "fig15_jaal_whole.png", dpi=200)
    plt.close(fig)


def fig_unseen_prints():
    """fig16: standard vs serial-disjoint accuracy, network vs network + watermark (all available seeds)."""
    p = ROOT / "results/watermark/hybrid_final_seeds.json"  # final: watermark model chosen on VAL AUC (MobileNetV2)
    probe = ROOT / "results/watermark/serial_split_eval.json"
    if not p.is_file():
        return
    s = json.loads(p.read_text(encoding="utf-8"))["summary"]
    for key in s:
        s[key].setdefault("seeds", [42, 43, 44])
    pr = json.loads(probe.read_text(encoding="utf-8"))["PROBE"]
    labels = ["1 view", "6 views"]
    groups = [("ResNet-50 probe", [pr["k1"]["accuracy"] * 100, pr["k6"]["accuracy"] * 100], [0, 0]),
              ("prefix network", [s["k1/network"]["mean"] * 100, s["k6/network"]["mean"] * 100],
               [(s["k1/network"]["sd"] or 0) * 100, (s["k6/network"]["sd"] or 0) * 100]),
              ("prefix network + watermark", [s["k1/hybrid"]["mean"] * 100, s["k6/hybrid"]["mean"] * 100],
               [(s["k1/hybrid"]["sd"] or 0) * 100, (s["k6/hybrid"]["sd"] or 0) * 100])]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(2)
    for i, (name, v, e) in enumerate(groups):
        ax.bar(x + (i - 1) * 0.26, v, 0.24, yerr=e, capsize=3, color=COLORS[i], label=name)
    ax.set_xticks(x, labels)
    ax.set_ylim(70, 100)
    ax.set_ylabel("test accuracy, % (222 notes)")
    ax.set_title("Unseen counterfeit prints (serial-disjoint split; mean ± sd, 3 seeds)", fontsize=10, color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.0))
    ax.set_ylim(70, 102)
    fig.tight_layout()
    fig.savefig(FIG / "fig16_unseen_prints.png", dpi=200)
    plt.close(fig)


def fig_watermark_examples():
    """fig17: back-lit watermark windows, genuine (top) and counterfeit (bottom), TEST notes of the seed-42 split."""
    import random
    import cv2
    rows = [r for r in json.loads((ROOT / "results/watermark/features.json").read_text(encoding="utf-8")) if r["ok"] and r["split"] == "test"]
    rng = random.Random(7)
    pick = [rng.sample([r for r in rows if r["label"] == lab], 6) for lab in (1, 0)]
    fig, axes = plt.subplots(2, 6, figsize=(9, 3.4))
    for r_i, grp in enumerate(pick):
        for c_i, r in enumerate(grp):
            img = cv2.imread(str(ROOT / "results/watermark/crops" / (r["note_id"].replace(":", "_") + ".png")))
            axes[r_i, c_i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[r_i, c_i].axis("off")
    axes[0, 0].set_title("genuine", loc="left", fontsize=9, color=INK)
    axes[1, 0].set_title("counterfeit", loc="left", fontsize=9, color=INK)
    fig.suptitle("Back-lit watermark window (JaalTaka view 6, registered to the note template)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig17_watermark_windows.png", dpi=200)
    plt.close(fig)


def fig_fusion():
    """fig18: MVP-N, fixed vs prefix training per fusion head, 1 and 6 views."""
    v = json.loads((ROOT / "results/mvpn/vcds.json").read_text(encoding="utf-8"))["summary"]
    f = json.loads((ROOT / "results/mvpn/fusion_fix.json").read_text(encoding="utf-8"))
    heads = [("attention", v["attn/fixed"], v["attn/prefix"]), ("mean pool", f["meanpool/fixed"], f["meanpool/prefix"]),
             ("concat", v["concat/fixed"], v["concat/prefix"]), ("concat, rescaled", f["concat_scaled/fixed"], f["concat_scaled/prefix"])]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, k in zip(axes, ("1", "6")):
        x = np.arange(len(heads))
        ax.bar(x - 0.18, [h[1][k]["mean"] * 100 for h in heads], 0.34, color=COLORS[5], label="fixed 6-view")
        ax.bar(x + 0.18, [h[2][k]["mean"] * 100 for h in heads], 0.34, color=COLORS[0], label="prefix")
        ax.set_xticks(x, [h[0] for h in heads], fontsize=8)
        ax.set_title(f"{k} view{'s' if k != '1' else ''}", fontsize=9, color=INK)
        style(ax)
    axes[0].set_ylabel("MVP-N test accuracy, % (3 seeds)")
    axes[0].set_ylim(30, 90)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Prefix training helps pooling fusion, not slot concatenation", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig18_fusion_designs.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_same_arch()
    fig_leaderboard()
    fig_jaal()
    fig_unseen_prints()
    fig_watermark_examples()
    fig_fusion()
    print("figures written to", FIG)
