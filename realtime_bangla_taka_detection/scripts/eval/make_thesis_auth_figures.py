"""Rebuild the thesis authentication figures from saved result files.

  fig_auth_views.png       accuracy vs views: baseline, PRMVT and the 15 other variants
  fig_auth_multiseed.png   1-view / 6-view accuracy over seeds 42-44 (box plots)
  fig_auth_confusion.png   normalised 6-view confusion matrices, baseline vs PRMVT
  fig_auth_robustness.png  clean-minus-corrupted accuracy at 6 views (severity sweep)

PRMVT numbers come from results/qduig/prefix_ft/seed<N>/test_views_20260928 (the
re-evaluation after the 2026-09-28 NaN-entropy fix). For each other variant the figure
uses its v1 (results/novel) or v2 (results/novel_v2) seed-42 run, whichever has the higher
validation accuracy (best_val_mean_1_to_6); the test split is not used to choose.

Usage:
  python scripts/eval/make_thesis_auth_figures.py [--out "<thesis figure folder>"] [--only views multiseed ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
THESIS_FIG = ROOT.parent / "Thesis Report and paper" / "thesis report" / "My report" / "DIagram fig tab"
VARIANTS = ["ndal", "pravt", "vat", "ugf", "cvs", "apc", "mtpt", "cris", "mavt", "sfaq", "igcr", "savs",
            "vcie", "ogpd", "sfpl"]
SEEDS = [42, 43, 44]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _novel_views(algo: str, group: str, seed: int) -> list[float] | None:
    p = R / group / algo / f"seed{seed}" / "test" / "test_metrics.json"
    if not p.is_file():
        return None
    rows = {r["k"]: r["accuracy"] for r in _load(p)["views"]}
    return [rows[k] for k in range(1, 7)]


def _prmvt_views(seed: int) -> list[float]:
    return [_load(R / "qduig" / "prefix_ft" / f"seed{seed}" / "test_views_20260928" / f"{k}view" / "test_metrics.json")["accuracy"]
            for k in range(1, 7)]


def _baseline_views() -> list[float]:
    return [_load(R / "camva" / "metrics" / f"baseline_{k}view.json")["accuracy"] for k in range(1, 7)]


def fig_views(out: Path) -> None:
    lines = {"Baseline": _baseline_views(), "PRMVT": _prmvt_views(42)}
    for a in VARIANTS:
        # v1 or v2 run, chosen by validation accuracy (never by test accuracy)
        cands = []
        for g in ("novel", "novel_v2"):
            v, val = _novel_views(a, g, 42), R / g / a / "seed42" / "val_metrics.json"
            if v and val.is_file():
                cands.append((_load(val)["best_val_mean_1_to_6"], v))
        if cands:
            lines[a.upper()] = max(cands, key=lambda c: c[0])[1]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(lines) + 2))[2:]
    for (name, ys), c in zip(lines.items(), colors):
        kw = {"Baseline": {"color": "black", "lw": 1.6, "zorder": 3},
              "PRMVT": {"color": "#0b3d91", "lw": 2.2, "zorder": 4}}.get(name, {"color": c, "lw": 1.1})
        ax.plot(range(1, 7), ys, marker="o", ms=3.5, label=name, **kw)
    ax.set_xlabel("Number of views")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.55, 1.01)
    ax.legend(ncol=2, fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_auth_views.png")
    plt.close(fig)


def fig_multiseed(out: Path) -> None:
    groups = {"PRMVT": [_prmvt_views(s) for s in SEEDS]}
    for a in ("ndal", "pravt", "vat", "ugf", "cvs"):
        runs = []
        for s in SEEDS:
            v = _novel_views(a, "novel_v2", s) or _novel_views(a, "novel", s)
            if v:
                runs.append(v)
        groups[a.upper()] = runs
    data, labels, pos = [], [], []
    for i, (name, runs) in enumerate(groups.items()):
        for j, k in enumerate((1, 6)):
            data.append([r[k - 1] for r in runs])
            labels.append(f"{name}\n{k}v")
            pos.append(i * 2.4 + j)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.boxplot(data, positions=pos, widths=0.55, patch_artist=True,
               boxprops={"facecolor": "#d6e8f5", "linewidth": 0.8}, medianprops={"color": "#333333"})
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(out / "fig_auth_multiseed.png")
    plt.close(fig)


def _cm(pred_file: Path) -> np.ndarray:
    d = _load(pred_file)
    y = np.array(d["y_true"])
    p = (np.array(d["genuine_score"]) >= 0.5).astype(int)
    cm = np.zeros((2, 2))
    for t, h in zip(y, p):
        cm[t, h] += 1
    return cm / cm.sum()


def fig_confusion(out: Path) -> None:
    mats = {"Baseline": _cm(R / "camva" / "predictions" / "baseline_6view.json"),
            "PRMVT": _cm(R / "qduig" / "prefix_ft" / "seed42" / "test_views_20260928" / "6view" / "test_predictions.json")}
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.7), gridspec_kw={"wspace": 0.45})
    for ax, (name, m) in zip(axes, mats.items()):
        im = ax.imshow(m, cmap="cividis", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=9)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_xlabel(name)
    fig.colorbar(im, ax=axes, fraction=0.03)
    fig.savefig(out / "fig_auth_confusion.png", bbox_inches="tight")
    plt.close(fig)


def fig_robustness(out: Path) -> None:
    d = _load(R / "robustness" / "severity_seed42.json")
    corrs = list(dict.fromkeys(r["corruption"] for r in d["rows"]))
    fig, axes = plt.subplots(3, 4, figsize=(8, 5.4), sharey=True)
    colors = {"prmvt": "#0072B2", "ndal": "#D55E00", "baseline": "#009E73"}
    for ax, corr in zip(axes.flat, corrs):
        rows = sorted((r for r in d["rows"] if r["corruption"] == corr), key=lambda r: r["severity"])
        for m, c in colors.items():
            ax.plot([r["severity"] for r in rows], [r[f"{m}_drop"] for r in rows], marker="o", ms=3, color=c, label=m)
        ax.axhline(0, color="black", lw=0.4)
        ax.set_title(corr.replace("_", " "), fontsize=8)
        ax.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=7, frameon=False)
    fig.supylabel("Clean minus corrupted accuracy", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_auth_robustness.png")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(THESIS_FIG))
    p.add_argument("--only", nargs="*", default=["views", "multiseed", "confusion", "robustness"])
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.only:
        {"views": fig_views, "multiseed": fig_multiseed, "confusion": fig_confusion, "robustness": fig_robustness}[name](out)
        print("wrote", name, "->", out)


if __name__ == "__main__":
    main()
