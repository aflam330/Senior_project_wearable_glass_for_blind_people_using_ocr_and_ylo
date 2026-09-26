"""Publication figures from saved artifacts. Missing inputs are skipped, not drawn from guesses."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT.parent / "paper_evidence"
FIG = PAPER / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Wong colorblind palette, then Tol muted extras. No red/green pair as the only contrast.
COLORS = [
    "#000000", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
    "#56B4E9", "#F0E442", "#332288", "#88CCEE", "#44AA99", "#117733",
    "#882255", "#661100", "#6699CC", "#999933", "#AA4499",
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _box(ax, xy, text, w=2.4, h=0.7, fc="#E6E6E6"):
    x, y = xy
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                            facecolor=fc, edgecolor="#222222", linewidth=0.6)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7)


def _acc_file(path: Path) -> dict[int, float] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "views" in payload:
        return {int(row["k"]): float(row["accuracy"]) for row in payload["views"]}
    if "proposed" in payload:
        return {int(row["k"]): float(row["accuracy"]) for row in payload["proposed"]}
    return None


def _series() -> dict[str, dict[int, float]]:
    novel = {
        "NDAL": ROOT / "results" / "novel_v2" / "ndal" / "seed42" / "test" / "test_metrics.json",
        "PRAVT": ROOT / "results" / "novel_v2" / "pravt" / "seed42" / "test" / "test_metrics.json",
        "VAT": ROOT / "results" / "novel_v2" / "vat" / "seed42" / "test" / "test_metrics.json",
        "UGF": ROOT / "results" / "novel" / "ugf" / "seed42" / "test" / "test_metrics.json",
        "CVS": ROOT / "results" / "novel_v2" / "cvs" / "seed42" / "test" / "test_metrics.json",
        "APC": ROOT / "results" / "novel_v2" / "apc" / "seed42" / "test" / "test_metrics.json",
        "MTPT": ROOT / "results" / "novel_v2" / "mtpt" / "seed42" / "test" / "test_metrics.json",
        "CRIS": ROOT / "results" / "novel_v2" / "cris" / "seed42" / "test" / "test_metrics.json",
        "MAVT": ROOT / "results" / "novel_v2" / "mavt" / "seed42" / "test" / "test_metrics.json",
        "SFAQ": ROOT / "results" / "novel" / "sfaq" / "seed42" / "test" / "test_metrics.json",
        "IGCR": ROOT / "results" / "novel" / "igcr" / "seed42" / "test" / "test_metrics.json",
        "SAVS": ROOT / "results" / "novel_v2" / "savs" / "seed42" / "test" / "test_metrics.json",
        "VCIE": ROOT / "results" / "novel_v2" / "vcie" / "seed42" / "test" / "test_metrics.json",
        "OGPD": ROOT / "results" / "novel" / "ogpd" / "seed42" / "test" / "test_metrics.json",
        "SFPL": ROOT / "results" / "novel_v2" / "sfpl" / "seed42" / "test" / "test_metrics.json",
    }
    out = {}
    base = {}
    for k in range(1, 7):
        path = ROOT / "results" / "qduig" / "eval" / "seed42" / "baseline" / f"baseline_{k}view" / "test_metrics.json"
        if path.is_file():
            base[k] = float(json.loads(path.read_text(encoding="utf-8"))["accuracy"])
    if base:
        out["Baseline"] = base
    prmvt = _acc_file(ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "test_views" / "views_1_to_6.json")
    if prmvt:
        out["PRMVT"] = prmvt
    for name, path in novel.items():
        acc = _acc_file(path)
        if acc:
            out[name] = acc
    return out


def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    measured = [
        (0.2, 4.8, "Camera frames"),
        (3.2, 4.8, "Six ordered views"),
        (6.2, 4.8, "Resize 128, ImageNet norm"),
        (9.2, 4.8, "MobileNetV3-Small + TinyViT"),
        (0.2, 3.4, "Prefix mask 1–6"),
        (3.2, 3.4, "Quality, diversity,\nuncertainty, info-gain"),
        (6.2, 3.4, "Authenticity logit"),
        (9.2, 3.4, "Threshold 0.5"),
    ]
    for x, y, text in measured:
        _box(ax, (x, y), text, w=2.8, h=0.95, fc="#D6EAF8")
    ax.text(0.2, 2.55, "Present in the repository, not scored on the JaalTaka authenticity test", fontsize=7)
    other = [
        (0.2, 1.3, "YOLO currency\ndetector"),
        (3.2, 1.3, "OCR"),
        (6.2, 1.3, "Spoken output"),
        (9.2, 1.3, "FER emotion\n(other labels)"),
    ]
    for x, y, text in other:
        _box(ax, (x, y), text, w=2.8, h=0.95, fc="#F4F6F6")
    fig.tight_layout()
    _save(fig, "fig1_architecture")


def fig_sequential() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    labels = [
        (0.2, "View encoder"),
        (2.6, "Auth / quality /\nuncertainty"),
        (5.2, "Diversity /\nredundancy"),
        (7.8, "Info-gain"),
        (10.2, "STOP or NEXT"),
        (12.4, "Fusion"),
    ]
    for x, text in labels:
        _box(ax, (x, 0.7), text, w=2.1, h=0.9, fc="#FDEBD0")
    _save(fig, "fig2_sequential")


def fig_curves(series: dict[str, dict[int, float]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, (name, acc) in enumerate(series.items()):
        xs = sorted(acc)
        ax.plot(xs, [acc[k] for k in xs], marker="o", ms=3, lw=1.1, color=COLORS[i % len(COLORS)], label=name)
    ax.set_xlabel("Number of views")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(range(1, 7))
    ax.set_ylim(0.55, 1.01)
    ax.legend(ncol=2, frameon=False, loc="lower right")
    _save(fig, "fig3_accuracy_curves")


def _host_latency_ms() -> dict[int, float] | None:
    path = ROOT / "results" / "qduig" / "edge" / "edge.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("views_1_to_6") or {}
    out = {}
    for k in range(1, 7):
        row = rows.get(str(k)) or {}
        if "median_ms" not in row:
            return None
        out[k] = float(row["median_ms"])
    return out


def fig_pareto(series: dict[str, dict[int, float]]) -> None:
    latency = _host_latency_ms()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    points = []
    for i, (name, acc) in enumerate(series.items()):
        for k, a in acc.items():
            x = latency[k] if latency else float(k)
            points.append((x, a, name, COLORS[i % len(COLORS)]))
            ax.scatter(x, a, s=18, color=COLORS[i % len(COLORS)], zorder=3)
    front = []
    for x, a, name, _c in points:
        dominated = any(x2 <= x and a2 >= a and (x2 < x or a2 > a) for x2, a2, _n, _ in points)
        if not dominated:
            front.append((x, a))
    front = sorted(set(front))
    if front:
        ax.plot([p[0] for p in front], [p[1] for p in front], color="#222222", lw=1.0, zorder=2)
    if latency:
        ax.set_xlabel("Prefix-model host latency (ms)")
    else:
        ax.set_xlabel("Views used")
    ax.set_ylabel("Accuracy")
    _save(fig, "fig4_accuracy_vs_views")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for i, (name, acc) in enumerate(series.items()):
        xs = sorted(acc)
        plot_x = [latency[k] if latency else k for k in xs]
        ax.plot(plot_x, [acc[k] for k in xs], marker="o", ms=3, lw=1.0, color=COLORS[i % len(COLORS)], label=name)
    if front:
        ax.scatter([p[0] for p in front], [p[1] for p in front], s=40, facecolors="none", edgecolors="#222222", linewidths=0.8, zorder=4)
    if latency:
        ax.set_xlabel("Prefix-model host latency (ms)")
    else:
        ax.set_xlabel("Views used")
    ax.set_ylabel("Accuracy")
    ax.legend(ncol=2, frameon=False, fontsize=5)
    _save(fig, "fig5_pareto")


def fig_robustness() -> None:
    sweep = ROOT / "results" / "robustness" / "severity_seed42.json"
    if sweep.is_file():
        payload = json.loads(sweep.read_text(encoding="utf-8"))
        rows = payload["rows"]
        names = []
        for row in rows:
            if row["corruption"] not in names:
                names.append(row["corruption"])
        fig, axes = plt.subplots(3, 4, figsize=(8.2, 5.6), sharey=True)
        for ax, name in zip(axes.ravel(), names):
            sub = [r for r in rows if r["corruption"] == name]
            sub = sorted(sub, key=lambda r: float(r["severity"]))
            xs = [float(r["severity"]) for r in sub]
            for i, key in enumerate(("prmvt_drop", "ndal_drop", "baseline_drop")):
                ax.plot(xs, [r[key] for r in sub], marker="o", ms=3, lw=1.0, color=COLORS[i + 1], label=key.split("_")[0])
            ax.set_title(name.replace("_", " "), fontsize=7)
            ax.axhline(0, color="#222222", lw=0.3)
        for ax in axes.ravel()[len(names):]:
            ax.axis("off")
        axes[0, 0].legend(frameon=False, fontsize=6)
        fig.supylabel("Clean minus corrupted accuracy", fontsize=8)
        fig.tight_layout()
        _save(fig, "fig8_robustness")
        return
    path = ROOT / "results" / "robustness" / "top_seed42.json"
    if not path.is_file():
        return
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    six = [r for r in rows if int(r["k"]) == 6]
    labels = [f"{r['corruption']}\n{r['severity']}" for r in six]
    x = np.arange(len(six))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    for i, key in enumerate(("prmvt_drop", "ndal_drop", "baseline_drop")):
        vals = [r.get(key) if r.get(key) is not None else np.nan for r in six]
        ax.bar(x + (i - 1) * width, vals, width=width, color=COLORS[i + 1], label=key.split("_")[0])
    ax.axhline(0, color="#222222", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5)
    ax.set_ylabel("Clean minus corrupted accuracy")
    ax.legend(frameon=False)
    _save(fig, "fig8_robustness")


def _cm_from_json(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("matrix", payload.get("confusion_matrix"))
    arr = np.array(payload, dtype=float)
    if arr.shape != (2, 2):
        return None
    return arr / max(arr.sum(), 1.0)


def fig_confusion() -> None:
    base = _cm_from_json(ROOT / "results" / "qduig" / "eval" / "seed42" / "baseline" / "baseline_6view" / "confusion_matrix.json")
    prop = _cm_from_json(ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "test_views" / "6view" / "confusion_matrix.json")
    if base is None or prop is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.8))
    for ax, mat, label in ((axes[0], base, "Baseline"), (axes[1], prop, "PRMVT")):
        im = ax.imshow(mat, cmap="cividis", vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"])
        ax.set_yticklabels(["true 0", "true 1"])
        ax.set_xlabel(label)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    _save(fig, "fig9_confusion")


def fig_multiseed() -> None:
    algos = {
        "PRMVT": None,
        "NDAL": ROOT / "results" / "novel_v2" / "ndal",
        "PRAVT": ROOT / "results" / "novel_v2" / "pravt",
        "VAT": ROOT / "results" / "novel_v2" / "vat",
        "UGF": ROOT / "results" / "novel_v2" / "ugf",
        "CVS": ROOT / "results" / "novel_v2" / "cvs",
    }
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    positions = []
    data = []
    labels = []
    pos = 0
    for name in algos:
        for k in (1, 6):
            vals = []
            for seed in (42, 43, 44):
                if name == "PRMVT":
                    path = ROOT / "results" / "qduig" / "prefix_ft" / f"seed{seed}" / "test_views" / "views_1_to_6.json"
                    acc = _acc_file(path)
                else:
                    path = algos[name] / f"seed{seed}" / "test" / "test_metrics.json"
                    if seed == 42 and name == "UGF" and not path.is_file():
                        path = ROOT / "results" / "novel" / "ugf" / "seed42" / "test" / "test_metrics.json"
                    acc = _acc_file(path)
                if acc and k in acc:
                    vals.append(acc[k])
            if vals:
                data.append(vals)
                positions.append(pos)
                labels.append(f"{name}\n{k}v")
                pos += 1
        pos += 0.4
    if not data:
        plt.close(fig)
        return
    ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
               boxprops={"facecolor": "#D6EAF8", "linewidth": 0.6},
               medianprops={"color": "#222222", "linewidth": 0.8})
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("Accuracy")
    _save(fig, "fig11_multiseed")


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2)
    ax.axis("off")
    steps = [(0.2, "Capture"), (2.6, "Encode"), (5.0, "Classify"), (7.4, "Threshold"), (9.6, "Spoken result")]
    for x, text in steps:
        _box(ax, (x, 0.6), text, w=2.0, h=0.7, fc="#E8DAEF")
    _save(fig, "fig12_pipeline")


def fig_failures() -> None:
    pred_path = ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "test_views" / "1view" / "test_predictions.json"
    if not pred_path.is_file():
        return
    import sys
    sys.path.insert(0, str(ROOT))
    from roboeye.camva.notes import load_splits
    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    y = np.array(pred["y_true"])
    p = np.array(pred["genuine_score"])
    wrong = np.where((p >= 0.5).astype(int) != y)[0]
    if len(wrong) == 0:
        return
    _splits, records = load_splits()
    ids = pred["note_id"]
    picks = wrong[:6]
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.2))
    import cv2
    drawn = 0
    for ax, idx in zip(axes.ravel(), picks):
        rec = records.get(ids[idx])
        ax.axis("off")
        if not rec:
            continue
        img = cv2.imread(str(rec["view_paths"][0]))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_xlabel(f"true {int(y[idx])}  pred {int(p[idx] >= 0.5)}", fontsize=7)
        drawn += 1
    if drawn == 0:
        plt.close(fig)
        return
    _save(fig, "fig10_failures")


def main() -> None:
    series = _series()
    fig_architecture()
    fig_sequential()
    if series:
        fig_curves(series)
        fig_pareto(series)
    fig_robustness()
    fig_confusion()
    fig_failures()
    fig_multiseed()
    fig_pipeline()
    note = FIG / "FIGURE_NOTES.md"
    note.write_text(
        "\n".join([
            "# Figure notes",
            "",
            "Axes are labeled. Captions are not drawn inside the figures.",
            "Figure 1 draws the measured authenticity path in blue. Grey boxes are other programs in this repository. They were not scored on the JaalTaka authenticity test. Microphone and tactile sensing are not in the repository and are not drawn.",
            "Figure 4 and 5 place each view count at the prefix model's measured host median latency for that count, from results/qduig/edge/edge.json. That file is this Windows CUDA host, not a Raspberry Pi. The same latency is used for every model because separate timings were not in that file. Latency already grows with the view count, so the axis is that measurement rather than view count times a constant.",
            "Figure 6 is written by the calibration script when reliability bins exist.",
            "Figure 7 is the collapsed acquisition policy from eval_cost_policy.py.",
            "Figure 8 uses results/robustness/severity_seed42.json when that file exists, one panel per corruption. Otherwise it is the single-severity bar chart.",
            "Figure 10 shows test notes the seed-42 PRMVT checkpoint got wrong at 1 view.",
            "",
        ]),
        encoding="utf-8",
    )
    print(FIG)


if __name__ == "__main__":
    main()
