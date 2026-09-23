"""12 publication figures from artifacts. Missing data → placeholder labeled NOT_MEASURED."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roboeye.qduig.artifacts import QDUIG_ROOT

FIG = WORKSPACE / "paper_evidence" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def save(fig, name: str):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=180)
    plt.close(fig)


def placeholder(name: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.text(0.5, 0.5, "NOT_MEASURED", ha="center", va="center", fontsize=16)
    ax.set_axis_off()
    ax.set_title(title)
    save(fig, name)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_axis_off()
    boxes = [
        (0.02, "Camera"),
        (0.16, "YOLO detect"),
        (0.32, "Q-DUIG auth"),
        (0.50, "OCR / emotion"),
        (0.68, "SCAFP"),
        (0.84, "TTS / haptic"),
    ]
    for x, t in boxes:
        ax.add_patch(plt.Rectangle((x, 0.35), 0.13, 0.3, fill=False))
        ax.text(x + 0.065, 0.5, t, ha="center", va="center", fontsize=8)
        if x < 0.8:
            ax.annotate("", xy=(x + 0.145, 0.5), xytext=(x + 0.13, 0.5), arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("RoboEye deployment context (not the scientific claim)")
    save(fig, "fig01_roboeye_architecture.png")


def fig_qduig():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_axis_off()
    boxes = ["Encoder", "Quality RSQA", "CVR diversity", "PCR-IG", "CRIQP policy", "HGEF fusion", "STOP/NEXT"]
    for i, t in enumerate(boxes):
        x = 0.02 + i * 0.14
        ax.add_patch(plt.Rectangle((x, 0.35), 0.12, 0.3, fill=False))
        ax.text(x + 0.06, 0.5, t, ha="center", va="center", fontsize=7)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 0.135, 0.5), xytext=(x + 0.12, 0.5), arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Q-DUIG-CAMVA sequential acquisition")
    save(fig, "fig02_qduig_architecture.png")


def fig_view_curves():
    xs, b, p = [], [], []
    for k in range(1, 7):
        bm = load(QDUIG_ROOT / "eval" / "seed42" / "baseline" / f"baseline_{k}view" / "test_metrics.json")
        pm = load(QDUIG_ROOT / "eval" / "seed42" / "policies" / f"full_proposed_{k}view" / "test_metrics.json")
        if bm and pm:
            xs.append(k)
            b.append(bm["accuracy"])
            p.append(pm["accuracy"])
    if not xs:
        placeholder("fig03_view_accuracy.png", "1–6 view accuracy")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(xs, b, marker="o", label="CNN+ViT baseline")
    ax.plot(xs, p, marker="s", label="Q-DUIG proposed")
    ax.set_xlabel("Views")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("1–6 view accuracy (test, seed 42)")
    save(fig, "fig03_view_accuracy.png")


def fig_acc_cost():
    rows = load(QDUIG_ROOT / "eval" / "seed42" / "policies" / "all_policies.json")
    if not rows:
        placeholder("fig04_acc_vs_cost.png", "Accuracy vs cost")
        placeholder("fig05_pareto.png", "Pareto frontier")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for pol in sorted({r.get("policy") for r in rows if r.get("policy")}):
        sub = [r for r in rows if r.get("policy") == pol and isinstance(r.get("n_views_requested"), int)]
        if not sub:
            continue
        ax.plot([r.get("average_views") for r in sub], [r.get("accuracy") for r in sub], marker="o", label=pol)
    ax.set_xlabel("Average views")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    save(fig, "fig04_acc_vs_cost.png")

    pareto = load(QDUIG_ROOT / "eval" / "seed42" / "pareto_val.json")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if pareto:
        pts = pareto.get("points", [])
        fr = pareto.get("pareto", [])
        ax.scatter([p["average_cost"] for p in pts], [p["accuracy"] for p in pts], label="val sweep")
        if fr:
            ax.plot([p["average_cost"] for p in fr], [p["accuracy"] for p in fr], color="C1", marker="s", label="Pareto")
        ax.set_xlabel("Average cost (val)")
        ax.set_ylabel("Val accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "NOT_MEASURED", ha="center")
    save(fig, "fig05_pareto.png")


def fig_calibration():
    cal = load(QDUIG_ROOT / "eval" / "seed42" / "calibration.json")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if not cal:
        placeholder("fig06_calibration.png", "Reliability")
        return
    for name, block in cal.get("test", {}).items():
        bins = block.get("reliability", {}).get("bins", [])
        xs, ys = [], []
        for b in bins:
            if b.get("confidence") is not None and b.get("accuracy") is not None:
                xs.append(b["confidence"])
                ys.append(b["accuracy"])
        if xs:
            ax.plot(xs, ys, marker="o", label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    save(fig, "fig06_calibration.png")


def fig_view_sel():
    pred = load(QDUIG_ROOT / "eval" / "seed42" / "policies" / "full_proposed_adaptive" / "test_predictions.json")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if not pred:
        placeholder("fig07_view_selection.png", "View selection")
        return
    counts = np.zeros(6)
    for sel in pred.get("selected_indices", []):
        for i in sel:
            if 0 <= i < 6:
                counts[i] += 1
    ax.bar(np.arange(1, 7), counts)
    ax.set_xlabel("View index")
    ax.set_ylabel("Times selected (test)")
    ax.set_title("Q-DUIG selected view indices")
    save(fig, "fig07_view_selection.png")


def fig_robust():
    rob = load(QDUIG_ROOT / "eval" / "seed42" / "robustness.json")
    if not rob:
        placeholder("fig08_robustness.png", "Robustness")
        return
    names = [r["corruption"] for r in rob]
    b = [r["baseline"]["accuracy"] if isinstance(r.get("baseline"), dict) else np.nan for r in rob]
    q = [r["proposed"]["accuracy"] if isinstance(r.get("proposed"), dict) else np.nan for r in rob]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(names))
    ax.plot(x, b, marker="o", label="baseline")
    ax.plot(x, q, marker="s", label="proposed")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "fig08_robustness.png")


def fig_cm():
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    for ax, path, title in (
        (axes[0], QDUIG_ROOT / "eval" / "seed42" / "baseline" / "baseline_6view" / "test_metrics.json", "Baseline 6-view"),
        (axes[1], QDUIG_ROOT / "eval" / "seed42" / "proposed_6view" / "test_metrics.json", "Proposed 6-view"),
    ):
        m = load(path)
        if not m:
            ax.text(0.5, 0.5, "NOT_MEASURED", ha="center")
            ax.set_title(title)
            continue
        cm = np.array(m["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_title(title)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred fake", "pred real"])
        ax.set_yticklabels(["true fake", "true real"])
    save(fig, "fig09_confusion.png")


def fig_edge():
    edge = load(QDUIG_ROOT / "edge" / "edge.json")
    if not edge or edge.get("pi5_metrics") == "NOT_MEASURED":
        # still plot current-host breakdown if present
        pass
    bd = (edge or {}).get("breakdown_ms")
    vk = (edge or {}).get("views_1_to_6")
    if not bd:
        placeholder("fig10_pi5_latency.png", "Latency (current host; Pi 5 NOT_MEASURED if not Pi)")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if vk:
        ks = sorted(vk, key=lambda s: int(s))
        ax.bar(ks, [vk[k]["median_ms"] for k in ks])
        ax.set_xlabel("Views")
        ax.set_ylabel("Median ms")
        ax.set_title(f"Latency on {edge.get('device')} (Pi5={edge.get('pi5_metrics')})")
    save(fig, "fig10_pi5_latency.png")


def fig_fail():
    pred = load(QDUIG_ROOT / "eval" / "seed42" / "proposed_6view" / "test_predictions.json")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if not pred:
        placeholder("fig11_failures.png", "Failure analysis")
        return
    y = np.array(pred["y_true"])
    p = np.array(pred["genuine_score"])
    pred_y = (p >= 0.5).astype(int)
    wrong = pred_y != y
    ax.hist(p[wrong], bins=15, alpha=0.7, label="errors")
    ax.hist(p[~wrong], bins=15, alpha=0.4, label="correct")
    ax.set_xlabel("Genuine score")
    ax.legend()
    ax.set_title("Failure score distribution (test)")
    save(fig, "fig11_failures.png")


def fig_e2e():
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.set_axis_off()
    steps = ["Hold note", "Detect", "Acquire view", "Q-DUIG", "SCAFP", "Speak / haptic"]
    for i, t in enumerate(steps):
        x = 0.03 + i * 0.16
        ax.add_patch(plt.Rectangle((x, 0.35), 0.14, 0.3, fill=False))
        ax.text(x + 0.07, 0.5, t, ha="center", va="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Assistive interaction pipeline")
    save(fig, "fig12_assistive_pipeline.png")


def main() -> None:
    fig_architecture()
    fig_qduig()
    fig_view_curves()
    fig_acc_cost()
    fig_calibration()
    fig_view_sel()
    fig_robust()
    fig_cm()
    fig_edge()
    fig_fail()
    fig_e2e()
    print("figures written", FIG)


if __name__ == "__main__":
    main()
