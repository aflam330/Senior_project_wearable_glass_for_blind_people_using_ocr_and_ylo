"""Figures 28-32 from saved serial-disjoint results. PNG and PDF at 300 dpi.

28  quarter-resolution fine-tune vs full-resolution fine-tune vs frozen probe vs watermark hybrid
29  per-seed six-view accuracy of those four
30  published half-decode watermark model vs full-decode retraining
31  denomination-matched watermark residual vs the published MobileNetV2
32  four-checkpoint watermark ensemble at 0.5 and at the validation threshold
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
FIG = ROOT.parent / "paper_evidence" / "figures"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=300)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def main() -> None:
    beat = load(R / "serial_split" / "beat_resnet.json")["summary"]
    hybrid = load(R / "serial_split" / "hybrid_ft_fullres.json")["summary"]
    labels = ["Frozen\nResNet-50", "Fine-tune\nquarter JPEG", "Fine-tune\nfull JPEG", "Watermark\nhybrid"]
    k1 = [
        beat["k1/PROBE"]["mean"] * 100,
        beat["k1/FT_RESNET"]["mean"] * 100,
        hybrid["k1/ft"]["mean"] * 100,
        beat["k1/HYBRID_FINAL"]["mean"] * 100,
    ]
    k6 = [
        beat["k6/PROBE"]["mean"] * 100,
        beat["k6/FT_RESNET"]["mean"] * 100,
        hybrid["k6/ft"]["mean"] * 100,
        beat["k6/HYBRID_FINAL"]["mean"] * 100,
    ]
    err1 = [0, beat["k1/FT_RESNET"]["sd"] * 100, hybrid["k1/ft"]["sd"] * 100, beat["k1/HYBRID_FINAL"]["sd"] * 100]
    err6 = [0, beat["k6/FT_RESNET"]["sd"] * 100, hybrid["k6/ft"]["sd"] * 100, beat["k6/HYBRID_FINAL"]["sd"] * 100]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, k1, w, yerr=err1, capsize=3, label="1 view", color="#1f4e79")
    ax.bar(x + w / 2, k6, w, yerr=err6, capsize=3, label="6 views", color="#c4a35a")
    ax.set_xticks(x, labels)
    ax.set_ylim(75, 100)
    ax.set_ylabel("Accuracy on unseen prints (%)")
    ax.set_title("Serial-disjoint test, 222 notes, seeds 42–44")
    ax.legend(frameon=False)
    ax.axhline(85.1, color="#1f4e79", lw=0.4, ls=":")
    save(fig, "fig28_pipeline_bug")

    per = load(R / "serial_split" / "beat_resnet.json")["per_seed"]
    full = load(R / "serial_split" / "hybrid_ft_fullres.json")["per_seed"]
    seeds = ["42", "43", "44"]
    series = {
        "Frozen": [per[s]["k6"]["PROBE"]["accuracy"] * 100 for s in seeds],
        "Quarter fine-tune": [per[s]["k6"]["FT_RESNET"]["accuracy"] * 100 for s in seeds],
        "Full fine-tune": [full[s]["k6"]["ft"]["accuracy"] * 100 for s in seeds],
        "Watermark hybrid": [per[s]["k6"]["HYBRID_FINAL"]["accuracy"] * 100 for s in seeds],
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, ys in series.items():
        ax.plot(seeds, ys, marker="o", label=name)
    ax.set_ylim(75, 100)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Six-view accuracy (%)")
    ax.set_title("Unseen prints, six views, per seed")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig29_fullres_vs_hybrid")

    pub = load(R / "watermark" / "mobilenetv2.json")["test"]["accuracy"] * 100
    wm = [load(R / "watermark_fullres" / f"mobilenetv2_seed{s}.json")["test"]["accuracy"] * 100 for s in (42, 43, 44)]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(["Published\nhalf decode", "Full decode\nseed 42", "Full decode\nseed 43", "Full decode\nseed 44"],
           [pub, *wm], color=["#1f4e79", "#c4a35a", "#c4a35a", "#c4a35a"])
    ax.set_ylim(80, 100)
    ax.set_ylabel("Accuracy on registered test crops (%)")
    ax.set_title("Watermark MobileNetV2, unseen prints")
    save(fig, "fig30_watermark_full_decode")

    dmwr = load(R / "watermark_dmwr" / "summary.json")
    order = [
        ("Matched\nfilter", dmwr["logistic_matched_filter"]["test"]["accuracy"] * 100, 0.0),
        ("Portrait\nstream", dmwr["test_summary"]["portrait"]["accuracy"]["mean"] * 100, dmwr["test_summary"]["portrait"]["accuracy"]["std"] * 100),
        ("Residual\nstream", dmwr["test_summary"]["residual"]["accuracy"]["mean"] * 100, dmwr["test_summary"]["residual"]["accuracy"]["std"] * 100),
        ("DMWR\nboth streams", dmwr["test_summary"]["both"]["accuracy"]["mean"] * 100, dmwr["test_summary"]["both"]["accuracy"]["std"] * 100),
        ("Published\nMobileNetV2", dmwr["published_mobilenet_on_same_test_notes"]["accuracy"] * 100, 0.0),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    yerr = [e if e > 0 else np.nan for _, _, e in order]
    ax.bar([n for n, _, _ in order], [v for _, v, _ in order], yerr=yerr,
           color=["#8aa0b4", "#8aa0b4", "#8aa0b4", "#1f4e79", "#c4a35a"], capsize=3)
    ax.set_ylim(70, 100)
    ax.set_ylabel("Accuracy on registered test crops (%)")
    ax.set_title("Watermark window, unseen prints")
    save(fig, "fig31_watermark_dmwr")
    ens = load(R / "watermark_ensemble" / "summary.json")
    labels = ["Published\nthreshold 0.5", "Ensemble\nthreshold 0.5", "Ensemble\nvalidation threshold"]
    vals = [
        ens["published_rescored"]["accuracy"] * 100,
        ens["ensemble_at_0.5"]["accuracy"] * 100,
        ens["ensemble_val_threshold"]["accuracy"] * 100,
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(labels, vals, color=["#c4a35a", "#8aa0b4", "#1f4e79"])
    ax.set_ylim(88, 96)
    ax.set_ylabel("Accuracy on registered test crops (%)")
    ax.set_title("Watermark MobileNet ensemble, unseen prints")
    save(fig, "fig32_watermark_ensemble")
    print("wrote fig28, fig29, fig30, fig31 and fig32")


if __name__ == "__main__":
    main()
