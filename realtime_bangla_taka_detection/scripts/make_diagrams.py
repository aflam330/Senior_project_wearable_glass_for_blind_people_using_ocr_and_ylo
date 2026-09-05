"""Generate methodology-pipeline and model-architecture diagrams for the report."""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

OUT = Path("results/report_assets")
OUT.mkdir(parents=True, exist_ok=True)

BLUE   = "#2E5EAA"
GREEN  = "#3A7D44"
ORANGE = "#D9822B"
PURPLE = "#6A4C93"
GREY   = "#4A4A4A"
LIGHT  = "#EAF0F8"


def box(ax, x, y, w, h, text, color, tcolor="white", fs=10):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                       linewidth=1.2, edgecolor=color, facecolor=color, alpha=0.92)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tcolor, weight="bold", wrap=True)


def arrow(ax, x1, y1, x2, y2, color=GREY):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                        linewidth=1.6, color=color, shrinkA=2, shrinkB=2)
    ax.add_patch(a)


# ── 1. Methodology pipeline ──────────────────────────────────────────────────

def methodology_diagram():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Figure 1 — Methodology Pipeline", fontsize=13, weight="bold", pad=12)

    # Row 1: inputs
    box(ax, 0.4, 8.0, 3.0, 1.3,
        "Source Notes\n5,073 background-free\nBDT banknote images\n(9 denominations)", BLUE, fs=9)
    box(ax, 4.5, 8.0, 3.0, 1.3,
        "Real Backgrounds\n1,500 COCO val2017\nphotos (desks, hands,\nrooms, outdoor)", GREEN, fs=9)
    box(ax, 8.6, 8.0, 3.0, 1.3,
        "Pretrained Weights\nYOLOv8s\n(COCO-pretrained\nbackbone)", PURPLE, fs=9)

    # Row 2: synthesis
    box(ax, 2.2, 5.7, 7.6, 1.5,
        "Synthetic Copy-Paste Compositing Engine\n"
        "perspective warp · rotation ±30° · scale 0.25–0.90 · HSV/brightness jitter\n"
        "motion blur · Gaussian noise · JPEG re-compression · 10% negative samples", ORANGE, fs=9)

    # Row 3: dataset
    box(ax, 2.2, 3.6, 7.6, 1.2,
        "YOLO Dataset  (auto-labelled bounding boxes, 640x640)\n"
        "Train 17,844  ·  Valid 2,233  ·  Test 2,238", BLUE, fs=9.5)

    # Row 4: training
    box(ax, 0.6, 1.5, 4.7, 1.3,
        "Training\nYOLOv8s · 80 epochs · imgsz 640\nSGD(auto) · batch 16 · RTX 4060", GREEN, fs=9)
    box(ax, 6.7, 1.5, 4.7, 1.3,
        "Evaluation & Deployment\ntest-set mAP · confusion matrix\nreal-time webcam inference", PURPLE, fs=9)

    # arrows
    arrow(ax, 1.9, 8.0, 4.5, 7.25)
    arrow(ax, 6.0, 8.0, 6.0, 7.25)
    arrow(ax, 10.1, 8.0, 7.5, 7.25)
    arrow(ax, 6.0, 5.7, 6.0, 4.85)
    arrow(ax, 5.0, 3.6, 3.0, 2.85)
    arrow(ax, 7.0, 3.6, 9.0, 2.85)
    arrow(ax, 5.3, 2.15, 6.7, 2.15)

    plt.tight_layout()
    plt.savefig(OUT / "methodology.png", dpi=170, bbox_inches="tight")
    plt.close()
    print("saved methodology.png")


# ── 2. Model architecture ─────────────────────────────────────────────────────

def architecture_diagram():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Figure 2 — YOLOv8s Detection Architecture", fontsize=13, weight="bold", pad=10)

    box(ax, 0.2, 2.0, 1.7, 1.2, "Input\n640x640x3", GREY, fs=9)
    box(ax, 2.3, 1.6, 2.4, 2.0,
        "Backbone\nCSPDarknet\n(C2f blocks +\nSPPF)", BLUE, fs=9)
    box(ax, 5.1, 1.6, 2.4, 2.0,
        "Neck\nPAN-FPN\n(multi-scale\nfeature fusion)", GREEN, fs=9)
    box(ax, 7.9, 1.6, 2.4, 2.0,
        "Head\nDecoupled\nanchor-free\n(3 scales)", ORANGE, fs=9)
    box(ax, 10.7, 2.0, 2.1, 1.2,
        "Output\n9 classes\n+ boxes", PURPLE, fs=9)

    arrow(ax, 1.9, 2.6, 2.3, 2.6)
    arrow(ax, 4.7, 2.6, 5.1, 2.6)
    arrow(ax, 7.5, 2.6, 7.9, 2.6)
    arrow(ax, 10.3, 2.6, 10.7, 2.6)

    ax.text(6.5, 0.7, "11.1M parameters  ·  28.5 GFLOPs  ·  7.3 ms/image inference (RTX 4060)",
            ha="center", fontsize=9.5, style="italic", color=GREY)

    plt.tight_layout()
    plt.savefig(OUT / "architecture.png", dpi=170, bbox_inches="tight")
    plt.close()
    print("saved architecture.png")


if __name__ == "__main__":
    methodology_diagram()
    architecture_diagram()
    print("Diagrams written to", OUT)
