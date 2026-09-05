"""Generate a clean per-epoch training results plot from results.csv."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV = Path("runs/detect/train-4/results.csv")
OUT = Path("results/training_v2/epoch_results.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
df.columns = df.columns.str.strip()

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("YOLOv8s Training Results — Bangla Currency (v2, Real Backgrounds)", fontsize=13)

plots = [
    ("train/box_loss",        "Box Loss (train)",       axes[0, 0], "tab:blue"),
    ("train/cls_loss",        "Class Loss (train)",     axes[0, 1], "tab:orange"),
    ("train/dfl_loss",        "DFL Loss (train)",       axes[0, 2], "tab:red"),
    ("metrics/mAP50(B)",      "mAP@50",                 axes[1, 0], "tab:green"),
    ("metrics/mAP50-95(B)",   "mAP@50-95",              axes[1, 1], "tab:purple"),
    ("metrics/precision(B)",  "Precision & Recall",     axes[1, 2], "tab:cyan"),
]

for col, title, ax, color in plots:
    if col == "metrics/precision(B)":
        ax.plot(df["epoch"], df["metrics/precision(B)"], label="Precision", color="tab:cyan")
        ax.plot(df["epoch"], df["metrics/recall(B)"],    label="Recall",    color="tab:pink")
        ax.legend(fontsize=8)
    else:
        ax.plot(df["epoch"], df[col], color=color)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"Saved to {OUT}")
