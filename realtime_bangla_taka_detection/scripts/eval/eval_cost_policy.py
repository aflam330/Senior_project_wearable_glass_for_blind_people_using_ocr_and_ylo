"""Evaluate the saved PRMVT checkpoint with the policy cost weight on and off.

Lambda is the validation value from the prefix fine-tune config, and zero.
It is not chosen on the test split.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from roboeye.camva.notes import load_splits
from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import load_qduig, sequential_predict
from roboeye.qduig.policy import CostWeights

PAPER = ROOT.parent / "paper_evidence"


def main() -> None:
    splits, records = load_splits()
    model = load_qduig(
        ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "checkpoint.pt",
        load_config(ROOT / "configs" / "proposed_prefix_ft.yaml"),
    )
    out = {}
    for lamb, name in ((0.02, "with_cost"), (0.0, "no_cost")):
        rows = sequential_predict(
            model, splits["test"], records,
            policy="full_proposed",
            cost=CostWeights(alpha=1.0, beta=0.05, gamma=0.0, lamb=lamb),
            seed=42,
        )
        y = np.array(rows["y_true"])
        p = np.array(rows["genuine_score"])
        nviews = np.array(rows["n_views"], dtype=float)
        out[name] = {
            "lambda": lamb,
            "accuracy": float(((p >= 0.5).astype(int) == y).mean()),
            "mean_views": float(nviews.mean()),
            "n": int(len(y)),
        }
        if name == "with_cost":
            order = np.argsort(-(p >= 0.5).astype(int) == y)  # correct first is not sorted by accuracy of the note
            correct = ((p >= 0.5).astype(int) == y).astype(int)
            order = np.argsort(-correct)
            mat = np.zeros((len(y), 6), dtype=float)
            for i, sel in enumerate(rows["selected_indices"]):
                for j in sel:
                    if j < 6:
                        mat[i, j] = 1.0
            mat = mat[order]
            fig, ax = plt.subplots(figsize=(4.2, 5.0))
            ax.imshow(mat, aspect="auto", cmap="cividis", interpolation="nearest")
            ax.set_xlabel("View index")
            ax.set_ylabel("Test note, correct above incorrect")
            fig.savefig(PAPER / "figures" / "fig7_view_selection.png", dpi=300, bbox_inches="tight")
            fig.savefig(PAPER / "figures" / "fig7_view_selection.pdf", bbox_inches="tight")
            plt.close(fig)
        print(name, out[name], flush=True)
    path = ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "cost_policy_test.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
