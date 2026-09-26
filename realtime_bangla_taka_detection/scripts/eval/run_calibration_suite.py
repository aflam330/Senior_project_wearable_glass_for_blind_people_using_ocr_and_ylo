"""Fit calibrators on validation. Score them on test. Ten reliability bins."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from roboeye.camva.engine import load_baseline, make_loader
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE
from roboeye.qduig.calibration import (
    HERCalibrator,
    evaluate_calibration_suite,
    expected_calibration_error,
    fit_temperature,
)
from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import load_qduig
from roboeye.qduig.quality import image_quality_factors
from scripts.train.train_novel import BUILDERS, _load_checkpoint, _module

PAPER = ROOT.parent / "paper_evidence"
FIG = PAPER / "figures"


def _quality(views: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        factors = image_quality_factors(views)
    return factors.mean(dim=(1, 2)).detach().cpu().numpy()


def _collect(forward, loader) -> dict[str, np.ndarray]:
    ys, logits, qual = [], [], []
    for batch in loader:
        views = batch["views"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        logit = forward(views, mask)
        ys.append(batch["label"].numpy())
        logits.append(logit.detach().cpu().numpy())
        qual.append(_quality(views))
    return {"y": np.concatenate(ys), "logits": np.concatenate(logits), "quality": np.concatenate(qual)}


def _mc(forward, loader, n: int = 10) -> np.ndarray:
    acc = []
    for _ in range(n):
        probs = []
        for batch in loader:
            views = batch["views"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            logit = forward(views, mask)
            prob = torch.softmax(logit, dim=1)[:, 1]
            probs.append(prob.detach().cpu().numpy())
        acc.append(np.concatenate(probs))
    return np.mean(acc, axis=0)


def _enable_dropout(model) -> None:
    if hasattr(model, "enable_mc_dropout"):
        model.enable_mc_dropout()
        return
    model.eval()
    for mod in model.modules():
        if mod.__class__.__name__.startswith("Dropout"):
            mod.train()


def main() -> None:
    splits, records = load_splits()
    models = {}
    prmvt = ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "checkpoint.pt"
    if prmvt.is_file():
        models["prmvt"] = load_qduig(prmvt, load_config(ROOT / "configs" / "proposed_prefix_ft.yaml"))
    base = ROOT / "results" / "camva" / "checkpoints" / "baseline_cnnvit_seed42.pt"
    if base.is_file():
        models["baseline"] = load_baseline(base)
    for name, folder in (("ndal", ROOT / "results" / "novel_v2" / "ndal" / "seed42" / "checkpoint.pt"),
                         ("ugf", ROOT / "results" / "novel" / "ugf" / "seed42" / "checkpoint.pt")):
        if folder.is_file():
            net = BUILDERS[name]().to(DEVICE)
            state = _load_checkpoint(str(folder))["model"]
            module = _module(net)
            if name in {"ugf", "sfaq", "igcr", "ogpd"}:
                module.load_state_dict(state, strict=False)
            else:
                module.load_state_dict(state)
            net.eval()
            models[name] = net

    def forward_of(model):
        def _fwd(views, mask):
            if model.__class__.__name__ == "MultiViewCNNVIT":
                return model(views)
            return model(views, mask)["logits"]
        return _fwd

    report = {}
    for name, model in models.items():
        fwd = forward_of(model)
        val_loader = make_loader(splits["val"], records, n_views=6, train=False, batch=8, workers=0)
        test_loader = make_loader(splits["test"], records, n_views=6, train=False, batch=8, workers=0)
        model.eval()
        val = _collect(fwd, val_loader)
        test = _collect(fwd, test_loader)
        temperature = fit_temperature(val["logits"], val["y"])
        her = HERCalibrator()
        her.fit(val["logits"], val["y"], val["quality"])
        _enable_dropout(model)
        mc = _mc(fwd, test_loader, 10)
        model.eval()
        suite = evaluate_calibration_suite(test["logits"], test["y"], test["quality"], her, temperature, mc)
        # Recompute ECE at 10 bins so it matches the reliability diagram.
        from roboeye.qduig.calibration import apply_temperature, entropy_mapped_confidence
        probs = {
            "predictive_confidence": torch.softmax(torch.tensor(test["logits"]), dim=1).numpy()[:, 1],
            "temperature": apply_temperature(test["logits"], temperature)[:, 1],
            "entropy": entropy_mapped_confidence(test["logits"]),
            "her": her.transform(test["logits"], test["quality"]),
            "mc_dropout": mc,
        }
        for method, p in probs.items():
            suite[method]["ece_10"] = expected_calibration_error(test["y"], p, n_bins=10)
            suite[method]["reliability_10"] = suite[method]["reliability"]
        report[name] = {
            "temperature": temperature,
            "her_params": her.params,
            "fit_split": "val",
            "eval_split": "test",
            "n_test": int(len(test["y"])),
            "methods": suite,
        }
        print(name, {m: round(suite[m]["ece_10"], 4) for m in suite}, flush=True)

    out = ROOT / "results" / "calibration" / "suite_seed42.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Calibration, seed 42, 6 views",
        "",
        "Temperature and HER are fit on the validation split only. Metrics are on the test split. ECE uses 10 bins. MC dropout is 10 stochastic passes.",
        "",
        "| model | method | ECE | adaptive ECE | Brier | NLL |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, blob in report.items():
        for method, row in blob["methods"].items():
            lines.append(
                f"| {name} | {method} | {row['ece_10']:.4f} | {row['adaptive_ece']:.4f} | {row['brier']:.4f} | {row['nll']:.4f} |"
            )
    (PAPER / "CALIBRATION_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _fig6(report)
    print(PAPER / "CALIBRATION_RESULTS.md")


def _fig6(report: dict) -> None:
    # One panel per method, PRMVT if present else the first model.
    model = "prmvt" if "prmvt" in report else next(iter(report))
    methods = list(report[model]["methods"])
    fig, axes = plt.subplots(1, len(methods), figsize=(2.2 * len(methods), 2.3), sharey=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        rel = report[model]["methods"][method]["reliability"]["bins"]
        conf = np.array([row["confidence"] for row in rel if row["confidence"] is not None])
        acc = np.array([row["accuracy"] for row in rel if row["accuracy"] is not None])
        if len(conf) and len(acc):
            ax.plot([0, 1], [0, 1], color="#888888", lw=0.6)
            ax.plot(conf, acc, marker="o", ms=3, color="#0072B2")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(method.replace("_", "\n"), fontsize=6)
    axes[0].set_ylabel("Accuracy")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig6_calibration.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig6_calibration.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
