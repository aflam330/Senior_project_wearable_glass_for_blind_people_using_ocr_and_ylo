"""0.5-threshold accuracy after calibrators fit on validation.

Temperature scaling and the entropy map keep the sign of the logit, so they
cannot change a 0.5 decision. HER can, because the quality term is additive.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roboeye.camva.engine import make_loader
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE
from roboeye.qduig.calibration import HERCalibrator, binary_entropy, entropy_mapped_confidence, fit_temperature
from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import load_qduig
from roboeye.qduig.quality import image_quality_factors

OUT = ROOT / "results" / "calibration" / "accuracy_seed42.json"


def _softmax_p(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


def _acc(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p >= 0.5).astype(np.int64) == y))


def main() -> None:
    splits, records = load_splits()
    model = load_qduig(
        ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "checkpoint.pt",
        load_config(ROOT / "configs" / "proposed_prefix_ft.yaml"),
    )
    model.eval()

    def collect(split: str) -> dict[str, np.ndarray]:
        loader = make_loader(splits[split], records, n_views=6, train=False, batch=8, workers=0)
        ys, logits, qual = [], [], []
        for batch in loader:
            views = batch["views"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            with torch.inference_mode():
                logit = model(views, mask)["logits"]
                factors = image_quality_factors(views)
            ys.append(batch["label"].numpy())
            logits.append(logit.detach().cpu().numpy())
            qual.append(factors.mean(dim=(1, 2)).detach().cpu().numpy())
        return {"y": np.concatenate(ys), "logits": np.concatenate(logits), "quality": np.concatenate(qual)}

    val = collect("val")
    test = collect("test")
    temperature = fit_temperature(val["logits"], val["y"])
    her = HERCalibrator()
    her.fit(val["logits"], val["y"], val["quality"])
    raw = _softmax_p(test["logits"])
    scaled = 1.0 / (1.0 + np.exp(-(np.log(np.clip(raw, 1e-8, 1 - 1e-8) / np.clip(1 - raw, 1e-8, 1)) / max(temperature, 1e-6))))
    ent = entropy_mapped_confidence(test["logits"])
    her_p = her.transform(test["logits"], test["quality"], binary_entropy(raw))
    payload = {
        "split": "test",
        "n": int(len(test["y"])),
        "views": 6,
        "threshold": 0.5,
        "temperature_fit_on": "val",
        "her_fit_on": "val",
        "accuracy": {
            "raw": _acc(test["y"], raw),
            "temperature": _acc(test["y"], scaled),
            "entropy": _acc(test["y"], ent),
            "her": _acc(test["y"], her_p),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["accuracy"], indent=2))


if __name__ == "__main__":
    main()
