"""Accuracy at several severities. Notes are loaded in small batches so the full set is not held in RAM."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roboeye.authenticity import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from roboeye.camva.data import apply_corruption
from roboeye.camva.engine import load_baseline
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE
from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import load_qduig
from roboeye.qduig.metrics_ext import full_binary_report
from scripts.train.train_novel import BUILDERS, _load_checkpoint, _module

SWEEP = {
    "gaussian_blur": [1.0, 3.0, 6.0],
    "motion_blur": [3.0, 5.0, 9.0],
    "low_light": [0.6, 0.35, 0.2],
    "brightness": [0.7, 1.6, 2.2],
    "contrast": [0.6, 1.8, 2.5],
    "glare": [0.3, 0.65, 1.0],
    "occlusion": [0.1, 0.2, 0.55],
    "jpeg": [70.0, 30.0, 10.0],
    "rotation": [5.0, 20.0, 40.0],
    "perspective": [0.04, 0.1, 0.2],
    "scale": [0.85, 0.7, 0.5],
    "sensor_noise": [5.0, 12.0, 25.0],
}
_RESIZE = transforms.Resize((IMG_SIZE, IMG_SIZE))
_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
CHUNK = 4


def _acc(y, p) -> float:
    return float(full_binary_report(y, p)["accuracy"])


def _forward(model, views, kind: str) -> np.ndarray:
    probs = []
    for start in range(0, views.size(0), 4):
        v = views[start:start + 4].to(DEVICE)
        m = torch.ones(v.size(0), v.size(1), dtype=torch.long, device=DEVICE)
        with torch.inference_mode():
            if kind == "baseline":
                logit = model(v)
            else:
                logit = model(v, m)["logits"]
            probs.append(torch.softmax(logit, dim=1)[:, 1].detach().cpu().numpy())
        del v
    return np.concatenate(probs) if probs else np.zeros((0,), dtype=np.float32)


def _views_from_bgr(images: list[np.ndarray]) -> torch.Tensor:
    frames = []
    for bgr in images:
        rgb = bgr[:, :, ::-1]
        image = _RESIZE(Image.fromarray(rgb))
        frames.append(_TENSOR(image))
    return torch.stack(frames, dim=0)


def _read(records, nid: str) -> list[np.ndarray]:
    frames = []
    for path in records[nid]["view_paths"]:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        frames.append(image)
    return frames


def main() -> None:
    splits, records = load_splits()
    test = list(splits["test"])
    labels = np.array([int(records[n]["label"]) for n in test], dtype=np.int64)
    prmvt = load_qduig(
        ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "checkpoint.pt",
        load_config(ROOT / "configs" / "proposed_prefix_ft.yaml"),
    )
    ndal = BUILDERS["ndal"]().to(DEVICE)
    _module(ndal).load_state_dict(_load_checkpoint(str(ROOT / "results" / "novel_v2" / "ndal" / "seed42" / "checkpoint.pt"))["model"])
    ndal.eval()
    baseline = load_baseline(ROOT / "results" / "camva" / "checkpoints" / "baseline_cnnvit_seed42.pt")
    models = {"prmvt": prmvt, "ndal": ndal, "baseline": baseline}
    keys = [("clean", 0.0)] + [(name, float(sev)) for name, levels in SWEEP.items() for sev in levels]
    store = {key: {m: [] for m in models} for key in keys}
    for start in range(0, len(test), CHUNK):
        ids = test[start:start + CHUNK]
        raw = [_read(records, nid) for nid in ids]
        clean = torch.stack([_views_from_bgr(imgs) for imgs in raw], dim=0)
        for name, model in models.items():
            store[("clean", 0.0)][name].append(_forward(model, clean, "baseline" if name == "baseline" else "other"))
        del clean
        for corr, levels in SWEEP.items():
            for severity in levels:
                batch = []
                for i, imgs in enumerate(raw):
                    if corr == "occlusion":
                        np.random.seed(1000 + start + i)
                    batch.append(_views_from_bgr([apply_corruption(bgr, corr, severity) for bgr in imgs]))
                views = torch.stack(batch, dim=0)
                for name, model in models.items():
                    store[(corr, float(severity))][name].append(
                        _forward(model, views, "baseline" if name == "baseline" else "other")
                    )
                del views
        del raw
        print(f"notes {start + len(ids)}/{len(test)}", flush=True)
    clean_acc = {
        name: _acc(labels, np.concatenate(store[("clean", 0.0)][name]))
        for name in models
    }
    rows = []
    for corr, levels in SWEEP.items():
        for severity in levels:
            entry = {"corruption": corr, "severity": float(severity), "k": 6}
            for name in models:
                acc = _acc(labels, np.concatenate(store[(corr, float(severity))][name]))
                entry[name] = acc
                entry[f"{name}_drop"] = clean_acc[name] - acc
            rows.append(entry)
            print(corr, severity, {m: round(entry[m], 4) for m in models}, flush=True)
    out = ROOT / "results" / "robustness" / "severity_seed42.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"clean_6view": clean_acc, "rows": rows}, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
