"""Brightness repair for low-light images. The repair is chosen on validation."""
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
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE
from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import load_qduig

SEVERITY = 0.2
OUT = ROOT / "results" / "robustness" / "lowlight_enhance_seed42.json"
_RESIZE = transforms.Resize((IMG_SIZE, IMG_SIZE))
_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _enhance(bgr: np.ndarray, name: str, param: float) -> np.ndarray:
    if name == "identity":
        return bgr
    if name == "gamma":
        x = np.clip(bgr.astype(np.float32) / 255.0, 0, 1)
        return np.clip(255.0 * np.power(x, param), 0, 255).astype(np.uint8)
    if name == "gain":
        return np.clip(bgr.astype(np.float32) * param, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = cv2.createCLAHE(clipLimit=float(param), tileGridSize=(8, 8)).apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def _views(images: list[np.ndarray]) -> torch.Tensor:
    frames = []
    for bgr in images:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(_TENSOR(_RESIZE(Image.fromarray(rgb))))
    return torch.stack(frames, dim=0)


def _accuracy(model, records, ids, labels, name: str, param: float) -> float:
    correct = 0
    for start in range(0, len(ids), 8):
        batch = []
        for nid in ids[start:start + 8]:
            frames = []
            for path in records[nid]["view_paths"]:
                bgr = cv2.imread(str(path))
                dark = apply_corruption(bgr, "low_light", SEVERITY)
                frames.append(_enhance(dark, name, param))
            batch.append(_views(frames))
        views = torch.stack(batch, dim=0).to(DEVICE)
        mask = torch.ones(views.size(0), views.size(1), dtype=torch.long, device=DEVICE)
        with torch.inference_mode():
            prob = torch.softmax(model(views, mask)["logits"], dim=1)[:, 1]
        pred = (prob >= 0.5).detach().cpu().numpy()
        y = labels[start:start + len(pred)]
        correct += int(np.sum(pred == y))
    return correct / len(ids)


def main() -> None:
    splits, records = load_splits()
    model = load_qduig(
        ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "checkpoint.pt",
        load_config(ROOT / "configs" / "proposed_prefix_ft.yaml"),
    )
    model.eval()
    options = [("identity", 1.0), ("gamma", 0.4), ("gamma", 0.6), ("gamma", 0.8), ("gain", 2.0), ("gain", 3.0), ("clahe", 2.0), ("clahe", 4.0)]
    val_y = np.array([int(records[n]["label"]) for n in splits["val"]])
    test_y = np.array([int(records[n]["label"]) for n in splits["test"]])
    val_rows = []
    for name, param in options:
        acc = _accuracy(model, records, list(splits["val"]), val_y, name, param)
        val_rows.append({"name": name, "param": param, "val_accuracy": acc})
        print("val", name, param, round(acc, 4), flush=True)
    chosen = max(val_rows, key=lambda row: row["val_accuracy"])
    test_acc = _accuracy(model, records, list(splits["test"]), test_y, chosen["name"], chosen["param"])
    payload = {
        "checkpoint": "results/qduig/prefix_ft/seed42/checkpoint.pt",
        "corruption": "low_light",
        "severity": SEVERITY,
        "views": 6,
        "selection": "highest validation accuracy; test scored once",
        "val": val_rows,
        "chosen": chosen,
        "test_accuracy": test_acc,
        "n_test": int(len(test_y)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"chosen": chosen, "test_accuracy": test_acc}, indent=2))


if __name__ == "__main__":
    main()
