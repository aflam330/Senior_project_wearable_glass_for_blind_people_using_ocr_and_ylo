"""One test pass of the validation-selected occlusion checkpoint.

Reports clean 6-view, black-box occlusion 0.55, and median-fill occlusion 0.55.
The occlusion seed matches the severity sweep. Test labels are not used to choose weights.
"""
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

_RESIZE = transforms.Resize((IMG_SIZE, IMG_SIZE))
_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
CKPT = ROOT / "results" / "qduig" / "occlusion_matched" / "seed42" / "checkpoint.pt"
CFG = ROOT / "configs" / "occlusion_ft.yaml"
OUT = ROOT / "results" / "robustness" / "occlusion_matched_seed42.json"


def _median_fill(bgr: np.ndarray) -> np.ndarray:
    hole = bgr.sum(axis=2) == 0
    if not np.any(hole):
        return bgr
    out = bgr.copy()
    out[hole] = np.median(bgr[~hole], axis=0)
    return out


def _tensor(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _TENSOR(_RESIZE(Image.fromarray(rgb)))


def _accuracy(model, records, ids, labels, mode: str) -> float:
    correct = 0
    for start in range(0, len(ids), 4):
        chunk = ids[start:start + 4]
        batch = []
        for i, nid in enumerate(chunk):
            frames = []
            if mode != "clean":
                np.random.seed(1000 + start + i)
            for path in records[nid]["view_paths"]:
                bgr = cv2.imread(str(path))
                if mode != "clean":
                    bgr = apply_corruption(bgr, "occlusion", 0.55)
                if mode == "median":
                    bgr = _median_fill(bgr)
                frames.append(_tensor(bgr))
            batch.append(torch.stack(frames, dim=0))
        views = torch.stack(batch, dim=0).to(DEVICE)
        mask = torch.ones(views.size(0), views.size(1), dtype=torch.long, device=DEVICE)
        with torch.inference_mode():
            prob = torch.softmax(model(views, mask)["logits"], dim=1)[:, 1]
        pred = (prob >= 0.5).detach().cpu().numpy()
        correct += int(np.sum(pred == labels[start:start + len(pred)]))
        print(f"{mode} {min(start + 4, len(ids))}/{len(ids)}", flush=True)
    return correct / len(ids)


def main() -> None:
    splits, records = load_splits()
    ids = list(splits["test"])
    labels = np.array([int(records[nid]["label"]) for nid in ids])
    model = load_qduig(CKPT, load_config(CFG))
    model.eval()
    scores = {
        "clean_6": _accuracy(model, records, ids, labels, "clean"),
        "black_box_6": _accuracy(model, records, ids, labels, "black"),
        "median_fill_6": _accuracy(model, records, ids, labels, "median"),
    }
    payload = {
        "split": "test",
        "n": len(ids),
        "checkpoint": str(CKPT),
        "occlusion": 0.55,
        "seed_rule": "np.random.seed(1000 + start + i) once per note, then each view",
        "accuracy_6view": scores,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(scores), flush=True)


if __name__ == "__main__":
    main()
