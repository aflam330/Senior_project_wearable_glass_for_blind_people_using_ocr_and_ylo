"""6-view occlusion 0.55 repairs on the occlusion fine-tune. No training.

The clean-accuracy floor and the crop set are fixed here. Test labels are not used to choose a crop.
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
CKPT = ROOT / "results" / "qduig" / "occlusion_ft" / "seed42" / "checkpoint.pt"
CFG = ROOT / "configs" / "occlusion_ft.yaml"
OUT = ROOT / "results" / "robustness" / "occlusion_tta_seed42.json"


def _median_fill(bgr: np.ndarray) -> np.ndarray:
    mask = (bgr.sum(axis=2) == 0).astype(np.uint8)
    if int(mask.sum()) == 0:
        return bgr
    keep = mask == 0
    fill = np.median(bgr[keep], axis=0) if int(keep.sum()) else np.zeros(3)
    out = bgr.copy()
    out[mask > 0] = fill
    return out


def _crops(bgr: np.ndarray) -> list[np.ndarray]:
    h, w = bgr.shape[:2]
    rh, rw = int(h * 0.7), int(w * 0.7)
    boxes = [(0, 0, rh, rw), (0, w - rw, rh, w), (h - rh, 0, h, rw), (h - rh, w - rw, h, w)]
    return [bgr] + [bgr[y0:y1, x0:x1] for y0, x0, y1, x1 in boxes]


def _tensor(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _TENSOR(_RESIZE(Image.fromarray(rgb)))


def main() -> None:
    splits, records = load_splits()
    ids = list(splits["test"])
    labels = np.array([int(records[nid]["label"]) for nid in ids])
    model = load_qduig(CKPT, load_config(CFG))
    model.eval()
    names = ("median_full", "crop_mean", "crop_confident")
    correct = {name: 0 for name in names}
    for start in range(0, len(ids), 4):
        chunk = ids[start:start + 4]
        filled = []
        for i, nid in enumerate(chunk):
            np.random.seed(1000 + start + i)
            frames = []
            for path in records[nid]["view_paths"]:
                bgr = cv2.imread(str(path))
                frames.append(_median_fill(apply_corruption(bgr, "occlusion", 0.55)))
            filled.append(frames)
        probs = []
        for crop_i in range(5):
            batch = torch.stack([
                torch.stack([_tensor(_crops(frame)[crop_i]) for frame in frames], dim=0)
                for frames in filled
            ], dim=0).to(DEVICE)
            mask = torch.ones(batch.size(0), batch.size(1), dtype=torch.long, device=DEVICE)
            with torch.inference_mode():
                prob = torch.softmax(model(batch, mask)["logits"], dim=1)[:, 1]
            probs.append(prob)
        stacked = torch.stack(probs, dim=0)
        ys = labels[start:start + len(chunk)]
        full = (stacked[0] >= 0.5).detach().cpu().numpy()
        mean = (stacked.mean(dim=0) >= 0.5).detach().cpu().numpy()
        conf = (stacked - 0.5).abs()
        pick = conf.argmax(dim=0)
        chosen = stacked[pick, torch.arange(stacked.size(1), device=DEVICE)]
        vote = (chosen >= 0.5).detach().cpu().numpy()
        correct["median_full"] += int(np.sum(full == ys))
        correct["crop_mean"] += int(np.sum(mean == ys))
        correct["crop_confident"] += int(np.sum(vote == ys))
        print(f"notes {start + len(chunk)}/{len(ids)}", flush=True)
    payload = {
        "split": "test",
        "checkpoint": str(CKPT),
        "occlusion": 0.55,
        "repair": "median fill, then full frame and four 70% corner crops",
        "n": len(ids),
        "accuracy_6view": {name: correct[name] / len(ids) for name in names},
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["accuracy_6view"], indent=2), flush=True)


if __name__ == "__main__":
    main()
