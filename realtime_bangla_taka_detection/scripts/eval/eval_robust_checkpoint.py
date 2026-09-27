"""6-view accuracy of a checkpoint under the saved low-light and occlusion settings."""
from __future__ import annotations

import argparse
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
from roboeye.camva.engine import make_loader
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE
from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import load_qduig

_RESIZE = transforms.Resize((IMG_SIZE, IMG_SIZE))
_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _clean(model, ids, records, k: int) -> float:
    loader = make_loader(ids, records, n_views=6, train=False, batch=8, workers=0)
    correct = 0
    n = 0
    for batch in loader:
        views = batch["views"][:, :k].to(DEVICE)
        mask = torch.ones(views.size(0), k, dtype=torch.long, device=DEVICE)
        with torch.inference_mode():
            prob = torch.softmax(model(views, mask)["logits"], dim=1)[:, 1]
        pred = (prob >= 0.5).detach().cpu().numpy()
        y = batch["label"].numpy()
        correct += int(np.sum(pred == y))
        n += len(y)
    return correct / n


def _corrupt(model, ids, records, labels, name: str, severity: float) -> float:
    correct = 0
    for start in range(0, len(ids), 4):
        batch = []
        chunk = ids[start:start + 4]
        for i, nid in enumerate(chunk):
            if name == "occlusion":
                np.random.seed(1000 + start + i)
            frames = []
            for path in records[nid]["view_paths"]:
                bgr = cv2.imread(str(path))
                frames.append(apply_corruption(bgr, name, severity))
            rgb = [cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in frames]
            batch.append(torch.stack([_TENSOR(_RESIZE(Image.fromarray(x))) for x in rgb], dim=0))
        views = torch.stack(batch, dim=0).to(DEVICE)
        mask = torch.ones(views.size(0), views.size(1), dtype=torch.long, device=DEVICE)
        with torch.inference_mode():
            prob = torch.softmax(model(views, mask)["logits"], dim=1)[:, 1]
        pred = (prob >= 0.5).detach().cpu().numpy()
        correct += int(np.sum(pred == labels[start:start + len(pred)]))
    return correct / len(ids)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    splits, records = load_splits()
    model = load_qduig(Path(args.checkpoint), load_config(args.config))
    model.eval()
    ids = list(splits["test"])
    labels = np.array([int(records[n]["label"]) for n in ids])
    payload = {
        "checkpoint": args.checkpoint,
        "n_test": len(ids),
        "clean_1view": _clean(model, ids, records, 1),
        "clean_6view": _clean(model, ids, records, 6),
        "low_light_0.2_6view": _corrupt(model, ids, records, labels, "low_light", 0.2),
        "occlusion_0.55_6view": _corrupt(model, ids, records, labels, "occlusion", 0.55),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
