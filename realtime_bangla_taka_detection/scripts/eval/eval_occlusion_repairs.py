"""Inference repairs for occlusion 0.55. No training. Test labels are not used to choose a method."""
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
OUT = ROOT / "results" / "robustness" / "occlusion_repairs_seed42.json"


def _repair(bgr: np.ndarray, kind: str) -> np.ndarray:
    if kind == "identity":
        return bgr
    mask = (bgr.sum(axis=2) == 0).astype(np.uint8) * 255
    if int(mask.sum()) == 0:
        return bgr
    if kind == "inpaint":
        return cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    fill = np.median(bgr[mask == 0], axis=0) if int((mask == 0).sum()) else np.array([0, 0, 0])
    out = bgr.copy()
    out[mask > 0] = fill
    return out


def _score(model, views: torch.Tensor, k: int) -> torch.Tensor:
    x = views[:, :k]
    mask = torch.ones(x.size(0), k, dtype=torch.long, device=DEVICE)
    with torch.inference_mode():
        return torch.softmax(model(x, mask)["logits"], dim=1)[:, 1]


def main() -> None:
    splits, records = load_splits()
    ids = list(splits["test"])
    labels = {nid: int(records[nid]["label"]) for nid in ids}
    jobs = [
        ("prefix_ft", ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "checkpoint.pt", ROOT / "configs" / "proposed_prefix_ft.yaml"),
        ("occlusion_ft", ROOT / "results" / "qduig" / "occlusion_ft" / "seed42" / "checkpoint.pt", ROOT / "configs" / "occlusion_ft.yaml"),
    ]
    kinds = ("identity", "inpaint", "median_fill")
    correct = {name: {kind: [0] * 6 for kind in kinds} for name, _, _ in jobs}
    models = []
    for name, ckpt, cfg in jobs:
        model = load_qduig(ckpt, load_config(cfg))
        model.eval()
        models.append((name, model))
    for start in range(0, len(ids), 4):
        chunk = ids[start:start + 4]
        packed = {kind: [] for kind in kinds}
        for i, nid in enumerate(chunk):
            np.random.seed(1000 + start + i)
            raw = []
            for path in records[nid]["view_paths"]:
                bgr = cv2.imread(str(path))
                raw.append(apply_corruption(bgr, "occlusion", 0.55))
            for kind in kinds:
                frames = [_repair(b, kind) for b in raw]
                rgb = [cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in frames]
                packed[kind].append(torch.stack([_TENSOR(_RESIZE(Image.fromarray(x))) for x in rgb], dim=0))
        ys = np.array([labels[nid] for nid in chunk])
        for name, model in models:
            for kind in kinds:
                views = torch.stack(packed[kind], dim=0).to(DEVICE)
                for k in range(1, 7):
                    pred = (_score(model, views, k) >= 0.5).detach().cpu().numpy()
                    correct[name][kind][k - 1] += int(np.sum(pred == ys))
        print(f"notes {start + len(chunk)}/{len(ids)}", flush=True)
    payload = {"split": "test", "occlusion": 0.55, "seed_rule": "np.random.seed(1000+start+i) before the six views", "n": len(ids), "models": {}}
    for name, _, _ in jobs:
        payload["models"][name] = {
            kind: [c / len(ids) for c in correct[name][kind]] for kind in kinds
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["models"], indent=2), flush=True)


if __name__ == "__main__":
    main()
