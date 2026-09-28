"""Classify occlusion 0.55 after the trained reconstructor fills the box.

One test pass. The reconstructor checkpoint was chosen on validation hole-L1.
"""
from __future__ import annotations

import importlib.util
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
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE
from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import load_qduig
_spec = importlib.util.spec_from_file_location("train_occlusion_inpaint", ROOT / "scripts" / "train" / "train_occlusion_inpaint.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Reconstructor = _mod.Reconstructor

INPAINT = ROOT / "results" / "robustness" / "occlusion_inpaint" / "seed42" / "checkpoint.pt"
CLS = ROOT / "results" / "qduig" / "occlusion_ft" / "seed42" / "checkpoint.pt"
CFG = ROOT / "configs" / "occlusion_ft.yaml"
OUT = ROOT / "results" / "robustness" / "occlusion_inpaint_seed42.json"
_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def _occlude(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = bgr.shape[:2]
    frac = 0.55
    rh, rw = max(1, int(h * frac ** 0.5)), max(1, int(w * frac ** 0.5))
    y0 = np.random.randint(0, max(h - rh, 1))
    x0 = np.random.randint(0, max(w - rw, 1))
    out = bgr.copy()
    mask = np.zeros((h, w), np.uint8)
    out[y0:y0 + rh, x0:x0 + rw] = 0
    mask[y0:y0 + rh, x0:x0 + rw] = 1
    return out, mask


def main() -> None:
    splits, records = load_splits()
    ids = list(splits["test"])
    labels = np.array([int(records[nid]["label"]) for nid in ids])
    net = Reconstructor().to(DEVICE)
    try:
        blob = torch.load(INPAINT, map_location=DEVICE, weights_only=False)
    except TypeError:
        blob = torch.load(INPAINT, map_location=DEVICE)
    net.load_state_dict(blob["model"])
    net.eval()
    clf = load_qduig(CLS, load_config(CFG))
    clf.eval()
    correct = 0
    for start in range(0, len(ids), 4):
        chunk = ids[start:start + 4]
        batch = []
        for i, nid in enumerate(chunk):
            np.random.seed(1000 + start + i)
            frames = []
            for path in records[nid]["view_paths"]:
                bgr, mask = _occlude(cv2.imread(str(path)))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                m = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
                hole = rgb.astype(np.float32) / 255.0
                hole[m > 0] = 0.0
                x = torch.from_numpy(np.concatenate([hole.transpose(2, 0, 1), m[None].astype(np.float32)], axis=0))
                with torch.inference_mode():
                    pred = net(x[None].to(DEVICE))[0].detach().cpu().numpy().transpose(1, 2, 0)
                filled = hole.copy()
                filled[m > 0] = pred.clip(0, 1)[m > 0]
                frames.append(_TENSOR(Image.fromarray((filled * 255).astype(np.uint8))))
            batch.append(torch.stack(frames, dim=0))
        views = torch.stack(batch, dim=0).to(DEVICE)
        mask_v = torch.ones(views.size(0), views.size(1), dtype=torch.long, device=DEVICE)
        with torch.inference_mode():
            prob = torch.softmax(clf(views, mask_v)["logits"], dim=1)[:, 1]
        pred_y = (prob >= 0.5).detach().cpu().numpy()
        correct += int(np.sum(pred_y == labels[start:start + len(pred_y)]))
        print(f"notes {min(start + 4, len(ids))}/{len(ids)}", flush=True)
    acc = correct / len(ids)
    payload = {
        "split": "test",
        "n": len(ids),
        "occlusion": 0.55,
        "repair": "learned reconstructor, trained on train notes, selected by validation hole-L1",
        "classifier": str(CLS),
        "accuracy_6view": acc,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"accuracy_6view": acc}), flush=True)


if __name__ == "__main__":
    main()
