"""Save PRMVT probabilities on val and test, clean and 55%-occluded, with and without a
horizontal-flip average. Decisions (flip averaging, seed ensemble, rejection threshold) are
made later on the val arrays only; see scripts/eval/occlusion_decide.py.

Occlusion follows the test protocol: per note index i, np.random.seed(1000 + i), then one
55% black box per view (apply_corruption). Output: results/robustness/occ_probs/<name>.npz
"""
from __future__ import annotations

import argparse
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

TF = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                         transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def note_views(records, nid, i, occ):
    if occ:
        np.random.seed(1000 + i)
    out = []
    for p in records[nid]["view_paths"]:
        bgr = cv2.imread(str(p))
        if occ:
            bgr = apply_corruption(bgr, "occlusion", 0.55)
        out.append(TF(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))))
    return torch.stack(out)


@torch.inference_mode()
def probs(model, records, ids, occ):
    plain, flip = [], []
    for i, nid in enumerate(ids):
        v = note_views(records, nid, i, occ).unsqueeze(0).to(DEVICE)
        m = torch.ones(1, v.size(1), dtype=torch.long, device=DEVICE)
        plain.append(float(torch.softmax(model(v, m)["logits"], 1)[0, 1]))
        flip.append(float(torch.softmax(model(torch.flip(v, dims=[4]), m)["logits"], 1)[0, 1]))
    return np.array(plain), np.array(flip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="folder with checkpoint.pt and config.yaml")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    splits, records = load_splits()
    run = Path(args.run)
    model = load_qduig(run / "checkpoint.pt", load_config(run / "config.yaml")).to(DEVICE).eval()
    arrays = {}
    for split in ("val", "test"):
        ids = list(splits[split])
        arrays[f"{split}_y"] = np.array([int(records[n]["label"]) for n in ids])
        for occ in (False, True):
            p, f = probs(model, records, ids, occ)
            tag = f"{split}_{'occ' if occ else 'clean'}"
            arrays[tag], arrays[tag + "_flip"] = p, f
            print(args.name, tag, "acc", round(float(((p >= .5) == arrays[f'{split}_y']).mean()), 4), flush=True)
    dst = ROOT / "results/robustness/occ_probs"
    dst.mkdir(parents=True, exist_ok=True)
    np.savez(dst / f"{args.name}.npz", **arrays)


if __name__ == "__main__":
    main()
