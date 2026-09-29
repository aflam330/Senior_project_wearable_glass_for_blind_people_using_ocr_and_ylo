"""Late-fusion baseline on the same JaalTaka test split.

Score-level late fusion: each of the first k views is classified on its own (1-view input to the
same PRMVT network), and the k genuine probabilities are averaged. This is the standard
"average the per-view predictions" baseline for multi-view classification; it needs no training.
Compared with PRMVT's own k-view (feature-level) fusion on the same notes.

Output: results/sota/late_fusion_seed{42,43,44}.json
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
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE
from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import load_qduig

TF = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                         transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


@torch.inference_mode()
def main() -> None:
    splits, records = load_splits()
    test = list(splits["test"])
    y = np.array([int(records[n]["label"]) for n in test])
    for seed in (42, 43, 44):
        run = ROOT / "results" / "qduig" / "prefix_ft" / f"seed{seed}"
        model = load_qduig(run / "checkpoint.pt", load_config(run / "config.yaml")).to(DEVICE).eval()
        single = np.zeros((len(test), 6))   # per-view 1-view probability
        joint = np.zeros((len(test), 6))    # PRMVT k-view probability
        for i, nid in enumerate(test):
            views = torch.stack([TF(Image.fromarray(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)))
                                 for p in records[nid]["view_paths"][:6]]).to(DEVICE)
            for j in range(6):
                v = views[j:j + 1].unsqueeze(0)
                single[i, j] = float(torch.softmax(model(v, torch.ones(1, 1, dtype=torch.long, device=DEVICE))["logits"], 1)[0, 1])
                vk = views[: j + 1].unsqueeze(0)
                joint[i, j] = float(torch.softmax(model(vk, torch.ones(1, j + 1, dtype=torch.long, device=DEVICE))["logits"], 1)[0, 1])
        res = {"seed": seed, "n": len(test), "late_fusion_mean_prob": {}, "prmvt_feature_fusion": {},
               "best_single_view_index_acc": {}}
        for k in range(1, 7):
            lf = single[:, :k].mean(1)
            res["late_fusion_mean_prob"][k] = float(((lf >= 0.5) == (y == 1)).mean())
            res["prmvt_feature_fusion"][k] = float(((joint[:, k - 1] >= 0.5) == (y == 1)).mean())
        for j in range(6):
            res["best_single_view_index_acc"][j] = float(((single[:, j] >= 0.5) == (y == 1)).mean())
        dst = ROOT / "results" / "sota"
        dst.mkdir(parents=True, exist_ok=True)
        (dst / f"late_fusion_seed{seed}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(seed, "late", res["late_fusion_mean_prob"], "prmvt", res["prmvt_feature_fusion"], flush=True)


if __name__ == "__main__":
    main()
