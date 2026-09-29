"""Literature-backbone baselines on the note-disjoint JaalTaka split (frozen-feature linear probes).

The JaalTaka data article reports pretrained CNNs (ResNet50, InceptionV3, DenseNet121, VGG16,
MobileNetV2); its numbers and split were not retrievable here, so they are not compared. This
script puts the same five ImageNet backbones on THIS project's split:
  - frozen ImageNet features per view (224 px; 299 px for InceptionV3), L2-normalised
  - one logistic regression per backbone, fitted on single views of TRAIN notes
  - C chosen on VAL notes by mean accuracy over first-k late fusion, k = 1..6
  - k-view prediction = mean of the per-view genuine probabilities of views 1..k
  - TEST scored once with the chosen C
Frozen probes are weaker than full fine-tuning; they are labelled as probes, not reproductions.

Output: results/sota/backbone_probes.json (+ cached features in results/sota/feats_<name>.npz)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from roboeye.camva.notes import load_splits  # noqa: E402

OUT = ROOT / "results" / "sota"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def backbone(name: str):
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2); m.fc = torch.nn.Identity(); size = 224
    elif name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1); m.classifier = torch.nn.Identity(); size = 224
    elif name == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1); m.classifier = m.classifier[:-1]; size = 224
    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2); m.classifier = torch.nn.Identity(); size = 224
    elif name == "inception_v3":
        m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1); m.fc = torch.nn.Identity(); size = 299
    else:
        raise ValueError(name)
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return m.eval().to(DEV), tf


@torch.inference_mode()
def features(name: str, records, ids: list[str]) -> np.ndarray:
    cache = OUT / f"feats_{name}.npz"
    if cache.is_file():
        d = np.load(cache, allow_pickle=True)
        if list(d["ids"]) == ids:
            return d["x"]
    model, tf = backbone(name)
    feats = np.zeros((len(ids), 6, 0), dtype=np.float32)
    rows = []
    for nid in ids:
        views = [tf(Image.fromarray(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)))
                 for p in records[nid]["view_paths"][:6]]
        z = model(torch.stack(views).to(DEV)).float().cpu().numpy()
        rows.append(z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8))
    feats = np.stack(rows).astype(np.float32)  # (notes, 6, d)
    np.savez_compressed(cache, x=feats, ids=np.array(ids))
    return feats


def late_fusion_acc(clf, x: np.ndarray, y: np.ndarray) -> dict[int, float]:
    n, v, d = x.shape
    p = clf.predict_proba(x.reshape(n * v, d))[:, 1].reshape(n, v)
    return {k: float(((p[:, :k].mean(1) >= 0.5) == (y == 1)).mean()) for k in range(1, 7)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    splits, records = load_splits()
    ids = {s: list(splits[s]) for s in ("train", "val", "test")}
    ys = {s: np.array([int(records[n]["label"]) for n in ids[s]]) for s in ids}
    all_ids = ids["train"] + ids["val"] + ids["test"]
    results = {}
    for name in ("resnet50", "densenet121", "vgg16", "mobilenet_v2", "inception_v3"):
        print("features", name, flush=True)
        x = features(name, records, all_ids)
        ntr, nva = len(ids["train"]), len(ids["val"])
        xs = {"train": x[:ntr], "val": x[ntr:ntr + nva], "test": x[ntr + nva:]}
        d = x.shape[2]
        xtr = xs["train"].reshape(-1, d)
        ytr = np.repeat(ys["train"], 6)
        best = None
        for c in (0.01, 0.1, 1.0, 10.0, 100.0):
            clf = LogisticRegression(C=c, max_iter=3000).fit(xtr, ytr)
            va = late_fusion_acc(clf, xs["val"], ys["val"])
            score = float(np.mean(list(va.values())))
            if best is None or score > best[0]:
                best = (score, c, clf, va)
        _, c, clf, va = best
        te = late_fusion_acc(clf, xs["test"], ys["test"])  # scored once
        results[name] = {"C_chosen_on_val": c, "val_acc": va, "test_acc": te, "feature_dim": d}
        print(name, "C", c, "test", {k: round(v, 4) for k, v in te.items()}, flush=True)
    (OUT / "backbone_probes.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
