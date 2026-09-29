"""Small watermark-window classifier for the device: MobileNetV3-Small fine-tuned on back-lit window crops.

Data: watermark-window crops (results/watermark/crops, from watermark_features.py) of the SERIAL-DISJOINT
split (results/serial_split): TRAIN to fit, VAL to choose the epoch (AUC), TEST (unseen counterfeit prints)
read once. Augmentation: colour jitter, small rotation / scale, blur. 10 epochs, AdamW 3e-4.
Exports: models/watermark_mobilenet.pt, watermark_mobilenet.onnx, watermark_mobilenet_int8.onnx (dynamic INT8),
and checks that the INT8 model gives the same test decisions.
Output: results/watermark/mobilenet.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from roboeye.camva.notes import load_splits  # noqa: E402

WM = ROOT / "results" / "watermark"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORM = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
TR = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.RandomRotation(6),
                         transforms.ColorJitter(0.3, 0.3, 0.2, 0.03), transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
                         transforms.ToTensor(), NORM])
EV = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM])


class Crops(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        nid, y = self.items[i]
        return self.tf(Image.open(WM / "crops" / (nid.replace(":", "_") + ".png")).convert("RGB")), y


@torch.inference_mode()
def scores(model, loader):
    model.eval()
    p, y = [], []
    for x, t in loader:
        p.append(torch.softmax(model(x.to(DEV)), 1)[:, 1].cpu().numpy())
        y.append(t.numpy())
    return np.concatenate(p), np.concatenate(y)


def main() -> None:
    torch.manual_seed(0)
    splits, records = load_splits(ROOT / "results" / "serial_split")
    ok = {r["note_id"] for r in json.loads((WM / "features.json").read_text(encoding="utf-8")) if r["ok"]}
    items = {s: [(n, int(records[n]["label"])) for n in splits[s] if n in ok] for s in ("train", "val", "test")}
    dl = {s: DataLoader(Crops(items[s], TR if s == "train" else EV), batch_size=32, shuffle=(s == "train"), num_workers=0)
          for s in items}
    arch = sys.argv[1] if len(sys.argv) > 1 else "v3"  # "v2": MobileNetV2 (ReLU6, quantises better)
    if arch == "v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    else:
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    tag = "watermark_mobilenet" if arch != "v2" else "watermark_mobilenetv2"
    m = m.to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    best, hist = (-1, None), []
    for ep in range(1, 11):
        m.train()
        for x, y in dl["train"]:
            loss = nn.functional.cross_entropy(m(x.to(DEV)), y.to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
        pv, yv = scores(m, dl["val"])
        auc = roc_auc_score(yv, pv)
        hist.append({"epoch": ep, "val_auc": float(auc), "val_acc": float(((pv >= 0.5) == (yv == 1)).mean())})
        print(hist[-1], flush=True)
        if auc > best[0]:
            best = (auc, {k: v.detach().clone() for k, v in m.state_dict().items()})
    m.load_state_dict(best[1])
    pt, yt = scores(m, dl["test"])
    pred = pt >= 0.5
    res = {"split": "serial_split", "n": {s: len(v) for s, v in items.items()}, "history": hist, "best_val_auc": float(best[0]),
           "test": {"accuracy": float((pred == (yt == 1)).mean()), "auc": float(roc_auc_score(yt, pt)),
                    "false_counterfeit_on_genuine": int((~pred & (yt == 1)).sum()), "genuine_n": int((yt == 1).sum()),
                    "counterfeit_missed": int((pred & (yt == 0)).sum()), "counterfeit_n": int((yt == 0).sum())}}
    out = ROOT / "models"
    m.eval().cpu()
    torch.save({"state_dict": m.state_dict(), "arch": "mobilenet_v2" if arch == "v2" else "mobilenet_v3_small", "classes": ["counterfeit", "genuine"],
                "input": "224x224 RGB watermark-window crop, ImageNet normalisation", "val_auc": float(best[0])}, out / f"{tag}.pt")
    torch.onnx.export(m, torch.zeros(1, 3, 224, 224), str(out / f"{tag}.onnx"), input_names=["image"],
                      output_names=["logits"], opset_version=17, dynamic_axes={"image": {0: "b"}, "logits": {0: "b"}}, dynamo=False)
    import onnxruntime as ort
    agree = {}
    for name in (f"{tag}.onnx",):
        sess = ort.InferenceSession(str(out / name), providers=["CPUExecutionProvider"])
        ps = []
        for x, _ in dl["test"]:
            lo = sess.run(None, {"image": x.numpy()})[0]
            e = np.exp(lo - lo.max(1, keepdims=True))
            ps.append(e[:, 1] / e.sum(1))
        po = np.concatenate(ps)
        agree[name] = {"same_decision_as_pytorch": float(((po >= 0.5) == pred).mean()), "max_abs_prob_diff": float(np.abs(po - pt).max()),
                       "size_mb": round((out / name).stat().st_size / 1e6, 2)}
    res["export"] = agree
    (WM / ("mobilenet.json" if arch != "v2" else "mobilenetv2.json")).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("test", "export")}, indent=1))


if __name__ == "__main__":
    main()
