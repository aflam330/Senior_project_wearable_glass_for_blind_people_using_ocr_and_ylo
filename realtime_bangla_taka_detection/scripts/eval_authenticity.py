"""Measure authenticity accuracies on JaalTaka (note-level split) and BCCID.

Writes results/authenticity_metrics.json with the numbers that belong in the
paper — whatever this run actually scores. Does not invent 96.7 / 98.1 / 96.5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from roboeye.authenticity import (
    AUTH_WEIGHTS,
    JaalTakaMultiView,
    MultiViewCNNVIT,
    bgr_to_views_tensor,
    list_jaaltaka_notes,
    split_notes,
    _view_paths,
)
from roboeye.clip_zero_shot import PrototypeAuthenticator
from roboeye.config import COUNTERFEIT_DIR, DEVICE, MODELS_DIR

OUT = ROOT / "results" / "authenticity_metrics.json"


def _acc(model, notes, n_views: int) -> float:
    if not notes:
        return 0.0
    ds = JaalTakaMultiView(notes, n_views=n_views, train=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for views, y in loader:
            pred = model(views.to(DEVICE)).argmax(1).cpu()
            correct += int((pred == y).sum())
            total += len(y)
    return correct / max(total, 1)


def list_bccid() -> list[tuple[Path, int]]:
    """Bangladeshi Counterfeit Currency Image Dataset: 1=genuine, 0=fake."""
    root = COUNTERFEIT_DIR / "Denomination-wise Distribution"
    if not root.is_dir():
        root = COUNTERFEIT_DIR
    items = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        name = str(p).lower()
        if "counterfeit" in name or "fake" in name:
            items.append((p, 0))
        elif "genuine" in name or "real" in name:
            items.append((p, 1))
    return items


@torch.inference_mode()
def cnn_vit_on_paths(model, items: list[tuple[Path, int]]) -> float:
    tf_notes = []
    # reuse MultiView path loader by wrapping single images as fake "note dirs"
    correct = total = 0
    for path, label in items:
        img = cv2.imread(str(path))
        if img is None:
            continue
        x = bgr_to_views_tensor([img], n_views=2).to(DEVICE)
        pred = int(model(x).argmax(1).item())
        correct += int(pred == label)
        total += 1
    return correct / max(total, 1)


def proto_on_notes(proto: PrototypeAuthenticator, notes) -> float:
    correct = total = 0
    for note_dir, label in notes:
        paths = _view_paths(note_dir)
        if not paths:
            continue
        img = cv2.imread(str(paths[0]))
        if img is None:
            continue
        pred = proto.predict(img)
        yhat = 1 if pred["label"] == "genuine" else 0
        correct += int(yhat == label)
        total += 1
    return correct / max(total, 1)


def proto_on_paths(proto: PrototypeAuthenticator, items) -> float:
    correct = total = 0
    for path, label in items:
        img = cv2.imread(str(path))
        if img is None:
            continue
        pred = proto.predict(img)
        yhat = 1 if pred["label"] == "genuine" else 0
        correct += int(yhat == label)
        total += 1
    return correct / max(total, 1)


def ensemble_on_notes(model, proto, notes) -> float:
    correct = total = 0
    model.eval()
    for note_dir, label in notes:
        paths = _view_paths(note_dir)
        if not paths:
            continue
        crops = []
        for p in paths[:6]:
            im = cv2.imread(str(p))
            if im is not None:
                crops.append(im)
        if not crops:
            continue
        x = bgr_to_views_tensor(crops[:2], n_views=2).to(DEVICE)
        with torch.inference_mode():
            p_cnn = float(F.softmax(model(x), dim=1)[0, 1])
        p_clip = proto.predict(crops[0])["genuine_prob"]
        genuine = (p_cnn + p_clip) / 2.0
        yhat = 1 if genuine >= 0.5 else 0
        correct += int(yhat == label)
        total += 1
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    notes = list_jaaltaka_notes()
    splits = split_notes(notes)
    print(f"JaalTaka notes {len(notes)}  train {len(splits['train'])} val {len(splits['val'])} test {len(splits['test'])}")

    if not args.skip_train:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "train_authenticity", ROOT / "scripts" / "train_authenticity.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        class A:
            epochs = 4
            batch = 8
            views = 2
            lr = 1e-3
            max_notes = 0
            quick = False

        print("Retraining CNN+ViT on full JaalTaka note split...")
        mod.train_cnn_vit(A())

    model = MultiViewCNNVIT(freeze_cnn=True).to(DEVICE)
    blob = torch.load(AUTH_WEIGHTS, map_location=DEVICE, weights_only=True)
    model.load_state_dict(blob["model"])
    model.eval()

    acc_1 = _acc(model, splits["test"], n_views=1)
    acc_2 = _acc(model, splits["test"], n_views=2)
    acc_6 = _acc(model, splits["test"], n_views=6)
    print(f"CNN+ViT test 1-view={acc_1:.4f}  2-view={acc_2:.4f}  6-view={acc_6:.4f}")

    print("Building CLIP/encoder prototypes from TRAIN notes only...")
    proto = PrototypeAuthenticator()
    # rebuild using train split only
    from roboeye.clip_zero_shot import PROTO_WEIGHTS
    buckets = {0: [], 1: []}
    for note_dir, label in splits["train"]:
        if len(buckets[label]) >= 120:
            continue
        views = _view_paths(note_dir)[:2]
        embs = []
        for vp in views:
            img = cv2.imread(str(vp))
            if img is None:
                continue
            embs.append(proto.encode_bgr(img))
        if embs:
            buckets[label].append(torch.stack(embs).mean(0))
    proto.fake_proto = F.normalize(torch.stack(buckets[0]).mean(0), dim=0)
    proto.real_proto = F.normalize(torch.stack(buckets[1]).mean(0), dim=0)
    torch.save(
        {
            "backend": proto.backend,
            "real": proto.real_proto,
            "fake": proto.fake_proto,
            "n_real": len(buckets[1]),
            "n_fake": len(buckets[0]),
        },
        PROTO_WEIGHTS,
    )
    proto.loaded = True
    acc_clip = proto_on_notes(proto, splits["test"])
    print(f"prototype ({proto.backend}) test={acc_clip:.4f}")

    acc_ens = ensemble_on_notes(model, proto, splits["test"])
    print(f"ensemble CNN+ViT+proto test={acc_ens:.4f}")

    bccid = list_bccid()
    acc_bccid_cnn = cnn_vit_on_paths(model, bccid) if bccid else None
    acc_bccid_clip = proto_on_paths(proto, bccid) if bccid else None
    print(f"BCCID n={len(bccid)} cnn={acc_bccid_cnn} proto={acc_bccid_clip}")

    metrics = {
        "n_jaaltaka_notes": len(notes),
        "n_test_notes": len(splits["test"]),
        "cnn_vit_1view": acc_1,
        "cnn_vit_2view": acc_2,
        "cnn_vit_6view": acc_6,
        "prototype_backend": proto.backend,
        "prototype_1view": acc_clip,
        "ensemble": acc_ens,
        "bccid_n": len(bccid),
        "bccid_cnn_vit": acc_bccid_cnn,
        "bccid_prototype": acc_bccid_clip,
        "paper_draft_claims": {"cnn_vit": 0.967, "multiview_or_ensemble": 0.981, "clip": 0.965},
        "note": (
            "These are measured values. Paper draft 96.7/98.1/96.5 must be replaced "
            "by cnn_vit_1view / cnn_vit_6view_or_ensemble / prototype_1view (or BCCID transfer)."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("wrote", OUT)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
