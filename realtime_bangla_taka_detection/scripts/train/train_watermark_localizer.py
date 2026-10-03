"""Learned watermark localizer: finds the transmitted-light window in a back-lit photo without a template.

Replaces the template + SIFT + fixed-box step at inference time. A MobileNetV3-Small regresses the
four window corners (TL, TR, BR, BL, fractions of the photo) from the 320 x 320 photo; the window is
then cut with a perspective warp to 224 x 224 and scored by the published watermark MobileNetV2
(models/watermark_mobilenetv2.pt, unchanged, trained on registration crops).
Labels: watermark_localizer_labels.py (registration corners). Split: serial-disjoint (results/serial_split).
TRAIN fits, VAL picks the epoch (mean corner error), TEST read once per seed.
Honest scope: labels come from the registration box, so the localizer learns where that box is.
It removes the template and SIFT from the device path and works on photos where SIFT fails;
it does not discover a window definition on its own, and it is one currency.
Usage: python scripts/train/train_watermark_localizer.py <seed> [labels_dir]
  labels_dir: output of scripts/dataset/manifest_to_localizer_labels.py (a new print-disjoint dataset,
  any currency). Then the SIFT comparison is skipped and crops are cut from the dataset photos.
Output: results/watermark_localizer/seed<seed>.json, models/watermark_localizer_seed<seed>.pt/.onnx
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from roboeye.camva.notes import load_splits  # noqa: E402

L = ROOT / "results" / "watermark_localizer"
CACHE = ROOT / "cache" / "wm_loc"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
S = 320


class Localizer(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = m.features
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(576, 256), nn.Hardswish(), nn.Linear(256, 8))

    def forward(self, x):
        return torch.sigmoid(self.head(self.features(x)))


def to_tensor(bgr):
    return torch.from_numpy(((cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))


def augment(img, pts, rng):
    """Random affine (the corners move with the image) plus brightness / contrast / blur."""
    ang = rng.uniform(-10, 10)
    sc = rng.uniform(0.85, 1.1)
    tx, ty = rng.uniform(-0.08, 0.08, 2) * S
    M = cv2.getRotationMatrix2D((S / 2, S / 2), ang, sc)
    M[:, 2] += (tx, ty)
    out = cv2.warpAffine(img, M, (S, S), borderMode=cv2.BORDER_REFLECT)
    p = np.hstack([pts * S, np.ones((4, 1), np.float32)]) @ M.T / S
    out = np.clip(out.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (5, 5), rng.uniform(0.5, 1.5))
    return out, p.astype(np.float32)


def quad_iou(a, b):
    a, b = np.float32(a), np.float32(b)
    try:
        inter, _ = cv2.intersectConvexConvex(a, b)
    except cv2.error:
        return 0.0
    ua = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / ua) if ua > 0 else 0.0


@torch.inference_mode()
def predict(model, imgs):
    model.eval()
    out = []
    for i in range(0, len(imgs), 64):
        out.append(model(torch.stack([to_tensor(x) for x in imgs[i:i + 64]]).to(DEV)).cpu().numpy())
    return np.concatenate(out).reshape(-1, 4, 2)


def window_crop(bgr700, corners):
    src = np.float32(corners) * np.float32([bgr700.shape[1], bgr700.shape[0]])
    M = cv2.getPerspectiveTransform(src, np.float32([[0, 0], [224, 0], [224, 224], [0, 224]]))
    return cv2.warpPerspective(bgr700, M, (224, 224))


def _research_record(kind, *args):
    """Experiment-level record for research_results/ (see research_results/README.md). Never raises."""
    try:
        import sys as _sys
        from pathlib import Path as _P
        root = next(p for p in _P(__file__).resolve().parents if (p / "research_results" / "hooks.py").exists())
        if str(root) not in _sys.path:
            _sys.path.insert(0, str(root))
        from research_results import hooks
        hooks.call(kind, *args)
    except Exception as exc:  # noqa: BLE001
        print("[research_results] not recorded:", exc)


def main() -> None:
    global L, CACHE
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    generic = len(sys.argv) > 2
    if generic:
        L = Path(sys.argv[2])
        CACHE = L / "cache"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    lab = json.loads((L / "labels.json").read_text(encoding="utf-8"))["rows"]
    meta = json.loads((L / "labels.json").read_text(encoding="utf-8"))
    img = {r["note_id"]: cv2.imread(str(CACHE / (r["note_id"].replace(":", "_") + ".jpg"))) for r in lab}
    by = {s: [r for r in lab if r["split"] == s] for s in ("train", "val", "test")}
    tr = [r for r in by["train"] if r["corners"]]
    va = [r for r in by["val"] if r["corners"]]
    model = Localizer().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 40
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * math.ceil(len(tr) / 32))
    best, hist = (1e9, None, 0), []
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(tr))
        for i in range(0, len(order), 32):
            batch = [augment(img[tr[j]["note_id"]], np.float32(tr[j]["corners"]), rng) for j in order[i:i + 32]]
            x = torch.stack([to_tensor(b[0]) for b in batch]).to(DEV)
            y = torch.from_numpy(np.stack([b[1] for b in batch]).reshape(-1, 8)).to(DEV)
            loss = nn.functional.smooth_l1_loss(model(x), y, beta=0.02)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        pv = predict(model, [img[r["note_id"]] for r in va])
        err = float(np.linalg.norm(pv - np.float32([r["corners"] for r in va]), axis=2).mean())
        hist.append({"epoch": ep, "val_mean_corner_err": err})
        print(hist[-1], flush=True)
        if err < best[0]:
            best = (err, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1])

    model.eval().cpu()
    torch.save({"state_dict": model.state_dict(), "input": "320x320 RGB back-lit photo, ImageNet normalisation",
                "output": "4 window corners TL,TR,BR,BL as fractions of width/height", "seed": seed}, (L if generic else ROOT / "models") / f"watermark_localizer_seed{seed}.pt")
    try:
        torch.onnx.export(model, torch.zeros(1, 3, S, S), str((L if generic else ROOT / "models") / f"watermark_localizer_seed{seed}.onnx"),
                          input_names=["image"], output_names=["corners"], opset_version=17, dynamo=False)
    except Exception as exc:  # the .pt is saved; export again later
        print("ONNX export failed:", exc, flush=True)
    model.to(DEV)
    ok_reg = {r["note_id"] for r in json.loads((ROOT / "results" / "watermark" / "features.json").read_text(encoding="utf-8")) if r["ok"]}

    # ---- TEST, read once ----
    te = by["test"]
    pt = predict(model, [img[r["note_id"]] for r in te])
    lab_te = [(i, r) for i, r in enumerate(te) if r["corners"]]
    errs = [float(np.linalg.norm(pt[i] - np.float32(r["corners"]), axis=1).mean()) for i, r in lab_te]
    ious = [quad_iou(pt[i] * S, np.float32(r["corners"]) * S) for i, r in lab_te]

    # crops at the registration resolution (half decode, 700 px wide), scored by the published MobileNetV2
    if generic:
        records = {r["note_id"]: {"view_paths": [None] * 5 + [str(Path(meta["dataset_root"]) / r["photo"])]} for r in te}
    else:
        _, records = load_splits(ROOT / "results" / "serial_split")
    ck = torch.load(ROOT / "models" / "watermark_mobilenetv2.pt", map_location="cpu", weights_only=False)
    clf = models.mobilenet_v2()
    clf.classifier[1] = nn.Linear(clf.classifier[1].in_features, 2)
    clf.load_state_dict(ck["state_dict"])
    clf = clf.to(DEV).eval()
    crops = []
    for i, r in enumerate(te):
        im = cv2.imread(records[r["note_id"]]["view_paths"][5], cv2.IMREAD_REDUCED_COLOR_2)
        s = 700 / im.shape[1]
        crops.append(window_crop(cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), pt[i]))
    with torch.inference_mode():
        pg = torch.softmax(clf(torch.stack([to_tensor(c) for c in crops]).to(DEV)), 1)[:, 1].cpu().numpy()
    y = np.array([r["label"] for r in te])  # 1 = genuine
    # registration pipeline: the same classifier on the saved registration crops (watermark_features.py)
    reg_ids = [] if generic else [r["note_id"] for r in te if (ROOT / "results" / "watermark" / "crops" / (r["note_id"].replace(":", "_") + ".png")).exists()
               and r["note_id"] in ok_reg]
    with torch.inference_mode():
        rc = [cv2.imread(str(ROOT / "results" / "watermark" / "crops" / (n.replace(":", "_") + ".png"))) for n in reg_ids]
        rp = torch.softmax(clf(torch.stack([to_tensor(c) for c in rc]).to(DEV)), 1)[:, 1].cpu().numpy() if rc else []
    reg_p = dict(zip(reg_ids, rp))

    def summary(mask, p):
        pred = p >= 0.5
        yy = y[mask] == 1
        pp = pred[mask]
        return {"n": int(mask.sum()), "accuracy": float((pp == yy).mean()),
                "genuine_called_counterfeit": int((~pp & yy).sum()), "genuine_n": int(yy.sum()),
                "counterfeit_passed": int((pp & ~yy).sum()), "counterfeit_n": int((~yy).sum())}

    has_reg = np.array([r["note_id"] in reg_p for r in te])
    p_reg = np.array([reg_p.get(r["note_id"], np.nan) for r in te])
    from scipy.stats import binomtest
    a = (pg >= 0.5) == (y == 1)
    b = (p_reg >= 0.5) == (y == 1)
    b01 = int((a & ~b & has_reg).sum())
    b10 = int((~a & b & has_reg).sum())
    mc = float(binomtest(min(b01, b10), b01 + b10, 0.5).pvalue) if b01 + b10 else 1.0
    res = {
        "seed": seed, "split": "serial_split", "best_epoch": best[2], "best_val_mean_corner_err": best[0], "history": hist,
        "test_localization": {"n_labelled": len(lab_te), "mean_corner_err_frac": float(np.mean(errs)),
                              "median_quad_iou": float(np.median(ious)), "iou_ge_0.5": float(np.mean(np.array(ious) >= 0.5)),
                              "iou_ge_0.7": float(np.mean(np.array(ious) >= 0.7))},
        "test_classification": {
            "localizer_on_registered_notes": summary(has_reg, pg),
            "registration_on_registered_notes": summary(has_reg, np.nan_to_num(p_reg, nan=0.0)),
            "localizer_on_notes_registration_missed": summary(~has_reg, pg),
            "localizer_all_test_notes": summary(np.ones(len(te), bool), pg),
            "mcnemar_localizer_vs_registration": {"localizer_right_only": b01, "registration_right_only": b10, "p": mc}},
        "test_note_ids": [r["note_id"] for r in te], "test_probs_localizer": np.round(pg, 6).tolist(),
        "test_corners": np.round(pt, 5).tolist()}
    (L / f"seed{seed}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    if not generic:
        _research_record("watermark_localizer", res, L / f"seed{seed}.json")
    print(json.dumps({k: res[k] for k in ("best_epoch", "test_localization", "test_classification")}, indent=1))


if __name__ == "__main__":
    main()
