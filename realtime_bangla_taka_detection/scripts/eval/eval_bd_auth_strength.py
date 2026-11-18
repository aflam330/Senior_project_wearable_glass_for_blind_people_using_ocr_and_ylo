"""Honest stress + gated evaluation for bd_counterfeit_auth.pt.

Breaks the "100% lab accuracy" claim into:
  - clean locked test
  - corruption suite (blur, JPEG, dark, bright, noise, crop)
  - confidence gate sweep
  - ECE calibration
  - glass-safe policy metrics (REAL only if conf>=tau else check-by-hand)

Usage:
  python scripts/eval/eval_bd_auth_strength.py
Output:
  results/bd_auth_strength.json
  results/bd_auth_strength.md
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roboeye.bd_counterfeit_auth import BDCounterfeitAuth, IMG, _TF
from roboeye.config import BD_COUNTERFEIT_AUTH_WEIGHTS, DATA_SET, DEVICE, MODELS_DIR

MANIFEST = DATA_SET / "BD_Auth_Combined" / "manifest.csv"
OUT_JSON = ROOT / "results" / "bd_auth_strength.json"
OUT_MD = ROOT / "results" / "bd_auth_strength.md"
SEED = 42


def set_seed(s: int = SEED) -> None:
    np.random.seed(s)
    torch.manual_seed(s)


def load_test_rows() -> list[dict]:
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    return [r for r in rows if r["split"] == "test"]


def corrupt_bgr(bgr: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    img = bgr.copy()
    h, w = img.shape[:2]
    if kind == "clean":
        return img
    if kind == "blur":
        k = int(rng.choice([5, 7, 9, 11]))
        return cv2.GaussianBlur(img, (k, k), 0)
    if kind == "jpeg":
        q = int(rng.choice([25, 35, 45, 55]))
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img
    if kind == "dark":
        return np.clip(img.astype(np.float32) * float(rng.uniform(0.25, 0.45)), 0, 255).astype(np.uint8)
    if kind == "bright":
        return np.clip(img.astype(np.float32) * float(rng.uniform(1.4, 1.9)), 0, 255).astype(np.uint8)
    if kind == "noise":
        sigma = float(rng.uniform(8, 22))
        noise = rng.normal(0, sigma, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if kind == "crop":
        # keep 70-85% center crop then resize back (webcam framing miss)
        scale = float(rng.uniform(0.70, 0.85))
        nh, nw = int(h * scale), int(w * scale)
        y0 = (h - nh) // 2
        x0 = (w - nw) // 2
        crop = img[y0 : y0 + nh, x0 : x0 + nw]
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "rotate":
        ang = float(rng.uniform(-25, 25))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    return img


CORRUPTIONS = ["clean", "blur", "jpeg", "dark", "bright", "noise", "crop", "rotate"]


@torch.inference_mode()
def score_rows(clf: BDCounterfeitAuth, rows: list[dict], kind: str, rng: np.random.Generator) -> dict:
    ys, ps, confs, p_real, notes, lat = [], [], [], [], [], []
    for r in rows:
        path = Path(r["path"])
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        bgr = corrupt_bgr(bgr, kind, rng)
        t0 = time.perf_counter()
        out = clf.predict_bgr(bgr)
        lat.append((time.perf_counter() - t0) * 1000.0)
        y = int(r["label"])
        pred = 1 if out["model_class"] == "genuine" else 0
        ys.append(y)
        ps.append(pred)
        confs.append(float(out["confidence"]))
        p_real.append(float(out["genuine_prob"]))
        notes.append(r["note_id"])

    ys = np.array(ys)
    ps = np.array(ps)
    confs = np.array(confs)
    p_real = np.array(p_real)
    acc = float((ys == ps).mean()) if len(ys) else 0.0

    by = defaultdict(list)
    for n, y, p in zip(notes, ys, ps):
        by[n].append((int(y), int(p)))
    n_ok = sum(int(Counter(p for _, p in pairs).most_common(1)[0][0] == pairs[0][0]) for pairs in by.values())
    note_acc = n_ok / max(len(by), 1)

    fake = ys == 0
    real = ys == 1
    fake_rec = float((ps[fake] == 0).mean()) if fake.any() else 0.0
    real_rec = float((ps[real] == 1).mean()) if real.any() else 0.0

    # ECE with 10 bins on max-class confidence
    ece = 0.0
    if len(ys):
        bins = np.linspace(0, 1, 11)
        for i in range(10):
            m = (confs >= bins[i]) & (confs < bins[i + 1] if i < 9 else confs <= bins[i + 1])
            if not m.any():
                continue
            ece += (m.mean()) * abs(float((ys[m] == ps[m]).mean()) - float(confs[m].mean()))

    return {
        "corruption": kind,
        "n_images": int(len(ys)),
        "n_notes": int(len(by)),
        "image_acc": round(acc, 4),
        "note_acc": round(note_acc, 4),
        "fake_recall": round(fake_rec, 4),
        "real_recall": round(real_rec, 4),
        "mean_conf": round(float(confs.mean()) if len(confs) else 0.0, 4),
        "ece": round(float(ece), 4),
        "latency_ms_median": round(float(np.median(lat)) if lat else 0.0, 2),
        "p_real": p_real.tolist(),
        "y": ys.tolist(),
        "pred": ps.tolist(),
        "conf": confs.tolist(),
    }


def gate_metrics(y: np.ndarray, pred: np.ndarray, conf: np.ndarray, tau: float) -> dict:
    decided = conf >= tau
    abstain = int((~decided).sum())
    n_dec = int(decided.sum())
    acc = float((y[decided] == pred[decided]).mean()) if n_dec else 0.0
    # glass-safe: only announce REAL when pred=1 and conf>=tau; else check-by-hand
    # false REAL rate among true fakes that would be spoken as REAL
    spoken_real = decided & (pred == 1)
    false_real = int(((y == 0) & spoken_real).sum())
    true_fake = int((y == 0).sum())
    true_real = int((y == 1).sum())
    genuine_confirmed = int(((y == 1) & spoken_real).sum())
    return {
        "tau": tau,
        "coverage": round(n_dec / max(len(y), 1), 4),
        "accuracy_when_decided": round(acc, 4),
        "abstain": abstain,
        "decided": n_dec,
        "false_real_on_fakes": false_real,
        "false_real_rate": round(false_real / max(true_fake, 1), 4),
        "genuine_confirmed": genuine_confirmed,
        "genuine_confirm_rate": round(genuine_confirmed / max(true_real, 1), 4),
    }


def quick_baselines(rows: list[dict], device, epochs: int = 2, train_n: int = 480) -> dict:
    """Fast MobileNetV3-Small / ResNet18 baselines (no network download: weights=None)."""
    all_rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    train_rows = [r for r in all_rows if r["split"] == "train"]
    rng = np.random.default_rng(SEED)
    if len(train_rows) > train_n:
        idx = rng.choice(len(train_rows), train_n, replace=False)
        train_rows = [train_rows[i] for i in idx]

    class DS(Dataset):
        def __init__(self, rows, train=False):
            self.rows = rows
            if train:
                self.tf = transforms.Compose(
                    [
                        transforms.Resize((IMG, IMG)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                )
            else:
                self.tf = _TF

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            x = self.tf(Image.open(r["path"]).convert("RGB"))
            return x, int(r["label"])

    def make(name: str):
        if name == "mobilenet_v3_small":
            m = models.mobilenet_v3_small(weights=None)
            m.classifier[3] = torch.nn.Linear(m.classifier[3].in_features, 2)
        else:
            m = models.resnet18(weights=None)
            m.fc = torch.nn.Linear(m.fc.in_features, 2)
        return m.to(device)

    out = {}
    train_loader = DataLoader(DS(train_rows, True), batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(DS(rows, False), batch_size=64, shuffle=False, num_workers=0)
    for name in ("mobilenet_v3_small", "resnet18"):
        print(f"  baseline {name} ...", flush=True)
        m = make(name)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()
        for ep in range(epochs):
            m.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss_fn(m(x), y).backward()
                opt.step()
            print(f"    epoch {ep+1}/{epochs} done", flush=True)
        m.eval()
        ys, ps = [], []
        with torch.inference_mode():
            for x, y in test_loader:
                pred = m(x.to(device)).argmax(1).cpu().tolist()
                ys.extend(y.tolist())
                ps.extend(pred)
        ys = np.array(ys)
        ps = np.array(ps)
        out[name] = {
            "epochs": epochs,
            "train_n": len(train_rows),
            "pretrained": False,
            "image_acc": round(float((ys == ps).mean()), 4),
            "fake_recall": round(float((ps[ys == 0] == 0).mean()) if (ys == 0).any() else 0.0, 4),
            "real_recall": round(float((ps[ys == 1] == 1).mean()) if (ys == 1).any() else 0.0, 4),
        }
        torch.save({"model": m.state_dict(), "arch": name}, MODELS_DIR / f"bd_auth_baseline_{name}.pt")
        print(f"  {name}: {out[name]}", flush=True)
    return out


def write_md(rep: dict) -> None:
    lines = [
        "# BD Auth Strength Report",
        "",
        f"Weights: `{rep['weights']}`",
        f"Device: `{rep['device']}`",
        f"Test images: {rep['n_test']}",
        "",
        "## Clean vs corruptions (image accuracy)",
        "",
        "| Corruption | Image acc | Note acc | Fake recall | Real recall | ECE | Median ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in rep["corruptions"]:
        lines.append(
            f"| {c['corruption']} | {c['image_acc']:.3f} | {c['note_acc']:.3f} | "
            f"{c['fake_recall']:.3f} | {c['real_recall']:.3f} | {c['ece']:.3f} | {c['latency_ms_median']:.1f} |"
        )
    lines += ["", "## Confidence gate (clean test)", ""]
    lines += [
        "| tau | coverage | acc when decided | false REAL on fakes | genuine confirm rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for g in rep["gates"]:
        lines.append(
            f"| {g['tau']:.2f} | {g['coverage']:.3f} | {g['accuracy_when_decided']:.3f} | "
            f"{g['false_real_on_fakes']} ({g['false_real_rate']:.3f}) | {g['genuine_confirm_rate']:.3f} |"
        )
    lines += ["", "## Quick baselines (3 epochs, clean test)", ""]
    for k, v in rep.get("baselines", {}).items():
        lines.append(f"- **{k}**: image_acc={v['image_acc']:.3f} fake_rec={v['fake_recall']:.3f} real_rec={v['real_recall']:.3f}")
    lines += [
        "",
        "## Paper wording",
        "",
        "- Do **not** claim field 95% from clean lab alone.",
        "- Report clean + worst corruption + gated false-REAL rate.",
        "- Glass speaks REAL only above tau; else check-by-hand (never 'counterfeit').",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    set_seed()
    if not MANIFEST.is_file():
        raise SystemExit(f"Missing manifest: {MANIFEST}")
    if not BD_COUNTERFEIT_AUTH_WEIGHTS.is_file():
        raise SystemExit(f"Missing weights: {BD_COUNTERFEIT_AUTH_WEIGHTS}")

    rows = load_test_rows()
    clf = BDCounterfeitAuth(min_conf=0.0)  # raw scores; gating applied after
    assert clf.loaded
    rng = np.random.default_rng(SEED)

    print(f"Device={DEVICE}  test={len(rows)}  weights={BD_COUNTERFEIT_AUTH_WEIGHTS.name}")
    corrupt_reports = []
    clean_detail = None
    for kind in CORRUPTIONS:
        print(f"  corruption={kind} ...", flush=True)
        m = score_rows(clf, rows, kind, rng)
        slim = {k: v for k, v in m.items() if k not in ("p_real", "y", "pred", "conf")}
        corrupt_reports.append(slim)
        print(
            f"    img={slim['image_acc']:.3f} note={slim['note_acc']:.3f} "
            f"fake={slim['fake_recall']:.3f} real={slim['real_recall']:.3f} ece={slim['ece']:.3f}"
        )
        if kind == "clean":
            clean_detail = m

    y = np.array(clean_detail["y"])
    pred = np.array(clean_detail["pred"])
    conf = np.array(clean_detail["conf"])
    gates = [gate_metrics(y, pred, conf, tau) for tau in (0.50, 0.70, 0.80, 0.90, 0.95, 0.99)]

    print("Training quick baselines (scratch, 2 epochs)...", flush=True)
    baselines = quick_baselines(rows, DEVICE)

    # glass policy at 0.90 using already-scored clean detail + re-score only non-clean kinds once
    glass = []
    for kind, detail in zip(CORRUPTIONS, [None] * len(CORRUPTIONS)):
        if kind == "clean":
            m = clean_detail
        else:
            # reuse corruption metrics already computed for accuracy; gate needs vectors — rescore once
            m = score_rows(clf, rows, kind, np.random.default_rng(SEED + (abs(hash(kind)) % 1000)))
        g = gate_metrics(np.array(m["y"]), np.array(m["pred"]), np.array(m["conf"]), 0.90)
        glass.append({"corruption": kind, **{k: v for k, v in g.items() if k != "tau"}})

    worst = min(corrupt_reports, key=lambda r: r["image_acc"])
    report = {
        "weights": str(BD_COUNTERFEIT_AUTH_WEIGHTS),
        "device": str(DEVICE),
        "n_test": len(rows),
        "clean_image_acc": next(c["image_acc"] for c in corrupt_reports if c["corruption"] == "clean"),
        "worst_corruption": worst["corruption"],
        "worst_image_acc": worst["image_acc"],
        "corruptions": corrupt_reports,
        "gates": gates,
        "glass_tau_0_90": glass,
        "baselines": baselines,
        "recommendation": {
            "deploy_tau": 0.90,
            "paper_claim": (
                f"Clean locked-test image accuracy {corrupt_reports[0]['image_acc']:.1%}; "
                f"worst stress ({worst['corruption']}) {worst['image_acc']:.1%}. "
                "Glass announces likely-genuine only when confidence >= 0.90."
            ),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    # JSON without huge arrays
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_md(report)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(report["recommendation"]["paper_claim"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
