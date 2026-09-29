"""Approach A, step 2: fine-tune PRMVT on synthetic whole-note photographs.

Input: reconstructed whole notes (results/jaal_whole/synth_notes, from synth_whole_notes.py), which
keep the JaalTaka note-disjoint split.
Each training sample is made on the fly:
  note -> random perspective (corners up to 8 %) and scale -> pasted on a random COCO val2017 image ->
  cropped to the note box plus a 0-8 % margin (like a YOLO box) -> brightness x0.6-1.4, contrast,
  gamma, blur (p 0.3), JPEG q40-95, an occluding patch up to 15 % of the area (p 0.3) ->
  640 px wide -> the four view windows the glass cuts (savior_glass/config.py JAAL_VIEW_WINDOWS) ->
  PRMVT input; training uses a random prefix of 1-4 views.
Start: results/qduig/prefix_ft/seed42/checkpoint.pt (the deployed model). Batch-norm statistics frozen.
Validation: one fixed composite per VAL note (seeded), 4 views; the epoch with the best VAL accuracy
is kept. TEST is not read here.
Output: results/jaal_whole/approach_a/checkpoint.pt, train_log.json
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORK / "savior_glass"))
from roboeye.authenticity import default_transform  # noqa: E402
from roboeye.qduig.engine import load_qduig  # noqa: E402
import config as glass_config  # noqa: E402

SYN = ROOT / "results" / "jaal_whole"
OUT = SYN / "approach_a"
COCO = WORK / "data set" / "coco2017" / "val2017"
PRMVT = ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "checkpoint.pt"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TF = default_transform(train=False)


def cut_views(crop: np.ndarray) -> list[np.ndarray]:
    h, w = crop.shape[:2]
    return [crop[int(y0 * h):max(int(y1 * h), int(y0 * h) + 2), int(x0 * w):max(int(x1 * w), int(x0 * w) + 2)]
            for x0, y0, x1, y1 in glass_config.JAAL_VIEW_WINDOWS]


def composite(note: np.ndarray, bg: np.ndarray, rng: random.Random) -> np.ndarray:
    nh, nw = note.shape[:2]
    s = rng.uniform(0.45, 0.8) * 1000 / nw
    note = cv2.resize(note, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    nh, nw = note.shape[:2]
    j = 0.08
    src = np.float32([[0, 0], [nw, 0], [nw, nh], [0, nh]])
    dst = np.float32([[rng.uniform(-j, j) * nw + x, rng.uniform(-j, j) * nh + y] for x, y in src])
    dst -= dst.min(0)
    W, H = int(dst[:, 0].max()) + 1, int(dst[:, 1].max()) + 1
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(note, M, (W, H))
    mask = cv2.warpPerspective(np.ones((nh, nw), np.uint8), M, (W, H))
    bh, bw = int(H * rng.uniform(1.2, 1.6)), int(W * rng.uniform(1.2, 1.6))
    canvas = cv2.resize(bg, (bw, bh))
    oy, ox = rng.randint(0, bh - H), rng.randint(0, bw - W)
    roi = canvas[oy:oy + H, ox:ox + W]
    roi[mask > 0] = warped[mask > 0]
    m = rng.uniform(0, 0.08)
    y0, y1 = max(0, int(oy - m * H)), min(bh, int(oy + H + m * H))
    x0, x1 = max(0, int(ox - m * W)), min(bw, int(ox + W + m * W))
    crop = canvas[y0:y1, x0:x1].astype(np.float32)
    crop = crop * rng.uniform(0.6, 1.4)
    mean = crop.mean()
    crop = (crop - mean) * rng.uniform(0.75, 1.25) + mean
    crop = 255 * np.power(np.clip(crop, 0, 255) / 255, rng.uniform(0.7, 1.4))
    crop = np.clip(crop, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        crop = cv2.GaussianBlur(crop, (0, 0), rng.uniform(0.5, 2.0))
    if rng.random() < 0.3:
        ch, cw = crop.shape[:2]
        a = rng.uniform(0.03, 0.15) * ch * cw
        ph, pw = int(min(ch, (a * rng.uniform(0.5, 2)) ** 0.5)), 0
        pw = int(min(cw, a / max(ph, 1)))
        py, px = rng.randint(0, max(0, ch - ph)), rng.randint(0, max(0, cw - pw))
        crop[py:py + ph, px:px + pw] = [rng.randint(60, 220) for _ in range(3)]
    q = rng.randint(40, 95)
    crop = cv2.imdecode(cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    return cv2.resize(crop, None, fx=640 / crop.shape[1], fy=640 / crop.shape[1], interpolation=cv2.INTER_AREA)


class SynthNotes(Dataset):
    def __init__(self, rows, bgs, fixed_seed: int | None):
        self.rows, self.bgs, self.fixed = rows, bgs, fixed_seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        rng = random.Random(self.fixed + i) if self.fixed is not None else random.Random()
        note = cv2.imread(str(SYN / "synth_notes" / r["file"]))
        bg = cv2.imread(str(self.bgs[rng.randrange(len(self.bgs))]))
        views = cut_views(composite(note, bg, rng))
        x = torch.stack([TF(Image.fromarray(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))) for v in views])
        return x, int(r["label"])


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


@torch.inference_mode()
def evaluate(model, loader, k=4):
    model.eval()
    correct = n = 0
    for x, y in loader:
        x = x[:, :k].to(DEV)
        mask = torch.ones(x.size(0), k, dtype=torch.long, device=DEV)
        p = model(x, mask)["prob"].reshape(-1).cpu()
        correct += int(((p >= 0.5).long() == y).sum())
        n += len(y)
    return correct / n


def main() -> None:
    torch.manual_seed(0)
    random.seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    idx = json.loads((SYN / "synth_index.json").read_text(encoding="utf-8"))["notes"]
    kept = [r for r in idx if r.get("kept")]
    tr = [r for r in kept if r["split"] == "train"]
    va = [r for r in kept if r["split"] == "val"]
    bgs = sorted(COCO.glob("*.jpg"))
    dl_tr = DataLoader(SynthNotes(tr, bgs, None), batch_size=12, shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)
    dl_va = DataLoader(SynthNotes(va, bgs, 777000), batch_size=24, shuffle=False, num_workers=2)
    model = load_qduig(PRMVT).to(DEV)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    log = {"train_notes": len(tr), "val_notes": len(va), "epochs": []}
    best = evaluate(model, dl_va)
    log["val_acc_start"] = best
    torch.save({"model": model.state_dict(), "config": torch.load(PRMVT, map_location="cpu", weights_only=False).get("config"), "epoch": 0}, OUT / "checkpoint.pt")
    print("start val acc (4 views)", round(best, 4), flush=True)
    for ep in range(1, 5):
        model.train()
        freeze_bn(model)
        t0, tot, nb = time.time(), 0.0, 0
        for x, y in dl_tr:
            k = random.randint(1, 4)
            x = x[:, :k].to(DEV)
            mask = torch.ones(x.size(0), k, dtype=torch.long, device=DEV)
            loss = F.cross_entropy(model(x, mask)["logits"], y.to(DEV))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        acc = evaluate(model, dl_va)
        log["epochs"].append({"epoch": ep, "loss": tot / max(nb, 1), "val_acc_4view": acc, "seconds": round(time.time() - t0)})
        print(log["epochs"][-1], flush=True)
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "config": torch.load(PRMVT, map_location="cpu", weights_only=False).get("config"), "epoch": ep}, OUT / "checkpoint.pt")
    log["best_val_acc_4view"] = best
    (OUT / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
