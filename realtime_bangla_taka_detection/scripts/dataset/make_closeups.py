"""Hand-held close-up training pictures for the Taka detector (14 October 2026).

Why: on NSTU-BDTAKA hand-held close-ups the deployed detector finds no note in 66.9 % of the photos and names the
right value in 18.5 % (results/external/external_taka.json). Its training pictures show whole notes inside scenes
(25-90 % of the frame). A blind user holds the note close, partly covered by fingers, often folded and moving.

Each picture here is built from a note photo of the detector's TRAINING split only (the source photos behind
currency_yolo_data/train), so no validation or test note is used:
  - the note is warped, turned by any angle, sometimes folded (one part darker and foreshortened), and scaled so
    that it fills or overflows the frame (longer side 1.2 to 2.6 times the frame in 70 %, 0.6 to 1.2 otherwise);
  - one to three finger shapes in skin tones cross it from the frame edge;
  - motion blur, low resolution (down to 256 pixels and back), colour and brightness changes, noise, JPEG.
The label is the box of the visible part of the note. No NSTU picture is used to make or select these.

    python scripts/dataset/make_closeups.py --n 6000
Writes "<workspace>/data set/currency_closeups/train/{images,labels}" and data/data_closeups.yaml.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.generate_synthetic_dataset import CLASSES, SOURCE_DIR, DATA_ROOT  # noqa: E402

S = 640
DEST = DATA_ROOT / "currency_closeups"
BG = DATA_ROOT / "coco2017" / "val2017"


def train_sources():
    """Source note photos whose composites are in the detector's training split."""
    stems = defaultdict(set)
    for p in (DATA_ROOT / "currency_yolo_data" / "train" / "images").glob("*.jpg"):
        m = re.match(r"^(\d+)_taka_(.+)_\d+$", p.stem)
        if m:
            stems[m.group(1)].add(m.group(2))
    out = []
    for value, names in stems.items():
        for name in names:
            src = SOURCE_DIR / value / f"{name}.jpg"
            if src.is_file():
                out.append((value, src))
    return sorted(out)


def background(rng):
    paths = background.paths
    img = cv2.imread(str(paths[rng.randrange(len(paths))]))
    h, w = img.shape[:2]
    s = min(h, w)
    y, x = rng.randrange(h - s + 1), rng.randrange(w - s + 1)
    return cv2.resize(img[y:y + s, x:x + s], (S, S))


def fold(note, rng):
    """One part of the note bent away: darker and foreshortened."""
    h, w = note.shape[:2]
    c = int(w * rng.uniform(0.3, 0.7))
    left, right = note[:, :c], note[:, c:]
    if rng.random() < 0.5:
        k = rng.uniform(0.45, 0.85)
        left = cv2.resize(left, (max(2, int(left.shape[1] * k)), h)).astype(np.float32)
        left[..., :3] *= rng.uniform(0.55, 0.9)
        left = left.astype(np.uint8)
    else:
        k = rng.uniform(0.45, 0.85)
        right = cv2.resize(right, (max(2, int(right.shape[1] * k)), h)).astype(np.float32)
        right[..., :3] *= rng.uniform(0.55, 0.9)
        right = right.astype(np.uint8)
    return np.concatenate([left, right], axis=1)


def place(note, rng):
    """Warp, turn and scale the note (BGRA); returns the canvas-sized note layer and its alpha."""
    h, w = note.shape[:2]
    d = 0.15
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst = np.float32([[rng.uniform(0, d * w), rng.uniform(0, d * h)], [rng.uniform((1 - d) * w, w), rng.uniform(0, d * h)],
                      [rng.uniform(0, d * w), rng.uniform((1 - d) * h, h)], [rng.uniform((1 - d) * w, w), rng.uniform((1 - d) * h, h)]])
    note = cv2.warpPerspective(note, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(0, 0, 0, 0))
    scale = rng.uniform(1.2, 2.6) if rng.random() < 0.7 else rng.uniform(0.6, 1.2)
    k = scale * S / max(h, w)
    note = cv2.resize(note, (max(2, int(w * k)), max(2, int(h * k))), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    angle = rng.uniform(-180, 180) if rng.random() < 0.5 else rng.uniform(-30, 30)
    h, w = note.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    note = cv2.warpAffine(note, M, (nw, nh), borderValue=(0, 0, 0, 0))
    # centre of the note somewhere in the middle part of the frame, so a good share stays visible
    cx, cy = rng.uniform(0.25, 0.75) * S, rng.uniform(0.25, 0.75) * S
    x0, y0 = int(cx - nw / 2), int(cy - nh / 2)
    layer = np.zeros((S, S, 4), np.uint8)
    xs, ys = max(0, x0), max(0, y0)
    xe, ye = min(S, x0 + nw), min(S, y0 + nh)
    if xe <= xs or ye <= ys:
        return None, None, 0.0
    layer[ys:ye, xs:xe] = note[ys - y0:ye - y0, xs - x0:xe - x0]
    full = float((note[..., 3] > 0).sum())
    visible = float((layer[..., 3] > 0).sum())
    return layer, visible / max(full, 1.0), visible / (S * S)


def finger(img, rng):
    """A finger: a rounded bar in a skin tone, shaded across its width, entering from a frame edge."""
    r = rng.uniform(95, 225)                                                               # dark to light skin
    tone = np.array([r * rng.uniform(0.55, 0.72), r * rng.uniform(0.68, 0.82), r])           # BGR
    width, length = rng.uniform(55, 120), rng.uniform(170, 420)
    side = rng.randrange(4)
    t = rng.uniform(0.1, 0.9) * S
    start = [(t, S + 20), (t, -20), (-20, t), (S + 20, t)][side]
    ang = [-90, 90, 0, 180][side] + rng.uniform(-35, 35)
    end = (start[0] + length * np.cos(np.radians(ang)), start[1] + length * np.sin(np.radians(ang)))
    mask = np.zeros((S, S), np.float32)
    cv2.line(mask, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), 1.0, int(width), lineType=cv2.LINE_AA)
    cv2.circle(mask, (int(end[0]), int(end[1])), int(width / 2), 1.0, -1, lineType=cv2.LINE_AA)
    dist = cv2.distanceTransform((mask > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    shade = 0.65 + 0.35 * np.clip(dist / (width / 2), 0, 1)
    mask = cv2.GaussianBlur(mask, (0, 0), 2.0)[..., None]
    skin = (tone[None, None, :] * shade[..., None]).clip(0, 255)
    return (img * (1 - mask) + skin * mask).astype(np.uint8)


def finish(img, rng):
    if rng.random() < 0.5:
        k = rng.choice([5, 9, 13, 17])
        ker = np.zeros((k, k), np.float32)
        cv2.line(ker, (0, k // 2), (k - 1, k // 2), 1.0, 1)
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
        ker = cv2.warpAffine(ker, M, (k, k))
        img = cv2.filter2D(img, -1, ker / max(ker.sum(), 1e-6))
    if rng.random() < 0.5:
        r = rng.choice([224, 256, 320])
        img = cv2.resize(cv2.resize(img, (r, r), interpolation=cv2.INTER_AREA), (S, S), interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-10, 10)) % 180
    hsv[..., 1] *= rng.uniform(0.6, 1.4)
    hsv[..., 2] *= rng.uniform(0.45, 1.4)
    img = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.5:
        img = (img + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(3, 12), img.shape)).clip(0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randrange(45, 92)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def make(value, src, rng):
    note = cv2.imread(str(src))
    if note is None:
        return None
    if max(note.shape[:2]) > 900:
        k = 900 / max(note.shape[:2])
        note = cv2.resize(note, (int(note.shape[1] * k), int(note.shape[0] * k)), interpolation=cv2.INTER_AREA)
    note = cv2.cvtColor(note, cv2.COLOR_BGR2BGRA)
    if rng.random() < 0.4:
        note = fold(note, rng)
    for _ in range(10):
        layer, vis_frac, _ = place(note, rng)
        if layer is not None and vis_frac >= 0.25:
            break
    else:
        return None
    a = layer[..., 3:4].astype(np.float32) / 255
    img = (background(rng) * (1 - a) + layer[..., :3] * a).astype(np.uint8)
    ys, xs = np.where(layer[..., 3] > 0)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    for _ in range(rng.choice([0, 1, 1, 2, 3])):
        img = finger(img, rng)
    img = finish(img, rng)
    cls = CLASSES[value][0]
    x0, y0, x1, y1 = box
    return img, f"{cls} {(x0 + x1) / 2 / S:.6f} {(y0 + y1) / 2 / S:.6f} {(x1 - x0) / S:.6f} {(y1 - y0) / S:.6f}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    background.paths = sorted(BG.glob("*.jpg"))
    sources = train_sources()
    by_value = defaultdict(list)
    for v, s in sources:
        by_value[v].append(s)
    print({v: len(s) for v, s in sorted(by_value.items(), key=lambda x: int(x[0]))}, "training source photos")
    (DEST / "train" / "images").mkdir(parents=True, exist_ok=True)
    (DEST / "train" / "labels").mkdir(parents=True, exist_ok=True)
    values = sorted(by_value, key=int)
    made = 0
    i = 0
    while made < args.n:
        value = values[i % len(values)]                       # the same number of pictures per note value
        i += 1
        out = make(value, rng.choice(by_value[value]), rng)
        if out is None:
            continue
        img, label = out
        stem = f"closeup_{value}_{made:05d}"
        cv2.imwrite(str(DEST / "train" / "images" / f"{stem}.jpg"), img)
        (DEST / "train" / "labels" / f"{stem}.txt").write_text(label)
        made += 1
        if made % 1000 == 0:
            print(made, flush=True)
    (ROOT / "data" / "data_closeups.yaml").write_text(
        "# Detector fine-tune with hand-held close-ups (scripts/dataset/make_closeups.py)\n"
        f"path: {DATA_ROOT.as_posix()}\n"
        "train:\n  - currency_yolo_data/train/images\n  - currency_closeups/train/images\n"
        "val: currency_yolo_data/valid/images\ntest: currency_yolo_data/test/images\n\nnc: 9\nnames:\n"
        + "".join(f"  - {CLASSES[v][1]}\n" for v in sorted(CLASSES, key=int)))
    print("made", made)


if __name__ == "__main__":
    main()
