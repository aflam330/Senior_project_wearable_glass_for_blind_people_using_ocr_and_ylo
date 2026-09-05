"""
Annotate BanglaTaka note images with YOLO 2D bounding boxes.

For each image:
  1. Estimate background from border pixels
  2. Find the largest note-like foreground contour -> tight box
  3. Fall back to full-frame box if contour detection fails

Writes:
  - a .txt label next to every source image
  - a YOLO-ready copy under data set/yolo_bbox_annotated/{train,valid,test}
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np

RANDOM_SEED = 42
SPLIT_RATIOS = (0.8, 0.1, 0.1)

SOURCE_DIR = Path(
    r"e:\Final SP\data set\Bangladeshi_Paper_Currency_Raw"
    r"\Bangladeshi_Paper_Currency_Raw"
)
DEST_DIR = Path(r"e:\Final SP\data set\yolo_bbox_annotated")

CLASSES = {
    "2": (0, "2_taka"),
    "5": (1, "5_taka"),
    "10": (2, "10_taka"),
    "20": (3, "20_taka"),
    "50": (4, "50_taka"),
    "100": (5, "100_taka"),
    "200": (6, "200_taka"),
    "500": (7, "500_taka"),
    "1000": (8, "1000_taka"),
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def yolo_line(class_idx: int, x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> str:
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    # clamp
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 1e-6), 1.0)
    bh = min(max(bh, 1e-6), 1.0)
    return f"{class_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def detect_note_box(img: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return x1,y1,x2,y2 of the note, or None to use full-frame."""
    h, w = img.shape[:2]
    if h < 20 or w < 20:
        return None

    border = max(2, min(h, w) // 40)
    top = img[:border, :, :]
    bottom = img[-border:, :, :]
    left = img[:, :border, :]
    right = img[:, -border:, :]
    border_pixels = np.concatenate(
        [top.reshape(-1, 3), bottom.reshape(-1, 3), left.reshape(-1, 3), right.reshape(-1, 3)],
        axis=0,
    )
    bg = np.median(border_pixels, axis=0).astype(np.float32)

    diff = np.linalg.norm(img.astype(np.float32) - bg, axis=2)
    # Adaptive threshold relative to image contrast
    thr = max(18.0, float(np.percentile(diff, 60)) * 0.55)
    mask = (diff > thr).astype(np.uint8) * 255

    k = max(3, (min(h, w) // 120) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    area_img = float(h * w)
    best = None
    best_area = 0.0
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = float(bw * bh)
        if area < 0.05 * area_img or area > 0.995 * area_img:
            continue
        aspect = bw / max(bh, 1)
        # Banknotes are wider than tall (or rotated); keep a wide range
        if aspect < 0.35 or aspect > 4.5:
            continue
        if area > best_area:
            best_area = area
            best = (x, y, x + bw, y + bh)

    if best is None:
        # Fall back to largest contour without aspect filter if big enough
        c = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(c)
        area = float(bw * bh)
        if area < 0.08 * area_img:
            return None
        best = (x, y, x + bw, y + bh)

    # Small padding
    x1, y1, x2, y2 = best
    pad_x = int(0.01 * w)
    pad_y = int(0.01 * h)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)
    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None
    return x1, y1, x2, y2


def annotate_image(img_path: Path, class_idx: int) -> tuple[str, str]:
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Could not read {img_path}")
    h, w = img.shape[:2]
    box = detect_note_box(img)
    if box is None:
        line = f"{class_idx} 0.500000 0.500000 1.000000 1.000000"
        return line, "fullframe"
    x1, y1, x2, y2 = box
    return yolo_line(class_idx, x1, y1, x2, y2, w, h), "tight"


def main() -> None:
    random.seed(RANDOM_SEED)
    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"Source not found: {SOURCE_DIR}")

    for split in ("train", "valid", "test"):
        (DEST_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DEST_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {"tight": 0, "fullframe": 0, "total": 0}
    split_counts = {"train": 0, "valid": 0, "test": 0}

    for folder_name, (class_idx, class_name) in CLASSES.items():
        src_folder = SOURCE_DIR / folder_name
        if not src_folder.is_dir():
            print(f"WARNING: missing {src_folder}")
            continue

        images = sorted(
            p for p in src_folder.iterdir()
            if p.suffix.lower() in IMG_EXTS
        )
        random.shuffle(images)

        n = len(images)
        n_train = int(n * SPLIT_RATIOS[0])
        n_valid = int(n * SPLIT_RATIOS[1])
        splits = (
            ["train"] * n_train
            + ["valid"] * n_valid
            + ["test"] * (n - n_train - n_valid)
        )

        for img_path, split in zip(images, splits):
            line, mode = annotate_image(img_path, class_idx)

            # Label next to source image
            side_label = img_path.with_suffix(".txt")
            side_label.write_text(line + "\n", encoding="utf-8")

            # YOLO export copy
            dest_img = DEST_DIR / split / "images" / img_path.name
            dest_lbl = DEST_DIR / split / "labels" / (img_path.stem + ".txt")
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)
            dest_lbl.write_text(line + "\n", encoding="utf-8")

            stats[mode] += 1
            stats["total"] += 1
            split_counts[split] += 1

        print(f"{class_name}: {n} annotated")

    yaml_text = f"""# Auto-annotated BanglaTaka with 2D YOLO bounding boxes
path: {DEST_DIR.as_posix()}
train: train/images
val: valid/images
test: test/images
nc: 9
names:
  - 2_taka
  - 5_taka
  - 10_taka
  - 20_taka
  - 50_taka
  - 100_taka
  - 200_taka
  - 500_taka
  - 1000_taka
"""
    (DEST_DIR / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print("\nDone.")
    print(f"Total: {stats['total']}")
    print(f"Tight boxes: {stats['tight']}")
    print(f"Full-frame fallback: {stats['fullframe']}")
    print(f"Splits: {split_counts}")
    print(f"YOLO dataset: {DEST_DIR}")


if __name__ == "__main__":
    main()
