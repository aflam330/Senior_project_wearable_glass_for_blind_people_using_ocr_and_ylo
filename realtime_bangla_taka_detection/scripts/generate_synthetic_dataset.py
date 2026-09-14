"""
Build a YOLO detection dataset suited for real-time webcam detection.

v2 improvements over v1:
  - Real COCO photo backgrounds (desks, rooms, hands, outdoor) instead of
    solid colours / gradients / noise.
  - Perspective warp to simulate notes held at an angle to the camera.
  - Motion blur, Gaussian noise and JPEG re-compression to mimic webcam artefacts.
  - Wider HSV / brightness jitter.
  - Wider rotation range (±30°).
  - 10 % negative images (background only, empty label) to cut false positives.
  - Falls back to synthetic backgrounds when COCO images run out.
"""

import argparse
import io
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

RANDOM_SEED = 42
SPLIT_RATIOS = (0.8, 0.1, 0.1)
CANVAS_SIZE = 640
COMPOSITES_PER_IMAGE = 4
MAX_SOURCE_DIM = 900
NEGATIVE_RATIO = 0.10   # fraction of output images with no note

BG_DIR = Path(r"E:\Final SP\data set\coco2017\val2017")

SOURCE_DIR = Path(
    r"E:\Final SP\data set\Bangladeshi_Paper_Currency_Raw"
    r"\Bangladeshi_Paper_Currency_Raw"
)
DEST_DIR = Path(r"E:\Final SP\data set\currency_yolo_data")

CLASSES = {
    "2":    (0, "2_taka"),
    "5":    (1, "5_taka"),
    "10":   (2, "10_taka"),
    "20":   (3, "20_taka"),
    "50":   (4, "50_taka"),
    "100":  (5, "100_taka"),
    "200":  (6, "200_taka"),
    "500":  (7, "500_taka"),
    "1000": (8, "1000_taka"),
}


# ── backgrounds ──────────────────────────────────────────────────────────────

def _synthetic_background(size):
    mode = random.choice(["solid", "gradient", "noise"])
    if mode == "solid":
        color = tuple(random.randint(20, 235) for _ in range(3))
        return Image.new("RGB", (size, size), color)
    if mode == "gradient":
        c1 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
        c2 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
        t = np.linspace(0, 1, size, dtype=np.float32).reshape(-1, 1, 1)
        arr = (c1 * (1 - t) + c2 * t).astype(np.uint8)
        arr = np.repeat(arr, size, axis=1)
        if random.random() < 0.5:
            arr = arr.transpose(1, 0, 2)
        return Image.fromarray(arr)
    base = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    img = Image.fromarray(base).filter(ImageFilter.GaussianBlur(radius=random.uniform(8, 25)))
    return img


_bg_paths = None

def _load_bg_pool():
    global _bg_paths
    if BG_DIR.exists():
        _bg_paths = list(BG_DIR.glob("*.jpg"))
        print(f"Found {len(_bg_paths)} COCO background images.")
    else:
        _bg_paths = []
        print("No COCO backgrounds found — using synthetic only.")


def get_background(size):
    if _bg_paths:
        try:
            path = random.choice(_bg_paths)
            img = Image.open(path).convert("RGB")
            # random crop to square
            w, h = img.size
            s = min(w, h)
            x0 = random.randint(0, w - s)
            y0 = random.randint(0, h - s)
            img = img.crop((x0, y0, x0 + s, y0 + s))
            img = img.resize((size, size), Image.LANCZOS)
            return img
        except Exception:
            pass
    return _synthetic_background(size)


# ── augmentations ─────────────────────────────────────────────────────────────

def load_source(path):
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    if max(w, h) > MAX_SOURCE_DIM:
        scale = MAX_SOURCE_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def perspective_warp(rgba_img, strength=0.08):
    """Random four-point perspective warp on an RGBA PIL image."""
    arr = np.array(rgba_img)
    h, w = arr.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dx, dy = w * strength, h * strength
    pts2 = np.float32([
        [random.uniform(0, dx),     random.uniform(0, dy)],
        [random.uniform(w-dx, w),   random.uniform(0, dy)],
        [random.uniform(0, dx),     random.uniform(h-dy, h)],
        [random.uniform(w-dx, w),   random.uniform(h-dy, h)],
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(arr, M, (w, h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
    return Image.fromarray(warped)


def hsv_jitter(rgb_img):
    arr = np.array(rgb_img, dtype=np.uint8)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-18, 18)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.5, 1.5), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.4, 1.6), 0, 255)
    return Image.fromarray(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))


def motion_blur(img):
    k = random.choice([3, 5, 7])
    kernel = np.zeros((k, k), dtype=np.float32)
    if random.random() < 0.5:
        kernel[k // 2, :] = 1.0 / k
    else:
        kernel[:, k // 2] = 1.0 / k
    arr = np.array(img, dtype=np.uint8)
    blurred = cv2.filter2D(arr, -1, kernel)
    return Image.fromarray(blurred)


def add_noise(img, std=12):
    arr = np.array(img, dtype=np.float32)
    arr += np.random.normal(0, std, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def jpeg_compress(img):
    quality = random.randint(55, 92)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def augment_note(note_rgba):
    """Apply random augmentations to the note (RGBA) before compositing."""
    # perspective warp
    if random.random() < 0.7:
        note_rgba = perspective_warp(note_rgba)

    # rotation ±30°
    angle = random.uniform(-30, 30)
    note_rgba = note_rgba.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))

    # scale
    scale_frac = random.uniform(0.25, 0.90)
    w, h = note_rgba.size
    scale = (scale_frac * CANVAS_SIZE) / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    note_rgba = note_rgba.resize((new_w, new_h), Image.LANCZOS)

    # HSV + brightness + contrast on the RGB channels
    rgb = note_rgba.convert("RGB")
    rgb = hsv_jitter(rgb)
    rgb = ImageEnhance.Brightness(rgb).enhance(random.uniform(0.4, 1.6))
    rgb = ImageEnhance.Contrast(rgb).enhance(random.uniform(0.7, 1.3))

    alpha = note_rgba.split()[3]
    note_rgba = Image.merge("RGBA", (*rgb.split(), alpha))
    return note_rgba


# ── composite ─────────────────────────────────────────────────────────────────

def make_composite(note_img):
    canvas = get_background(CANVAS_SIZE).convert("RGBA")
    note = augment_note(note_img.copy())
    new_w, new_h = note.size

    max_x = max(CANVAS_SIZE - new_w, 0)
    max_y = max(CANVAS_SIZE - new_h, 0)
    x, y = random.randint(0, max_x), random.randint(0, max_y)

    canvas.paste(note, (x, y), note)
    canvas_rgb = canvas.convert("RGB")

    # post-composite: motion blur (20 % chance), noise, JPEG artefacts
    if random.random() < 0.2:
        canvas_rgb = motion_blur(canvas_rgb)
    if random.random() < 0.5:
        canvas_rgb = add_noise(canvas_rgb)
    canvas_rgb = jpeg_compress(canvas_rgb)

    x_center = (x + new_w / 2) / CANVAS_SIZE
    y_center = (y + new_h / 2) / CANVAS_SIZE
    bw = new_w / CANVAS_SIZE
    bh = new_h / CANVAS_SIZE
    return canvas_rgb, (x_center, y_center, bw, bh)


def make_negative():
    """Background-only image with empty label (helps reduce false positives)."""
    bg = get_background(CANVAS_SIZE).convert("RGB")
    if random.random() < 0.3:
        bg = add_noise(bg, std=8)
    bg = jpeg_compress(bg)
    return bg


def _worker_init(dest_str: str):
    global DEST_DIR
    DEST_DIR = Path(dest_str)
    _load_bg_pool()


def _composite_job(item):
    """One composite (+ optional negative). Isolated RNG per job for multiprocessing."""
    dest_str, split, class_idx, class_name, img_path, i, seed = item
    dest = Path(dest_str)
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    note_img = load_source(Path(img_path))
    composite, bbox = make_composite(note_img)
    stem = f"{class_name}_{Path(img_path).stem}_{i}"
    composite.save(dest / split / "images" / f"{stem}.jpg", quality=90)
    x_c, y_c, bw, bh = bbox
    (dest / split / "labels" / f"{stem}.txt").write_text(
        f"{class_idx} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n"
    )
    made_neg = 0
    if random.random() < NEGATIVE_RATIO:
        neg = make_negative()
        neg_stem = f"neg_{stem}_{i}"
        neg.save(dest / split / "images" / f"{neg_stem}.jpg", quality=90)
        (dest / split / "labels" / f"{neg_stem}.txt").write_text("")
        made_neg = 1
    return split, 1, made_neg


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="max source notes per class (0 = all)")
    parser.add_argument("--composites", type=int, default=COMPOSITES_PER_IMAGE)
    parser.add_argument("--dest", type=Path, default=DEST_DIR)
    parser.add_argument("--no-wipe", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    dest = args.dest
    n_comp = args.composites

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    _load_bg_pool()

    for split in ("train", "valid", "test"):
        split_dir = dest / split
        if split_dir.exists() and not args.no_wipe:
            shutil.rmtree(split_dir)
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)

    jobs = []
    seed = RANDOM_SEED
    per_class = {}
    for folder_name, (class_idx, class_name) in CLASSES.items():
        src_folder = SOURCE_DIR / folder_name
        if not src_folder.is_dir():
            print(f"WARNING: missing folder {src_folder}, skipping")
            continue

        images = sorted(
            p for p in src_folder.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        random.shuffle(images)
        if args.limit:
            images = images[: args.limit]

        n = len(images)
        n_train = int(n * SPLIT_RATIOS[0])
        n_valid = int(n * SPLIT_RATIOS[1])
        splits = (
            ["train"] * n_train
            + ["valid"] * n_valid
            + ["test"] * (n - n_train - n_valid)
        )
        per_class[class_name] = n * n_comp
        for img_path, split in zip(images, splits):
            for i in range(n_comp):
                seed += 1
                jobs.append((str(dest), split, class_idx, class_name, str(img_path), i, seed))
        print(f"{class_name}: {n} source images -> {n * n_comp} composites queued")

    counts = {"train": 0, "valid": 0, "test": 0}
    neg_counts = {"train": 0, "valid": 0, "test": 0}
    done = 0
    workers = max(1, args.workers)
    print(f"Generating {len(jobs)} composites with {workers} workers → {dest}")
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(str(dest),)) as pool:
        futures = [pool.submit(_composite_job, job) for job in jobs]
        for fut in as_completed(futures):
            split, pos, neg = fut.result()
            counts[split] += pos
            neg_counts[split] += neg
            done += 1
            if done % 200 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}", flush=True)

    for split in ("train", "valid", "test"):
        print(f"{split}: {counts[split]} positives + {neg_counts[split]} negatives")
    print(f"TOTAL: {sum(counts.values())} positives + {sum(neg_counts.values())} negatives = "
          f"{sum(counts.values()) + sum(neg_counts.values())}")


if __name__ == "__main__":
    main()
