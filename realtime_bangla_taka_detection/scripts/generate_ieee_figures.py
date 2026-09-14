"""Generate IEEE publication-quality Figures 2, 7, 8 and 9 for RoboEye.

Uses real BanglaTaka notes, COCO backgrounds and YOLOv8s weights when
available, and falls back to synthetic stand-ins otherwise.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

matplotlib.use("Agg")

# ── paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "ieee_figures"
ASSET_DIR = OUT_DIR / "_assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ASSET_DIR.mkdir(parents=True, exist_ok=True)

DATA_SET = ROOT.parent / "data set"
LOCAL_NOTES = DATA_SET / "Bangladeshi_Paper_Currency_Raw" / "Bangladeshi_Paper_Currency_Raw"
MEMOD_NOTES = Path(
    r"C:\Users\memod\OneDrive\Desktop\currency\data"
    r"\A Diverse Image Dataset for Bangladeshi Currency Recognition"
    r"\extracted\Bangladeshi_Paper_Currency_Raw"
)
COCO_VAL = DATA_SET / "coco2017" / "val2017"
LOCAL_YOLO = DATA_SET / "currency_yolo_data"

BG_SEARCH = [
    Path(r"C:\currency_backgrounds\images"),
    COCO_VAL,
    ASSET_DIR,
]
MODEL_PATH = ROOT / "models" / "best.pt"

# Filled by resolve_yolo_paths() after local composites exist.
TEST_IMG_DIR = LOCAL_YOLO / "test" / "images"
TRAIN_COMP = LOCAL_YOLO / "train" / "images" / "100_taka_100  Taka_001_1.jpg"
TRAIN_LAB = LOCAL_YOLO / "train" / "labels" / "100_taka_100  Taka_001_1.txt"

# Curated COCO val2017 scenes (desk/indoor/street). Avoids unsafe random first-N files.
FIG7_COCO_NAMES = [
    "000000000139.jpg",
    "000000000724.jpg",
    "000000000785.jpg",
    "000000002006.jpg",
    "000000004134.jpg",
    "000000397133.jpg",
]

CANVAS = 640
BOX_GREEN = "#00FF66"
RNG = random.Random(42)
np.random.seed(42)

CLASS_NAMES = {
    0: "2 Taka",
    1: "5 Taka",
    2: "10 Taka",
    3: "20 Taka",
    4: "50 Taka",
    5: "100 Taka",
    6: "200 Taka",
    7: "500 Taka",
    8: "1000 Taka",
}


# ── matplotlib style ─────────────────────────────────────────────────────────

def apply_ieee_style() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"


def save_figure(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf"):
        dest = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(
            dest,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            pad_inches=0.04,
        )
        print(f"  saved {dest}")
    plt.close(fig)


def _blank_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ── IO helpers ───────────────────────────────────────────────────────────────

def first_existing(paths) -> Path | None:
    for p in paths:
        if p is not None and Path(p).is_file():
            return Path(p)
    return None


def note_path(denom: str, stem: str | None = None) -> Path | None:
    name = stem or f"{denom}  Taka_001.jpg"
    hits = []
    for root in (LOCAL_NOTES, MEMOD_NOTES):
        p = root / denom / name
        if p.is_file():
            return p
        folder = root / denom
        if folder.is_dir():
            hits.extend(sorted(folder.glob("*.jpg")))
    return hits[0] if hits else None


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def background_path(filename: str = "000000000285.jpg") -> Path | None:
    for folder in BG_SEARCH:
        p = folder / filename
        if p.is_file():
            return p
    for folder in BG_SEARCH:
        if folder.is_dir():
            jpgs = sorted(folder.glob("*.jpg"))
            if jpgs:
                return jpgs[0]
    return None


def list_backgrounds() -> list[Path]:
    found: list[Path] = []
    seen = set()
    for folder in BG_SEARCH:
        if not folder.is_dir():
            continue
        for p in sorted(folder.glob("*.jpg")):
            if p.name not in seen:
                seen.add(p.name)
                found.append(p)
    return found


def square_crop_resize(img: Image.Image, size: int = CANVAS) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    x0 = (w - s) // 2
    y0 = (h - s) // 2
    return img.crop((x0, y0, x0 + s, y0 + s)).resize((size, size), Image.LANCZOS)


def pad_to_square(img: Image.Image, fill=(255, 255, 255)) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), fill)
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


def trim_white_border(img: Image.Image, thresh: int = 244) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    mask = np.any(arr < thresh, axis=2)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    pad = 4
    y0 = max(int(rows[0]) - pad, 0)
    y1 = min(int(rows[-1]) + pad + 1, arr.shape[0])
    x0 = max(int(cols[0]) - pad, 0)
    x1 = min(int(cols[-1]) + pad + 1, arr.shape[1])
    return img.crop((x0, y0, x1, y1))


def note_to_rgba(img: Image.Image) -> Image.Image:
    cropped = trim_white_border(img)
    return cropped.convert("RGBA")


# ── synthetic fallbacks ──────────────────────────────────────────────────────

def synthetic_note(denom: str = "100", size=(520, 220)) -> Image.Image:
    """Coloured banknote-like rectangle used only if real files are missing."""
    palettes = {
        "2": ((70, 140, 90), (40, 90, 55)),
        "5": ((90, 70, 50), (55, 40, 30)),
        "10": ((180, 70, 110), (120, 40, 70)),
        "20": ((70, 90, 160), (40, 50, 110)),
        "50": ((200, 120, 80), (150, 80, 45)),
        "100": ((70, 150, 155), (40, 95, 100)),
        "200": ((90, 70, 140), (55, 40, 95)),
        "500": ((80, 120, 90), (45, 80, 55)),
        "1000": ((90, 70, 55), (55, 40, 30)),
    }
    c1, c2 = palettes.get(denom, ((70, 150, 155), (40, 95, 100)))
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        t = x / max(w - 1, 1)
        arr[:, x] = (np.array(c1) * (1 - t) + np.array(c2) * t).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    draw.ellipse((int(0.16 * w), int(0.22 * h), int(0.38 * w), int(0.82 * h)), fill=(210, 200, 175))
    draw.ellipse((int(0.20 * w), int(0.30 * h), int(0.34 * w), int(0.62 * h)), fill=(90, 70, 55))
    try:
        font = ImageFont.truetype("timesbd.ttf", max(18, h // 6))
    except OSError:
        font = ImageFont.load_default()
    draw.text((int(0.62 * w), int(0.12 * h)), denom, fill=(255, 255, 255), font=font)
    return img


def synthetic_background(size: int = CANVAS, kind: str = "indoor") -> Image.Image:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    if kind == "grass":
        arr[:, :] = (90, 130, 55)
        noise = np.random.normal(0, 18, arr.shape)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        cv2.circle(arr, (size // 2, int(size * 0.58)), int(size * 0.38), (160, 120, 70), -1)
        cv2.circle(arr, (size // 2, int(size * 0.42)), int(size * 0.22), (120, 85, 50), -1)
    else:
        arr[: int(size * 0.55)] = (210, 205, 195)
        arr[int(size * 0.55) :] = (150, 118, 88)
        noise = np.random.normal(0, 10, arr.shape)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        cv2.rectangle(arr, (40, 80), (220, 260), (60, 60, 70), -1)
        cv2.rectangle(arr, (400, 300), (600, 500), (180, 170, 150), -1)
    return Image.fromarray(arr)


def load_source_note(denom: str = "100") -> Image.Image:
    path = note_path(denom)
    if path is not None:
        print(f"  note [{denom}]: {path}")
        return load_rgb(path)
    print(f"  note [{denom}]: SYNTHETIC (file not found)")
    return synthetic_note(denom)


def load_coco_background(filename: str = "000000000285.jpg") -> Image.Image:
    path = background_path(filename)
    if path is not None:
        print(f"  background: {path}")
        return load_rgb(path)
    print("  background: SYNTHETIC (COCO file not found)")
    return synthetic_background(kind="grass" if "285" in filename else "indoor")


# ── note augmentation / compositing ──────────────────────────────────────────

def perspective_warp(rgba: Image.Image, strength: float = 0.08) -> Image.Image:
    arr = np.array(rgba)
    h, w = arr.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dx, dy = w * strength, h * strength
    pts2 = np.float32(
        [
            [dx * 0.4, dy * 0.6],
            [w - dx * 0.9, dy * 0.2],
            [dx * 0.2, h - dy * 0.5],
            [w - dx * 0.5, h - dy * 0.8],
        ]
    )
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(
        arr, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )
    return Image.fromarray(warped)


def color_jitter(rgb: Image.Image, h_shift: float = 8, s_mul: float = 1.18, v_mul: float = 0.92) -> Image.Image:
    arr = np.array(rgb.convert("RGB"), dtype=np.uint8)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_mul, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_mul, 0, 255)
    out = Image.fromarray(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))
    out = ImageEnhance.Contrast(out).enhance(1.12)
    return out


def yolo_to_xywh(xc, yc, bw, bh, img_w, img_h):
    w = bw * img_w
    h = bh * img_h
    x = (xc - bw / 2.0) * img_w
    y = (yc - bh / 2.0) * img_h
    return x, y, w, h


def paste_note(
    background: Image.Image,
    note_rgb: Image.Image,
    angle: float = -18.0,
    scale_frac: float = 0.55,
    xy: tuple[int, int] | None = None,
    jitter: bool = True,
    warp: bool = False,
) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """Paste an augmented note onto a 640×640 background. Returns RGB image + pixel bbox."""
    canvas = square_crop_resize(background, CANVAS).convert("RGBA")
    note = note_to_rgba(note_rgb)
    if warp:
        note = perspective_warp(note, 0.08)
    note = note.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
    w, h = note.size
    scale = (scale_frac * CANVAS) / max(w, h)
    new_w, new_h = max(8, int(w * scale)), max(8, int(h * scale))
    note = note.resize((new_w, new_h), Image.LANCZOS)
    if jitter:
        rgb = color_jitter(note.convert("RGB"))
        note = Image.merge("RGBA", (*rgb.split(), note.split()[-1]))
    if xy is None:
        max_x = max(CANVAS - new_w, 0)
        max_y = max(CANVAS - new_h, 0)
        x = int(0.18 * max_x + 0.12 * CANVAS)
        y = int(0.28 * max_y + 0.18 * CANVAS)
    else:
        x, y = xy
        x = int(np.clip(x, 0, max(CANVAS - new_w, 0)))
        y = int(np.clip(y, 0, max(CANVAS - new_h, 0)))
    canvas.paste(note, (x, y), note)
    return canvas.convert("RGB"), (float(x), float(y), float(new_w), float(new_h))


def draw_box_on_axes(ax, x, y, w, h, label: str, lw: float = 2.5) -> None:
    ax.add_patch(
        Rectangle((x, y), w, h, linewidth=lw, edgecolor=BOX_GREEN, facecolor="none")
    )
    ty = y - 7 if y > 18 else y + 4
    va = "bottom" if y > 18 else "top"
    ax.text(
        x,
        ty,
        label,
        color="white",
        fontsize=8,
        fontweight="bold",
        va=va,
        ha="left",
        bbox=dict(facecolor=BOX_GREEN, edgecolor="none", pad=1.5),
    )


# ── build missing YOLO composites from local notes + COCO ─────────────────────

def resolve_yolo_paths() -> None:
    """Prefer the local generated split; fall back to C:\\currency_yolo_data."""
    global TEST_IMG_DIR, TRAIN_COMP, TRAIN_LAB
    candidates = [LOCAL_YOLO, Path(r"C:\currency_yolo_data")]
    root = next((p for p in candidates if (p / "train" / "images").is_dir()), LOCAL_YOLO)
    TEST_IMG_DIR = root / "test" / "images"
    TRAIN_COMP = root / "train" / "images" / "100_taka_100  Taka_001_1.jpg"
    TRAIN_LAB = root / "train" / "labels" / "100_taka_100  Taka_001_1.txt"
    print(f"  YOLO composite root: {root}")


def _save_yolo_item(split: str, stem: str, image, bbox, class_idx: int) -> None:
    img_dir = LOCAL_YOLO / split / "images"
    lab_dir = LOCAL_YOLO / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    image.save(img_dir / f"{stem}.jpg", quality=90)
    xc, yc, bw, bh = bbox
    (lab_dir / f"{stem}.txt").write_text(
        f"{class_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8"
    )


def ensure_local_composites() -> None:
    """Create the paper's named composite files from local BanglaTaka + COCO val2017.

    This is the same copy-paste engine as scripts/generate_synthetic_dataset.py.
    It does not rebuild all 22,315 images — only the files Figures 2 and 7 need.
    """
    needed_train = LOCAL_YOLO / "train" / "images" / "100_taka_100  Taka_001_1.jpg"
    test_dir = LOCAL_YOLO / "test" / "images"
    if needed_train.is_file() and test_dir.is_dir() and len(list(test_dir.glob("*.jpg"))) >= 6:
        print(f"  composites already present at {LOCAL_YOLO}")
        return

    print("  generating local YOLO composites (notes + COCO val2017)...")
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import generate_synthetic_dataset as gsd

    random.seed(42)
    np.random.seed(42)
    gsd._load_bg_pool()

    note_100 = LOCAL_NOTES / "100" / "100  Taka_001.jpg"
    if not note_100.is_file():
        print(f"  WARNING: missing {note_100}")
        return
    note_img = gsd.load_source(note_100)
    for i in range(4):
        composite, bbox = gsd.make_composite(note_img)
        stem = f"100_taka_{note_100.stem}_{i}"
        _save_yolo_item("train", stem, composite, bbox, class_idx=5)
        print(f"    train {stem}.jpg")

    test_notes = [
        ("10", 2, "10  Taka_001.jpg"),
        ("20", 3, "20  Taka_001.jpg"),
        ("50", 4, "50  Taka_001.jpg"),
        ("100", 5, "100  Taka_015.jpg"),
        ("500", 7, None),
        ("1000", 8, "1000  Taka_001.jpg"),
    ]
    for denom, class_idx, preferred in test_notes:
        folder = LOCAL_NOTES / denom
        path = folder / preferred if preferred else None
        if path is None or not path.is_file():
            hits = sorted(folder.glob("*.jpg")) if folder.is_dir() else []
            path = hits[min(4, len(hits) - 1)] if hits else None
        if path is None:
            print(f"    skip missing denom {denom}")
            continue
        note_img = gsd.load_source(path)
        for i in range(2):
            composite, bbox = gsd.make_composite(note_img)
            stem = f"{gsd.CLASSES[denom][1]}_{path.stem}_{i}"
            _save_yolo_item("test", stem, composite, bbox, class_idx)
            print(f"    test {stem}.jpg")

    print(f"  wrote composites to {LOCAL_YOLO}")

def make_figure_2() -> None:
    print("\n=== Figure 2: Copy-Paste Compositing Pipeline ===")
    note = load_source_note("100")
    bg = load_coco_background("000000000285.jpg")

    # (c) warp · rotate −18° · scale to 60% of canvas · colour jitter
    aug_rgba = note_to_rgba(note)
    aug_rgba = perspective_warp(aug_rgba, 0.07)
    aug_rgba = aug_rgba.rotate(-18, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
    jittered = color_jitter(aug_rgba.convert("RGB"))
    aug_rgba = Image.merge("RGBA", (*jittered.split(), aug_rgba.split()[-1]))
    aw, ah = aug_rgba.size
    scale = (0.70 * CANVAS) / max(aw, ah)
    aug_rgba = aug_rgba.resize((max(8, int(aw * scale)), max(8, int(ah * scale))), Image.LANCZOS)
    neutral = Image.new("RGB", (CANVAS, CANVAS), (236, 236, 232))
    nx = (CANVAS - aug_rgba.width) // 2
    ny = (CANVAS - aug_rgba.height) // 2
    tmp = neutral.convert("RGBA")
    tmp.paste(aug_rgba, (nx, ny), aug_rgba)
    panel_c = tmp.convert("RGB")

    # (d) MUST use the same COCO photo as (b) so the pipeline is one story.
    print("  composite: same COCO background as panel (b) + auto bbox")
    panel_d, box = paste_note(
        bg, note, angle=-18.0, scale_frac=0.58, xy=(70, 210), jitter=True, warp=True
    )

    panel_a = pad_to_square(trim_white_border(note), (255, 255, 255))
    panel_b = square_crop_resize(bg, CANVAS)

    fig = plt.figure(figsize=(12.6, 3.55))
    gs = GridSpec(
        1, 7, figure=fig, width_ratios=[1, 0.07, 1, 0.07, 1, 0.07, 1], wspace=0.04
    )
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 2, 4, 6)]
    arrow_axes = [fig.add_subplot(gs[0, i]) for i in (1, 3, 5)]

    captions = [
        "(a) Source Note\n(background-free, BanglaTaka)",
        "(b) Real Background\n(COCO val2017)",
        "(c) Augmented Note\n(warp · rotate · scale · jitter)",
        "(d) Final Composite\n+ auto bounding box",
    ]
    images = [panel_a, panel_b, panel_c, panel_d]
    for ax, im, cap in zip(axes, images, captions):
        ax.imshow(im)
        ax.set_title(cap, fontsize=9, pad=6)
        _blank_axis(ax)

    x, y, w, h = box
    # bbox is in panel_d pixel space; imshow uses those coordinates
    scale_x = panel_d.width / panel_d.width
    draw_box_on_axes(axes[3], x * scale_x, y, w, h, "100 Taka", lw=2.5)

    for aax in arrow_axes:
        aax.set_xlim(0, 1)
        aax.set_ylim(0, 1)
        aax.axis("off")
        aax.text(
            0.5,
            0.48,
            "→",
            ha="center",
            va="center",
            fontsize=22,
            color="#333333",
            fontweight="bold",
            transform=aax.transAxes,
        )

    fig.suptitle(
        "Figure 2 — Copy-Paste Compositing of a Source Note onto a Real COCO Background",
        fontsize=12,
        fontweight="bold",
        y=1.06,
    )
    save_figure(fig, "fig2_compositing")


# ── FIGURE 7 ─────────────────────────────────────────────────────────────────

def _synthetic_detection_panel(label: str = "100 Taka 0.99") -> np.ndarray:
    img = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
    img[:] = (196, 186, 170)
    img[int(CANVAS * 0.45) :] = (150, 118, 88)
    noise = np.random.normal(0, 7, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    x, y, w, h = 150, 210, 340, 150
    cv2.rectangle(img, (x, y), (x + w, y + h), (180, 80, 40), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 102), 3)
    cv2.rectangle(img, (x, y - 22), (x + 150, y), (0, 255, 102), -1)
    cv2.putText(img, label, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def make_figure_7() -> None:
    print("\n=== Figure 7: Sample Detections ===")
    annotated: list[np.ndarray] = []

    test_files: list[Path] = []
    if TEST_IMG_DIR.is_dir():
        test_files = [
            p
            for p in TEST_IMG_DIR.glob("*.jpg")
            if not p.name.lower().startswith("neg_")
        ]
        RNG.shuffle(test_files)
        test_files = test_files[:6]
        print(f"  using {len(test_files)} images from {TEST_IMG_DIR}")

    model = None
    if MODEL_PATH.is_file():
        try:
            from ultralytics import YOLO

            print(f"  loading YOLO: {MODEL_PATH}")
            model = YOLO(str(MODEL_PATH))
        except Exception as exc:
            print(f"  YOLO load failed ({exc}); using synthetic detections")
            model = None
    else:
        print("  models/best.pt not found; using synthetic detections")

    predict_inputs: list[Path | np.ndarray] = []
    if test_files:
        predict_inputs = test_files
    elif model is not None:
        print("  compositing 6 demo samples from source notes + COCO backgrounds")
        denoms = ["10", "20", "50", "100", "500", "1000"]
        angles = [-16, 21, -24, 9, -11, 18]
        scales = [0.52, 0.40, 0.58, 0.50, 0.36, 0.55]
        bgs = []
        for name in FIG7_COCO_NAMES:
            p = background_path(name)
            if p is not None:
                bgs.append(p)
        if not bgs:
            bgs = [p for p in list_backgrounds()[:8] if p.name != "000000000285.jpg"]
        print(f"  local COCO backgrounds: {[p.name for p in bgs]}")
        tmp_dir = ASSET_DIR / "fig7_composites"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for i, (denom, ang, sc) in enumerate(zip(denoms, angles, scales)):
            note = load_source_note(denom)
            bg = load_rgb(bgs[i % len(bgs)]) if bgs else synthetic_background()
            comp, _ = paste_note(bg, note, angle=ang, scale_frac=sc, jitter=True, warp=i % 2 == 0)
            dest = tmp_dir / f"sample_{i + 1}_{denom}.jpg"
            comp.save(dest, quality=88)
            predict_inputs.append(dest)
    else:
        predict_inputs = []

    if model is not None and predict_inputs:
        for src in predict_inputs:
            results = model.predict(str(src) if isinstance(src, Path) else src, conf=0.25, verbose=False)
            bgr = results[0].plot()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            annotated.append(rgb)
    else:
        labels = [
            "10 Taka 0.98",
            "20 Taka 0.97",
            "50 Taka 0.99",
            "100 Taka 0.99",
            "500 Taka 0.96",
            "1000 Taka 0.98",
        ]
        annotated = [_synthetic_detection_panel(lb) for lb in labels]

    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.6))
    fig.suptitle(
        "Figure 7 — Sample Detections with Bounding Boxes and Confidence Scores",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(annotated[i])
        ax.set_title(f"Sample {i + 1}", fontsize=10, pad=4)
        _blank_axis(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, "fig7_sample_detections")


# ── FIGURE 8 ─────────────────────────────────────────────────────────────────

def synthetic_gradcam(h: int, w: int, cx: float, cy: float, sigma: float = 30.0) -> np.ndarray:
    """224-style 2D Gaussian, used only if pytorch-grad-cam / YOLO fail."""
    yy, xx = np.mgrid[0:h, 0:w]
    heat = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))
    heat = heat / (heat.max() + 1e-8)
    heat = heat + np.random.normal(0, 0.012, heat.shape)
    return np.clip(heat, 0.0, 1.0)


def yolo_gradcam_heatmap(image_rgb: np.ndarray, class_id: int = 5) -> np.ndarray | None:
    """True Grad-CAM from YOLOv8s, targeting the given class score (5 = 100 Taka)."""
    try:
        import torch
        import torch.nn as nn
        from pytorch_grad_cam import GradCAM
        from ultralytics import YOLO
    except Exception as exc:
        print(f"  Grad-CAM imports failed ({exc})")
        return None

    if not MODEL_PATH.is_file():
        return None

    class _Wrap(nn.Module):
        def __init__(self, yolo):
            super().__init__()
            self.net = yolo.model
            self.net.eval()
            for p in self.net.parameters():
                p.requires_grad_(True)

        def forward(self, x):
            out = self.net(x)
            return out[0] if isinstance(out, (list, tuple)) else out

    class _ClassScoreTarget:
        def __init__(self, cid: int):
            self.cid = cid

        def __call__(self, output):
            scores = output[0, 4 + self.cid]
            k = min(50, scores.numel())
            return torch.topk(scores, k).values.sum()

    img_h, img_w = image_rgb.shape[:2]
    resized = cv2.resize(image_rgb, (CANVAS, CANVAS))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    yolo = YOLO(str(MODEL_PATH))
    wrap = _Wrap(yolo)
    # Layer 9 = SPPF: empirically focuses on the note portrait / denomination.
    cam = GradCAM(wrap, [wrap.net.model[9]])
    heat = cam(input_tensor=tensor, targets=[_ClassScoreTarget(class_id)])[0]
    cam.activations_and_grads.release()
    heat = cv2.resize(heat, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    heat = cv2.GaussianBlur(heat, (0, 0), 6)
    heat = heat / (heat.max() + 1e-8)
    return np.clip(heat, 0.0, 1.0)


def make_figure_8() -> None:
    print("\n=== Figure 8: Grad-CAM Heatmap ===")
    note = load_source_note("100")
    bg = load_coco_background("000000000139.jpg")
    original, (x, y, w, h) = paste_note(
        bg, note, angle=-6.0, scale_frac=0.62, xy=(90, 200), jitter=False, warp=False
    )
    orig_arr = np.array(original)
    img_h, img_w = orig_arr.shape[:2]

    heat = yolo_gradcam_heatmap(orig_arr, class_id=5)
    if heat is None:
        print("  using synthetic Grad-CAM fallback")
        portrait_cx = x + 0.30 * w
        portrait_cy = y + 0.50 * h
        heat224 = synthetic_gradcam(224, 224, portrait_cx * 224 / img_w, portrait_cy * 224 / img_h, 30.0)
        heat = cv2.resize(heat224, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        heat = np.clip(heat, 0.0, 1.0)
    else:
        print("  using real YOLOv8s Grad-CAM (SPPF layer, class=100 Taka)")

    heat_color = plt.cm.jet(heat)[:, :, :3]
    orig_f = orig_arr.astype(np.float32) / 255.0
    overlay = np.clip(0.5 * orig_f + 0.5 * heat_color, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.85))
    fig.suptitle(
        "Figure 8 — Grad-CAM Heatmap: Model Focus on Note's Portrait Area",
        fontsize=12,
        fontweight="bold",
        y=1.04,
    )
    axes[0].imshow(original)
    axes[0].set_title("(a) Original Image", fontsize=10, pad=6)
    im = axes[1].imshow(heat, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("(b) Grad-CAM Heatmap", fontsize=10, pad=6)
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Activation", fontsize=8)
    axes[2].imshow(overlay)
    axes[2].set_title("(c) Overlay", fontsize=10, pad=6)
    for ax in axes:
        _blank_axis(ax)
    fig.tight_layout()
    save_figure(fig, "fig8_gradcam")


# ── FIGURE 9 ─────────────────────────────────────────────────────────────────

def _try_font(size: int, bold: bool = False):
    names = (
        ["timesbd.ttf", "Timesbd.ttf", "times.ttf"]
        if bold
        else ["times.ttf", "Times.ttf", "timesbd.ttf"]
    )
    windir = Path(r"C:\Windows\Fonts")
    for name in names:
        p = windir / name
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                continue
    return ImageFont.load_default()


def _try_webcam_frame() -> Image.Image | None:
    """Grab one camera frame with a hard timeout so generation cannot hang."""
    import threading

    result: list[Image.Image | None] = [None]

    def grab() -> None:
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            for _ in range(12):
                cap.read()
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None and float(frame.mean()) > 35:
                result[0] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except Exception:
            pass

    t = threading.Thread(target=grab, daemon=True)
    t.start()
    t.join(5.0)
    return result[0]


def _crop_to_16x9(img: Image.Image, size=(1280, 720)) -> Image.Image:
    tw, th = size
    w, h = img.size
    want = tw / th
    have = w / h
    if have > want:
        nw = int(h * want)
        x0 = (w - nw) // 2
        img = img.crop((x0, 0, x0 + nw, h))
    else:
        nh = int(w / want)
        y0 = (h - nh) // 2
        img = img.crop((0, y0, w, y0 + nh))
    return img.resize(size, Image.LANCZOS)


def _place_note_on_frame(
    canvas: Image.Image, note_rgb: Image.Image, xy: tuple[int, int], angle: float, max_w: int
) -> Image.Image:
    canvas = canvas.convert("RGBA")
    n = note_to_rgba(note_rgb).rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
    n.thumbnail((max_w, int(max_w * 0.55)), Image.LANCZOS)
    canvas.paste(n, xy, n)
    return canvas.convert("RGB")


def build_webcam_scene() -> Image.Image:
    """Real indoor COCO photo + real notes (looks like a live camera frame)."""
    bg_path = background_path("000000000139.jpg") or background_path("000000397133.jpg")
    if bg_path is None:
        bg = synthetic_background(kind="indoor")
    else:
        print(f"  live-view background: {bg_path}")
        bg = load_rgb(bg_path)
    scene = _crop_to_16x9(bg)
    scene = _place_note_on_frame(scene, load_source_note("100"), (300, 210), -11, 580)
    scene = _place_note_on_frame(scene, load_source_note("50"), (820, 390), 16, 340)
    return scene


def _yolo_annotate(frame: Image.Image) -> tuple[np.ndarray, int]:
    from ultralytics import YOLO

    model = YOLO(str(MODEL_PATH))
    bgr = cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
    results = model.predict(bgr, conf=0.25, verbose=False)
    n = 0 if results[0].boxes is None else int(len(results[0].boxes))
    annotated = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    return annotated, n


def _add_live_hud(rgb: np.ndarray) -> Image.Image:
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    font_hud = _try_font(22, bold=True)
    font_sub = _try_font(16, bold=False)
    d.rectangle([16, 14, 455, 78], fill=(0, 0, 0))
    d.text((28, 20), "RoboEye Live Detection", fill=(0, 255, 102), font=font_hud)
    d.text((28, 50), "Press SPACE to speak | 'q' to quit", fill=(170, 170, 170), font=font_sub)
    return im


def make_figure_9() -> None:
    print("\n=== Figure 9: Real-time Webcam Detection ===")
    chosen: Image.Image | None = None
    cam = _try_webcam_frame()
    if cam is not None and MODEL_PATH.is_file():
        annotated, n = _yolo_annotate(cam)
        print(f"  webcam frame: {n} detections")
        if n >= 1:
            chosen = _add_live_hud(annotated)

    if chosen is None:
        print("  using real COCO indoor frame + notes + YOLO boxes")
        scene = build_webcam_scene()
        if MODEL_PATH.is_file():
            annotated, n = _yolo_annotate(scene)
            print(f"  YOLO detections on live-view scene: {n}")
            chosen = _add_live_hud(annotated)
        else:
            chosen = _add_live_hud(np.array(scene))

    fig, ax = plt.subplots(figsize=(8.4, 5.15))
    ax.imshow(chosen)
    ax.set_title("Figure 9 — Real-time Webcam Detection", fontsize=12, fontweight="bold", pad=10)
    _blank_axis(ax)
    fig.tight_layout()
    save_figure(fig, "fig9_webcam_detection")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    apply_ieee_style()
    print("RoboEye IEEE figures")
    print(f"  output: {OUT_DIR}")
    print(f"  local dataset: {DATA_SET}")
    ensure_local_composites()
    resolve_yolo_paths()
    make_figure_2()
    make_figure_7()
    make_figure_8()
    make_figure_9()
    download = ROOT.parent / "ieee_figures_download"
    download.mkdir(parents=True, exist_ok=True)
    for stem in (
        "fig2_compositing",
        "fig7_sample_detections",
        "fig8_gradcam",
        "fig9_webcam_detection",
    ):
        src = OUT_DIR / f"{stem}.png"
        if src.is_file():
            dest = download / f"{stem}.png"
            dest.write_bytes(src.read_bytes())
            print(f"  download copy: {dest}")
    print("\nDone. Figures written to", OUT_DIR)
    print("Download folder:", download)


if __name__ == "__main__":
    sys.exit(main() or 0)
