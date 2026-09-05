"""Build an illustrative figure of the copy-paste compositing process using
REAL assets: a source note, a COCO background, and the actual generated
composite with its YOLO bounding box drawn from the label file.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image

OUT = Path("results/report_assets")
OUT.mkdir(parents=True, exist_ok=True)

NOTE = Path(r"C:\Users\memod\OneDrive\Desktop\currency\data"
            r"\A Diverse Image Dataset for Bangladeshi Currency Recognition"
            r"\extracted\Bangladeshi_Paper_Currency_Raw\100\100  Taka_001.jpg")
BG = Path(r"C:\currency_backgrounds\images\000000000285.jpg")
COMPOSITE = Path(r"C:\currency_yolo_data\train\images\100_taka_100  Taka_001_1.jpg")
LABEL = Path(r"C:\currency_yolo_data\train\labels\100_taka_100  Taka_001_1.txt")


def read_bbox(label_path, img_w, img_h):
    line = label_path.read_text().strip().splitlines()[0]
    _, xc, yc, bw, bh = map(float, line.split())
    x = (xc - bw / 2) * img_w
    y = (yc - bh / 2) * img_h
    return x, y, bw * img_w, bh * img_h


fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
fig.suptitle("Figure 2 — Copy-Paste Compositing of a Source Note onto a Real COCO Background",
             fontsize=13, weight="bold", y=1.02)

titles = [
    "(a) Source Note\n(background-free, BanglaTaka)",
    "(b) Real Background\n(COCO val2017)",
    "(c) Augmented Note\n(warp · rotate · scale · jitter)",
    "(d) Final Composite\n+ auto bounding box",
]

# (a) source note
note = Image.open(NOTE).convert("RGB")
axes[0].imshow(note)

# (b) background
bg = Image.open(BG).convert("RGB")
axes[1].imshow(bg)

# (c) augmented note preview (just show the note rotated/scaled on transparent-ish)
note_aug = note.rotate(-18, expand=True, fillcolor=(255, 255, 255))
note_aug = note_aug.resize((int(note_aug.width * 0.6), int(note_aug.height * 0.6)))
axes[2].imshow(note_aug)

# (d) composite with bbox
comp = Image.open(COMPOSITE).convert("RGB")
axes[3].imshow(comp)
x, y, w, h = read_bbox(LABEL, comp.width, comp.height)
axes[3].add_patch(Rectangle((x, y), w, h, linewidth=2.5,
                            edgecolor="#00FF66", facecolor="none"))
axes[3].text(x, max(y - 8, 10), "100_taka", color="black", fontsize=9, weight="bold",
             bbox=dict(facecolor="#00FF66", edgecolor="none", pad=1.5))

for ax, t in zip(axes, titles):
    ax.set_title(t, fontsize=9.5)
    ax.axis("off")

# arrows between panels
for i in range(3):
    x_pos = 0.25 * (i + 1) + 0.005
    fig.text(x_pos, 0.46, "→", fontsize=26, ha="center", va="center",
             color="#333333", weight="bold")

plt.tight_layout()
plt.savefig(OUT / "compositing.png", dpi=170, bbox_inches="tight")
plt.close()
print("saved compositing.png")
