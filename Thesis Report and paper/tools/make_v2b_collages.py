"""Collages from the second batch of prototype-v2 photos (Picture of testing/v2/v2 (N).jpeg, 29 October 2026):
v2_hardware_collage.jpg (the unit) and v2_use_collage.jpg (staged currency use), for the thesis and the slides."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Picture of testing" / "v2"
FIG = ROOT / "thesis report" / "My report" / "DIagram fig tab"
DECK = ROOT / "tools" / "deck" / "assets"


def tile(n, w, h, anchor=(0.5, 0.5)):
    im = ImageOps.exif_transpose(Image.open(SRC / f"v2 ({n}).jpeg")).convert("RGB")
    return ImageOps.fit(im, (w, h), Image.LANCZOS, centering=anchor)


def collage(items, name, w=1200, h=800, gap=12):
    out = Image.new("RGB", (2 * w + gap, 2 * h + gap), "white")
    try:
        font = ImageFont.truetype("arialbd.ttf", 44)
    except OSError:
        font = ImageFont.load_default()
    for k, (n, label, anchor) in enumerate(items):
        t = tile(n, w, h, anchor)
        d = ImageDraw.Draw(t)
        tw = d.textlength(label, font=font)
        d.rounded_rectangle([18, 18, 18 + tw + 28, 18 + 62], radius=12, fill=(11, 77, 59))
        d.text((32, 26), label, font=font, fill="white")
        out.paste(t, ((k % 2) * (w + gap), (k // 2) * (h + gap)))
    for d in (FIG, DECK):
        out.save(d / name, quality=88)
    print(name, out.size)


collage([(6, "(a) Complete unit", (0.5, 0.5)), (8, "(b) UV light on", (0.5, 0.55)),
         (13, "(c) Raspberry Pi 5 in the rear enclosure", (0.5, 0.45)), (12, "(d) Buttons, enclosure and power bank", (0.5, 0.5))],
        "v2_hardware_collage.jpg")
# two full frames stacked, for the slide "SARA in action"
a, b = tile(9, 1600, 900), tile(17, 1600, 900)
stack = Image.new("RGB", (1600, 1816), "white")
stack.paste(a, (0, 0))
stack.paste(b, (0, 916))
stack.save(DECK / "v2_action_stack.jpg", quality=88)
print("v2_action_stack.jpg", stack.size)

collage([(9, "(a) Receiving a note", (0.5, 0.5)), (17, "(b) Note held up to the camera", (0.5, 0.5)),
         (21, "(c) Notes under the UV light", (0.5, 0.4)), (30, "(d) UV check, close view", (0.5, 0.4))],
        "v2_use_collage.jpg")
