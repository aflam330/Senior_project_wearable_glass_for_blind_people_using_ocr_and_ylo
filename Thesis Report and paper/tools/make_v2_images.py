"""Second-prototype pictures (26 October 2026): collage for the title slide and the thesis, and close-ups for tables.

Sources: the four photographs in "Thesis Report and paper/Picture of testing/v2/" (back, front three-quarter, side with
buttons, side portrait).
    python make_v2_images.py
"""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Picture of testing" / "v2"
FIG = ROOT / "thesis report" / "My report" / "DIagram fig tab"
DECK = ROOT / "tools" / "deck" / "assets"
back, front, side, portrait = (Image.open(SRC / f).convert("RGB") for f in ("v2_back.jpg", "v2_front.jpg", "v2_side_buttons.jpg", "v2_side_portrait.jpg"))


def cover(im, w, h, cx=0.5, cy=0.5):
    """Scale to cover w x h, then crop around (cx, cy) given as fractions of the scaled image."""
    k = max(w / im.width, h / im.height)
    im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
    x = min(max(round(cx * im.width - w / 2), 0), im.width - w)
    y = min(max(round(cy * im.height - h / 2), 0), im.height - h)
    return im.crop((x, y, x + w, y + h))


# collage: front view across the top; portrait side view bottom left; back and button side stacked bottom right
W, H, G = 1180, 1500, 10
top_h = round(W * 9 / 16)
bot_h = H - top_h - G
left_w = round(bot_h * 9 / 16)
right_w = W - left_w - G
half = (bot_h - G) // 2
c = Image.new("RGB", (W, H), (255, 255, 255))
c.paste(cover(front, W, top_h), (0, 0))
c.paste(cover(portrait, left_w, bot_h, cy=0.45), (0, top_h + G))
c.paste(cover(side, right_w, half, cx=0.55, cy=0.45), (left_w + G, top_h + G))
c.paste(cover(back, right_w, bot_h - half - G, cy=0.55), (left_w + G, top_h + G + half + G))
c.save(DECK / "v2_collage.jpg", quality=90)
c.save(FIG / "sara_v2_collage.jpg", quality=90)

# close-ups for the hardware table and the comparison tables (4:3)
def box(im, x0, y0, x1, y1, name):
    """A 4:3 close-up cut from the original photograph (pixel box), saved at 800 x 600."""
    im.crop((x0, y0, x1, y1)).resize((800, 600), Image.LANCZOS).save(FIG / name, quality=90)


box(side, 500, 190, 940, 520, "v2_buttons.jpg")        # the five-button board on the temple
box(side, 420, 340, 700, 550, "v2_headphone.jpg")      # bone-conduction headphone
box(front, 220, 170, 660, 500, "v2_frame.jpg")         # frame with camera and LED
box(back, 440, 215, 1000, 635, "v2_enclosure.jpg")     # Pi 5 enclosure at the back of the head
cover(front, 900, 675, cx=0.45, cy=0.5).save(FIG / "sara_v2_thumb.jpg", quality=90)    # thumbnail of SARA
for f in ("v2_front.jpg", "v2_back.jpg", "v2_side_buttons.jpg", "v2_side_portrait.jpg"):
    Image.open(SRC / f).save(DECK / f, quality=90)
print("ok")
