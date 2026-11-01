# Version 2 collage for the "Evolution of the Prototype" slide, from three team photos of the second prototype:
# left, the front with the UV lights on (whole photo); right, two crops stacked: the front with the camera housing,
# and the rear enclosure with the power bank. White gaps between them. Output: assets/v2_collage3.jpg
from pathlib import Path
from PIL import Image

A = Path(__file__).parent / "assets"
front_uv = Image.open(A / "v2_photo_front_uv.jpg").convert("RGB")      # 3540x2893
front = Image.open(A / "v2_photo_front.jpg").convert("RGB")            # 3213x5216
rear = Image.open(A / "v2_photo_top.jpg").convert("RGB")               # 3213x5437

H, GAP = 1600, 12                       # collage height in pixels; slide area is 9.02 x 4.62 in (aspect 1.95)
W = round(H * 9.02 / 4.62)
left_w = round(H * front_uv.width / front_uv.height)
right_w = W - left_w - GAP
cell_h = (H - GAP) // 2


def crop_to(im, aspect, cy):
    """Full-width crop of the given aspect (w/h), centred on the fraction cy of the height."""
    h = round(im.width / aspect)
    top = min(max(0, round(cy * im.height - h / 2)), im.height - h)
    return im.crop((0, top, im.width, top + h))


canvas = Image.new("RGB", (W, H), "white")
canvas.paste(front_uv.resize((left_w, H), Image.LANCZOS), (0, 0))
canvas.paste(crop_to(front, right_w / cell_h, 0.57).resize((right_w, cell_h), Image.LANCZOS), (left_w + GAP, 0))
canvas.paste(crop_to(rear, right_w / (H - cell_h - GAP), 0.38).resize((right_w, H - cell_h - GAP), Image.LANCZOS),
             (left_w + GAP, cell_h + GAP))
canvas.save(A / "v2_collage3.jpg", quality=90)
print("saved", canvas.size)
