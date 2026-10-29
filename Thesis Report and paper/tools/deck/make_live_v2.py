# Smaller copies of the v2 live-test photos for the Live Detection Tests slide.
from pathlib import Path
from PIL import Image, ImageOps

d = Path(__file__).parent
src = d / "../../Picture of testing/v2"
for n in (16, 18, 19, 24, 28, 29):
    im = ImageOps.exif_transpose(Image.open(src / f"v2 ({n}).jpeg")).convert("RGB")
    im.thumbnail((1400, 1400))
    im.save(d / "assets" / f"live_v2_{n}.jpg", quality=90)
print("ok")
