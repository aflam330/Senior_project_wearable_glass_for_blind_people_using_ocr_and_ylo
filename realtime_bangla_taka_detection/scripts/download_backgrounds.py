"""Download ~1500 background images from COCO val2017 for use in synthetic
dataset generation. Images are saved to C:/currency_backgrounds/.

No API key needed — COCO images are publicly hosted.
"""

import json
import os
import urllib.request
from pathlib import Path

ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
ANNOTATIONS_JSON = Path("C:/currency_backgrounds/instances_val2017.json")
OUT_DIR = Path("C:/currency_backgrounds/images")
NUM_IMAGES = 1500


def download_annotations():
    import zipfile, io
    if ANNOTATIONS_JSON.exists():
        print("Annotations already downloaded.")
        return
    ANNOTATIONS_JSON.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading COCO 2017 annotations (~241 MB)...")
    with urllib.request.urlopen(ANNOTATIONS_URL) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        target = "annotations/instances_val2017.json"
        with z.open(target) as src, open(ANNOTATIONS_JSON, "wb") as dst:
            dst.write(src.read())
    print(f"Saved annotations to {ANNOTATIONS_JSON}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    download_annotations()

    print("Loading annotation data...")
    with open(ANNOTATIONS_JSON) as f:
        data = json.load(f)

    all_images = data["images"]  # list of {id, file_name, coco_url, ...}
    print(f"Total images in val2017: {len(all_images)}")

    # shuffle for variety, pick NUM_IMAGES
    import random
    random.seed(0)
    random.shuffle(all_images)
    to_download = all_images[:NUM_IMAGES]

    existing = set(p.name for p in OUT_DIR.glob("*.jpg"))
    todo = [im for im in to_download if im["file_name"] not in existing]
    print(f"Already have: {len(existing)}  |  To download: {len(todo)}")

    for i, im in enumerate(todo, 1):
        url = im["coco_url"]
        dest = OUT_DIR / im["file_name"]
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"  [{i}/{len(todo)}] FAILED {im['file_name']}: {e}")
            continue
        if i % 100 == 0:
            print(f"  {i}/{len(todo)} downloaded...")

    total = len(list(OUT_DIR.glob("*.jpg")))
    print(f"\nDone. {total} background images in {OUT_DIR}")


if __name__ == "__main__":
    main()
