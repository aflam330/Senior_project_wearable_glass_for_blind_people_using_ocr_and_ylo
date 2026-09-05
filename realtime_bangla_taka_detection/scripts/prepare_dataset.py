"""
Convert the BanglaTaka classification-style dataset (one folder per denomination,
note filling almost the entire frame) into a YOLOv8 detection dataset.

Each image gets a single full-frame bounding box labeled with its denomination class.
Images are split into train/valid/test and written to data/train, data/valid, data/test.
"""

import random
import shutil
from pathlib import Path

RANDOM_SEED = 42
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, valid, test

SOURCE_DIR = Path(
    r"C:\Users\memod\OneDrive\Desktop\currency\data"
    r"\A Diverse Image Dataset for Bangladeshi Currency Recognition"
    r"\extracted\Bangladeshi_Paper_Currency_Raw"
)
DEST_DIR = Path(r"C:\Users\memod\OneDrive\Desktop\currency\data")

# Folder name -> (class index, class name), ordered by denomination value
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

# Full-frame bounding box in YOLO format: x_center y_center width height (normalized)
FULL_FRAME_BOX = "0.5 0.5 1.0 1.0"


def main():
    random.seed(RANDOM_SEED)

    for split in ("train", "valid", "test"):
        (DEST_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DEST_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "valid": 0, "test": 0}

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

        n = len(images)
        n_train = int(n * SPLIT_RATIOS[0])
        n_valid = int(n * SPLIT_RATIOS[1])

        splits = (
            ["train"] * n_train
            + ["valid"] * n_valid
            + ["test"] * (n - n_train - n_valid)
        )

        for img_path, split in zip(images, splits):
            dest_img = DEST_DIR / split / "images" / img_path.name
            dest_label = DEST_DIR / split / "labels" / (img_path.stem + ".txt")

            shutil.copy2(img_path, dest_img)
            dest_label.write_text(f"{class_idx} {FULL_FRAME_BOX}\n")

            counts[split] += 1

        print(f"{class_name}: {n} images "
              f"(train={n_train}, valid={n_valid}, test={n - n_train - n_valid})")

    print("\nTotal:", counts)


if __name__ == "__main__":
    main()
