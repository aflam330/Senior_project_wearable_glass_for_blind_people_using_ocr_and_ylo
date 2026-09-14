"""Randomly sample images from the test set, run the trained model on them,
and check whether the predicted denomination matches the ground-truth label.
Saves annotated images for visual review and prints an accuracy summary.
"""

import random
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = str(ROOT / "models" / "best.pt")
_LOCAL = ROOT.parent / "data set" / "currency_yolo_data"
_LEGACY = Path(r"C:\currency_yolo_data")
_DATA = _LOCAL if (_LOCAL / "test" / "images").exists() else _LEGACY
TEST_IMAGES = _DATA / "test" / "images"
TEST_LABELS = _DATA / "test" / "labels"
NUM_SAMPLES = 30
CONF_THRESHOLD = 0.25


def true_class_id(label_path: Path) -> int:
    line = label_path.read_text().strip().splitlines()[0]
    return int(line.split()[0])


def main():
    model = YOLO(MODEL_PATH)

    all_images = sorted(TEST_IMAGES.glob("*.jpg"))
    random.seed()  # true random each run
    samples = random.sample(all_images, min(NUM_SAMPLES, len(all_images)))

    correct = 0
    missed = 0
    wrong = []

    for img_path in samples:
        label_path = TEST_LABELS / f"{img_path.stem}.txt"
        true_id = true_class_id(label_path)
        true_name = model.names[true_id]

        results = model.predict(source=str(img_path), conf=CONF_THRESHOLD, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            missed += 1
            wrong.append((img_path.name, true_name, "NO DETECTION"))
            continue

        best_idx = boxes.conf.argmax().item()
        pred_id = int(boxes.cls[best_idx].item())
        pred_conf = boxes.conf[best_idx].item()
        pred_name = model.names[pred_id]

        if pred_id == true_id:
            correct += 1
        else:
            wrong.append((img_path.name, true_name, f"{pred_name} ({pred_conf:.2f})"))

    print(f"\n=== Random test check: {len(samples)} images ===")
    print(f"Correct:       {correct}/{len(samples)}")
    print(f"Missed (none): {missed}/{len(samples)}")
    print(f"Wrong:         {len(samples) - correct - missed}/{len(samples)}")
    print(f"Accuracy:      {correct / len(samples) * 100:.1f}%")

    if wrong:
        print("\nMismatches:")
        for name, true_name, pred in wrong:
            print(f"  {name}: true={true_name}, pred={pred}")

    print("\n=== Saving annotated predictions ===")
    results = model.predict(
        source=[str(p) for p in samples],
        save=True,
        project="runs/detect/results",
        name="random_test_check",
        conf=CONF_THRESHOLD,
        exist_ok=True,
    )
    print(f"Annotated images saved to: {results[0].save_dir}")


if __name__ == "__main__":
    main()
