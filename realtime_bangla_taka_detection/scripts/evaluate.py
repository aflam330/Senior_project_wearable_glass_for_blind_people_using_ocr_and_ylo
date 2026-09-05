"""Evaluate the trained YOLOv8 model on the test split and save sample predictions."""

from pathlib import Path
import random

from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
DATA_CONFIG = "data/data.yaml"
TEST_IMAGES = Path(r"C:\currency_yolo_data\test\images")
NUM_SAMPLES = 10


def main():
    model = YOLO(MODEL_PATH)

    print("=== Running validation on test split ===")
    metrics = model.val(data=DATA_CONFIG, split="test", plots=True)

    print("\nOverall metrics:")
    print(f"  precision: {metrics.box.mp:.4f}")
    print(f"  recall:    {metrics.box.mr:.4f}")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")

    print("\nPer-class mAP50-95:")
    for idx, name in model.names.items():
        print(f"  {name}: {metrics.box.maps[idx]:.4f}")

    print(f"\nValidation artifacts (incl. confusion matrix) saved to: {metrics.save_dir}")

    print("\n=== Saving sample predictions on test images ===")
    all_images = sorted(TEST_IMAGES.glob("*.jpg"))
    random.seed(42)
    samples = random.sample(all_images, min(NUM_SAMPLES, len(all_images)))

    results = model.predict(
        source=[str(p) for p in samples],
        save=True,
        project="results",
        name="sample_predictions",
        conf=0.25,
    )
    print(f"Sample predictions saved to: {results[0].save_dir}")


if __name__ == "__main__":
    main()
