"""Train a YOLOv8s model on the Bangla currency dataset (v2 — real backgrounds)."""

from ultralytics import YOLO

DATA_CONFIG = "data/data.yaml"
EPOCHS = 80
IMG_SIZE = 640
MODEL = "yolov8s.pt"  # small: better accuracy than nano, still real-time on GPU


def main():
    model = YOLO(MODEL)
    model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device=0,
        patience=15,
        # colour / lighting augmentation
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        # geometric augmentation
        degrees=20,
        perspective=0.0005,
        scale=0.5,
        translate=0.1,
        shear=5,
        # no flips — currency orientation matters
        fliplr=0.0,
        flipud=0.0,
        # compositional
        mosaic=1.0,
        mixup=0.15,
    )


if __name__ == "__main__":
    main()
