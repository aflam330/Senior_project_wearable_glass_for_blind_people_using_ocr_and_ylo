"""Fine-tune the deployed Taka detector with hand-held close-ups added to its training pictures (14 October 2026).

Starts from models/best.pt (YOLOv8s, trained 80 epochs on the copy-paste set). Training data: the original training
split plus 6,000 close-ups from scripts/dataset/make_closeups.py (same source notes, training split only).
Validation during training: the original synthetic validation split. Every epoch is saved; the epoch is chosen
afterwards by scripts/eval/eval_closeup_ft.py on NSTU validation photos, never on a test set.

    python scripts/train_closeups.py
"""
from ultralytics import YOLO


def main():
    YOLO("models/best.pt").train(
        data="data/data_closeups.yaml", epochs=12, imgsz=640, device=0, batch=16, workers=4, seed=42,
        optimizer="SGD", lr0=0.002, lrf=0.1, momentum=0.937, weight_decay=0.0005, warmup_epochs=1,
        hsv_h=0.02, hsv_s=0.7, hsv_v=0.4, degrees=20, perspective=0.0005, scale=0.5, translate=0.1, shear=5,
        fliplr=0.0, flipud=0.0, mosaic=0.5, mixup=0.1, close_mosaic=3,
        save_period=1, patience=0, project="runs/closeup_ft", name="seed42", exist_ok=True, plots=False,
    )


if __name__ == "__main__":
    main()
