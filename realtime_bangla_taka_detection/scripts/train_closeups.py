"""Fine-tune the deployed Taka detector with hand-held close-ups added to its training pictures (14 October 2026).

Starts from models/best.pt (YOLOv8s, trained 80 epochs on the copy-paste set). Training data: the original training
split plus 6,000 close-ups from scripts/dataset/make_closeups.py (same source notes, training split only).
Validation during training: the original synthetic validation split. Every epoch is saved; the epoch is chosen
afterwards by scripts/eval/eval_closeup_ft.py on NSTU validation photos, never on a test set.

Memory (measured during the run on the 16 GB laptop): the trainer about 3.0 GB, each data-loading worker 0.8 to
1.0 GB, about 8.3 GB in all with workers=2 (Ultralytics starts extra workers for validation). The first run used
workers=4 and crashed with "Unable to allocate" when a Whisper study was started beside it. So: workers=2, one heavy
job at a time, and a crashed run is resumed from its last epoch instead of being started again.

    python scripts/train_closeups.py            # starts, or resumes the unfinished run if there is one
    python scripts/train_closeups.py --fresh    # starts again from models/best.pt
"""
import sys
from pathlib import Path

from ultralytics import YOLO

LAST = Path("runs/detect/runs/closeup_ft/seed42/weights/last.pt")     # where Ultralytics writes this run


def main():
    if LAST.exists() and "--fresh" not in sys.argv:
        print("resuming from", LAST, flush=True)
        YOLO(str(LAST)).train(resume=True)
        return
    YOLO("models/best.pt").train(
        data="data/data_closeups.yaml", epochs=12, imgsz=640, device=0, batch=16, workers=2, seed=42,
        optimizer="SGD", lr0=0.002, lrf=0.1, momentum=0.937, weight_decay=0.0005, warmup_epochs=1,
        hsv_h=0.02, hsv_s=0.7, hsv_v=0.4, degrees=20, perspective=0.0005, scale=0.5, translate=0.1, shear=5,
        fliplr=0.0, flipud=0.0, mosaic=0.5, mixup=0.1, close_mosaic=3,
        save_period=1, patience=0, project="runs/closeup_ft", name="seed42", exist_ok=True, plots=False,
    )


if __name__ == "__main__":
    main()
