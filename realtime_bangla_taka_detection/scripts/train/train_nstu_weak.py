"""Fine-tune the hand-held detector on data/data_nstu_weak.yaml (scripts/dataset/make_nstu_weak.py).
Same hyper-parameters as the close-up fine-tune (runs/.../closeup_ft/seed42/args.yaml); every epoch is saved so the
epoch can be chosen on NSTU VALIDATION by scripts/eval/eval_nstu_weak.py. NSTU test is not read here."""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":       # Windows: the data loader workers re-import this file
  YOLO(str(ROOT / "models" / "best_handheld.pt")).train(
    data=str(ROOT / "data" / "data_nstu_weak.yaml"), epochs=10, imgsz=640, batch=16, workers=2, optimizer="SGD",
    lr0=0.002, lrf=0.1, warmup_epochs=1, seed=42, patience=0, close_mosaic=3, amp=True, degrees=20, fliplr=0.0,
    mosaic=0.5, save_period=1, project=str(ROOT / "runs" / "nstu_weak"), name="seed42", exist_ok=True, plots=False)
