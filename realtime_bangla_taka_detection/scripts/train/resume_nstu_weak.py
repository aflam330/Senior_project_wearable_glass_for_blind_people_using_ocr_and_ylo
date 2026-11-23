"""Resume scripts/train/train_nstu_weak.py from its last checkpoint (same arguments, as saved in the run)."""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    YOLO(str(ROOT / "runs" / "nstu_weak" / "seed42" / "weights" / "last.pt")).train(resume=True)
