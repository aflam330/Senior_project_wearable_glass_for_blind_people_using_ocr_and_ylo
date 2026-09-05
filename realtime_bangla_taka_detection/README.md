# Real-Time Bangladeshi Taka Banknote Detection

A real-time object-detection system that localises and classifies the nine
denominations of Bangladeshi Taka (2, 5, 10, 20, 50, 100, 200, 500, 1000)
from a live webcam, with **push-to-speak** voice announcements of the detected note.

Built with **YOLOv8s** (Ultralytics) fine-tuned on a synthetically composited
dataset that pastes real banknotes onto real-world (COCO) backgrounds to close
the synthetic-to-real domain gap.

---

## Results (held-out test set, 2,238 images)

| Metric | Value |
|--------|-------|
| Precision | 0.998 |
| Recall | 0.997 |
| mAP@0.5 | 0.995 |
| mAP@0.5:0.95 | 0.847 |
| Inference speed | 7.3 ms / image (≈135 FPS, RTX 4060) |

Full write-up: **[Bangla_Currency_Detection_Report.pdf](Bangla_Currency_Detection_Report.pdf)** (16 pages).

---

## Setup

```powershell
# from the project root
python -m venv venv
.\venv\Scripts\pip.exe install -r requirements.txt
```

The trained model is already saved in `models/best.pt` — no retraining needed to run detection.

---

## Run the live detector (with voice)

```powershell
.\venv\Scripts\python.exe scripts\realtime_detect.py
```

**Controls**
- `SPACE` — speak the name of the note currently detected on screen
- `q` / `ESC` — quit

Voice uses the offline Windows SAPI engine on a background thread, so it never
slows the video. Dark frames are auto-brightened before inference.

---

## Project structure

```
currency/
├── models/
│   ├── best.pt            # trained YOLOv8s weights (production model)
│   ├── best.torchscript   # TorchScript export
│   └── best.pts           # TorchScript (.pts alias)
├── data/
│   └── data.yaml          # dataset config (9 classes)
├── scripts/
│   ├── download_backgrounds.py       # fetch 1,500 COCO val2017 backgrounds
│   ├── generate_synthetic_dataset.py # composite notes+backgrounds -> YOLO dataset
│   ├── train.py                      # fine-tune YOLOv8s
│   ├── evaluate.py                   # test-set metrics + confusion matrix
│   ├── random_test_check.py          # random spot-check accuracy
│   ├── realtime_detect.py            # live webcam detection + push-to-speak TTS
│   ├── plot_results.py               # per-epoch training curves
│   ├── make_diagrams.py              # report methodology/architecture diagrams
│   ├── make_compositing_figure.py    # report compositing figure
│   └── build_report.py               # compile the PDF report
├── results/
│   ├── training_v2/       # metrics, confusion matrices, epoch curves
│   └── report_assets/     # generated diagrams/figures
├── Bangla_Currency_Detection_Report.pdf
├── requirements.txt
└── README.md
```

> Note: the generated image dataset lives outside the repo at
> `C:\currency_yolo_data\` (train/valid/test) and backgrounds at
> `C:\currency_backgrounds\`, to keep large files off OneDrive sync.

---

## Reproducing the pipeline (optional)

```powershell
# 1. download real backgrounds
.\venv\Scripts\python.exe scripts\download_backgrounds.py
# 2. build the synthetic detection dataset
.\venv\Scripts\python.exe scripts\generate_synthetic_dataset.py
# 3. train
.\venv\Scripts\python.exe scripts\train.py
# 4. evaluate
.\venv\Scripts\python.exe scripts\evaluate.py
```

---

## Data sources

- **Notes:** *A Diverse Image Dataset for Bangladeshi Currency Recognition*
  (BanglaTaka), Mendeley Data — 5,073 background-free images, 9 denominations.
  https://data.mendeley.com/datasets/3cv2sypkkh/1
- **Backgrounds:** COCO val2017 (1,500 images).

## Key techniques

- Copy-paste compositing with automatic YOLO bounding-box labels
- Domain randomisation: perspective warp, rotation ±30°, scale/HSV/brightness
  jitter, motion blur, Gaussian noise, JPEG re-compression, 10% negative images
- Transfer learning from COCO-pretrained YOLOv8s
- Offline text-to-speech (Windows SAPI) for accessibility
