---
license: mit
tags: [object-detection, yolov8, banknote, bangladesh, assistive-technology]
datasets: [BanglaTaka (Mendeley 3cv2sypkkh) composited on COCO val2017]
metrics: [map50, map50-95, top1-denomination-accuracy]
---

# Taka detector (YOLOv8s, 9 denominations)

**Files** (`realtime_bangla_taka_detection/models/`):

| File | Format | Size |
|---|---|---:|
| `best.pt` | PyTorch | 22.5 MB |
| `best.onnx` | FP32 ONNX | — |
| `best_int8.onnx` | INT8 ONNX (head and first layer kept in FP32) | 11.5 MB |
| `best.torchscript` | TorchScript | — |

**Classes:** 2, 5, 10, 20, 50, 100, 200, 500 and 1000 taka.

**Training:** Ultralytics YOLOv8s, up to 80 epochs, image size 640, no horizontal flips. Training data: 22,333 synthetic composites (5,073 BanglaTaka note crops pasted onto COCO backgrounds, 10 % negatives), split by source image 80/10/10 (`scripts/generate_synthetic_dataset.py`, `scripts/train.py`).

## Evaluation

| Data | Result | Source |
|---|---|---|
| Composite test (2,282 images) | P 0.995, R 0.996, mAP@0.5 0.995, mAP@0.5:0.95 0.849 | `results/training_v2/test_eval/test_metrics.json` |
| Bangla Money, independent (1,536) | 91.5 % correct denomination (95 % CI 90.0–92.8) | `results/external/external_taka.json` |
| Counterfeit-currency set, independent (1,286; 500/1000) | 92.6 % (91.1–93.9) | `results/external/detector_counterfeit_ds.json` |
| NSTU-BDTAKA hand-held close-ups (1,144) | **18.5 %** (16.4–20.9) | `results/external/external_taka.json` |
| One-taka notes, not a class (101) | 43.6 % announced as another value at conf 0.25; 22.8 % at the deployed conf 0.60 | `results/open_set/summary.json` |

Latency: 19.1 ms on an RTX 3050 GPU and 60.8 ms as INT8 ONNX on a Ryzen 7 5800H CPU (`GPU_SPEED.md`). Raspberry Pi 5 latency is NOT_MEASURED.

## Intended use

Naming the denomination of a whole, flat note held in front of a camera.

## Out of scope

- Close-ups of part of a note.
- Folded notes, or notes covered by fingers.
- Coins and foreign currency.
- Any genuineness decision.

The glass abstains below confidence 0.60, which was chosen on validation (`OPEN_SET_REJECTION.md`).

This card is not a Hugging Face upload.
