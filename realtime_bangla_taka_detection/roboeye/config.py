"""Shared paths and constants for the RoboEye stack."""

from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
MODELS_DIR = ROOT / "models"
DATA_SET = WORKSPACE / "data set"

JAALTAKA_DIR = DATA_SET / "JaalTaka"
COUNTERFEIT_DIR = DATA_SET / "Bangladeshi Counterfeit Currency Image Dataset"
BANGATAKA_DIR = DATA_SET / "Bangladeshi_Paper_Currency_Raw" / "Bangladeshi_Paper_Currency_Raw"
COCO_VAL = DATA_SET / "coco2017" / "val2017"
YOLO_DATA_DIR = DATA_SET / "currency_yolo_data"

YOLO_WEIGHTS = MODELS_DIR / "best.pt"
AUTH_WEIGHTS = MODELS_DIR / "authenticity_cnn_vit.pt"
DUAL_HEAD_WEIGHTS = MODELS_DIR / "dual_head_auth.pt"
PROTO_WEIGHTS = MODELS_DIR / "authenticity_prototypes.pt"
FER_ONNX = MODELS_DIR / "emotion-ferplus-8.onnx"
ONNX_WEIGHTS = MODELS_DIR / "best.onnx"
INT8_ONNX_WEIGHTS = MODELS_DIR / "best_int8.onnx"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = {
    0: "2_taka",
    1: "5_taka",
    2: "10_taka",
    3: "20_taka",
    4: "50_taka",
    5: "100_taka",
    6: "200_taka",
    7: "500_taka",
    8: "1000_taka",
}

# Physical note sizes in millimetres (Bangladesh Bank series, approx.).
NOTE_SIZE_MM = {
    "2_taka": (100.0, 60.0),
    "5_taka": (117.0, 62.0),
    "10_taka": (123.0, 64.0),
    "20_taka": (130.0, 62.0),
    "50_taka": (130.0, 62.0),
    "100_taka": (140.0, 62.0),
    "200_taka": (140.0, 65.0),
    "500_taka": (152.0, 65.0),
    "1000_taka": (160.0, 70.0),
}

FER_LABELS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]

HAPTIC_PATTERNS = {
    "genuine": [(80, 80), (80, 80)],
    "counterfeit": [(220, 80), (220, 80), (220, 120)],
    "unknown": [(140, 160)],
    "low_conf": [(40, 40), (40, 40), (40, 40)],
    "straighten": [(100, 60), (40, 80), (100, 60)],
    "detect": [(50, 50)],
}

# GPIO pin for a vibration motor on Raspberry Pi (BCM).
HAPTIC_GPIO_PIN = 13
