"""Other text detectors in front of the same EasyOCR recognizer and the same final pipeline (Task 1 follow-up).

  east   OpenCV EAST (frozen_east_text_detection.pb), standard settings from the OpenCV text-detection sample
  db_td500 / db_ic15   OpenCV DB (DBNet, ResNet-18) ONNX models: the compiler-free way to run DBNet
  yolo   Daniil-Domino/yolo11x-text-detection (AGPL-3.0): trained on Russian handwritten school notebooks plus a few
         printed pages, i.e. NOT on scene text or Bangla; the only public general YOLO text detector found
Detector settings are the published defaults; nothing is tuned here. Boxes go to EasyOCR's recognizer, then the same
dominant-block rule, line ordering and cleanup rules as pipeline v2. Weights live in ocr_bench/detectors/ (not committed).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import final_pipeline as F, pipelines as P, postprocess as PO, text_region as T
from .run_task4 import prep_fn

DET = Path(__file__).resolve().parent / "detectors"


def _r32(x):
    return max(32, int(round(x / 32)) * 32)


@lru_cache(maxsize=8)
def _model(name: str, w: int, h: int):
    if name == "east":
        m = cv2.dnn.TextDetectionModel_EAST(str(DET / "frozen_east_text_detection.pb"))
        m.setConfidenceThreshold(0.5)
        m.setNMSThreshold(0.4)
        m.setInputParams(1.0, (w, h), (123.68, 116.78, 103.94), True)
        return m
    m = cv2.dnn.TextDetectionModel_DB(str(DET / ("DB_TD500_resnet18.onnx" if name == "db_td500" else "DB_IC15_resnet18.onnx")))
    m.setBinaryThreshold(0.3)
    m.setPolygonThreshold(0.5)
    m.setMaxCandidates(200)
    m.setUnclipRatio(2.0)
    m.setInputParams(1.0 / 255.0, (w, h), (122.67891434, 116.66876762, 104.00698793))
    return m


@lru_cache(maxsize=1)
def _yolo():
    from ultralytics import YOLO
    return YOLO(str(DET / "yolo11x_text_detection.pt"))


def detect(name: str, bgr: np.ndarray) -> list:
    """Axis-aligned boxes [x0, x1, y0, y1] in the coordinates of `bgr`."""
    H, W = bgr.shape[:2]
    if name == "yolo":
        res = _yolo().predict(bgr, verbose=False, imgsz=1280, conf=0.25)[0]
        return [[int(x0), int(x1), int(y0), int(y1)] for x0, y0, x1, y1 in res.boxes.xyxy.cpu().numpy()] if res.boxes is not None else []
    w, h = _r32(min(W, 1280)), _r32(H * min(W, 1280) / W)
    quads, *_ = _model(name, w, h).detect(bgr)
    out = []
    for q in quads:
        q = np.asarray(q, np.float32).reshape(-1, 2)
        x0, y0, x1, y1 = q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()
        if x1 - x0 > 4 and y1 - y0 > 4:
            out.append([int(max(0, x0)), int(min(W, x1)), int(max(0, y0)), int(min(H, y1))])
    return out


def make_reader(name: str):
    """The final pipeline (Tasks 2, 4, 5 choices) with detector `name` instead of CRAFT."""
    c = F.load_choices()
    prep = prep_fn(c["prep"])
    fix = PO.METHODS[c["post"]]
    min_conf = c["params"]["min_conf"]

    def read(bgr):
        h, w = bgr.shape[:2]
        big = cv2.resize(bgr, (1200, int(h * 1200 / w)), interpolation=cv2.INTER_CUBIC) if w < 1200 else bgr
        gray = prep(bgr)  # same preprocessing (and the same 1200 px size) as pipeline v2
        if gray.shape[:2] != big.shape[:2]:
            big = cv2.resize(big, (gray.shape[1], gray.shape[0]))
        boxes = detect(name, big)
        if not boxes:
            return ""
        H, W = gray.shape[:2]  # same padding EasyOCR gives its own CRAFT boxes (add_margin = 0.1 of the box height)
        boxes = [[max(0, x0 - int(0.1 * (y1 - y0))), min(W, x1 + int(0.1 * (y1 - y0))), max(0, y0 - int(0.1 * (y1 - y0))),
                  min(H, y1 + int(0.1 * (y1 - y0)))] for x0, x1, y0, y1 in boxes]
        det = P.reader("craft").recognize(gray, horizontal_list=boxes, free_list=[], decoder=c["params"]["decoder"], beamWidth=5,
                                          contrast_ths=c["params"]["contrast_ths"], adjust_contrast=c["params"]["adjust_contrast"], detail=1)
        block = T.dominant_block([b for b in T._boxes(det) if b["c"] >= min_conf])
        if not block:
            return ""
        return fix(P.join_lines([([[b["x0"], b["y0"]], [b["x1"], b["y0"]], [b["x1"], b["y1"]], [b["x0"], b["y1"]]], b["t"], b["c"])
                                 for b in block]))
    return read
