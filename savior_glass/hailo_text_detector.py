"""Text detector on the Raspberry Pi AI HAT (Hailo-8), for text mode when OCR_DETECTOR=hailo.

Only the step that FINDS the text moves to the chip; SARA's own Bangla + English letter recogniser still reads every
box on the CPU, and everything after it (main text block, reading order, line-by-line pages, rotation retry) is
unchanged. The chip model is Hailo's stock compile of the PaddleOCR PP-OCRv5 mobile text detector (DBNet, Hailo
Model Zoo v2.19, 544x960 uint8 in, normalisation inside the network, one probability map out). It is language-free:
it marks where text is, not which letters. CRAFT, SARA's CPU detector, can't go on the chip without Hailo's
Dataflow Compiler, which runs only on an x86 PC.

install(reader) replaces reader.detect with a function that returns EasyOCR's own format: the DBNet boxes go
through easyocr.utils.group_text_box with the same arguments CRAFT's boxes would get, so recognize() sees boxes of
the usual kind. On any problem install() leaves the reader as it was; if the chip fails during a capture the
reader goes back to its CPU detector for good and that capture is re-detected on the CPU.

Frame handling (config.OCR_HAILO_MODE): "tile" covers the picture at its own size with overlapping 544x960
windows merged by maximum (4 windows for the 1,200 x 900 picture text mode reads); "resize" fits the whole picture
into one window. Box extraction follows PaddleOCR's DBPostProcess (thresh 0.3, box 0.6, unclip 1.5, the model's
own inference.yml). Ported from ~/Desktop/glass_ocr/paddle_pipeline.py (HailoDet, db_boxes), tested there on the
glass_set pages (PADDLEOCR_G1_FINDINGS.md).
"""
from __future__ import annotations

import logging
import math
import os

import cv2
import numpy as np

import config

logger = logging.getLogger("smart_glass.ocr")


class HailoTextDetector:
    TH, TW = 544, 960

    def __init__(self, hef_path: str, mode: str = "tile") -> None:
        from hailo_device import open_model
        self.mode = mode
        self._model = open_model(hef_path)            # the AI HAT is shared with the other chip models (hailo_device.py)
        info = self._model.inputs[0]
        self.TH, self.TW = int(info.shape[0]), int(info.shape[1])
        self._out = self._model.outputs[0].name

    def _run(self, batch: np.ndarray) -> np.ndarray:
        return self._model.infer(batch)[self._out][..., 0]

    def prob(self, img: np.ndarray) -> np.ndarray:
        """Text probability for every pixel of img (H x W x 3 uint8)."""
        h, w = img.shape[:2]
        TH, TW = self.TH, self.TW
        if self.mode == "resize" or (h <= TH and w <= TW):
            s = min(TW / w, TH / h, 1.0 if self.mode == "tile" else 9.0)
            nw, nh = int(round(w * s)), int(round(h * s))
            canvas = np.zeros((TH, TW, 3), np.uint8)
            canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
            p = self._run(canvas[None])[0][:nh, :nw]
            return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else p
        if h < TH or w < TW:
            pad = np.zeros((max(h, TH), max(w, TW), 3), np.uint8)
            pad[:h, :w] = img
            img = pad
        H, W = img.shape[:2]
        ys = np.linspace(0, H - TH, math.ceil(H / TH)).round().astype(int)
        xs = np.linspace(0, W - TW, math.ceil(W / TW)).round().astype(int)
        tiles = [(y, x) for y in ys for x in xs]
        out = self._run(np.stack([img[y:y + TH, x:x + TW] for y, x in tiles]))
        p = np.zeros((H, W), np.float32)
        for (y, x), t in zip(tiles, out):
            np.maximum(p[y:y + TH, x:x + TW], t, out=p[y:y + TH, x:x + TW])
        return p[:h, :w]

    def boxes(self, img: np.ndarray, thresh=0.3, box_thresh=0.6, unclip=1.5) -> list:
        """Text boxes as EasyOCR's get_textbox gives them: int32 arrays of 8 values (4 corners, clockwise)."""
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        prob = self.prob(img)
        return [b.astype(np.int32).reshape(-1) for b, _ in db_boxes(prob, thresh, box_thresh, unclip)]

    def close(self) -> None:
        self._model.close()


# ------------------------------------------------------------------ PaddleOCR DBPostProcess (box_type quad)
def _mini_box(contour):
    rect = cv2.minAreaRect(contour)
    pts = sorted(list(cv2.boxPoints(rect)), key=lambda p: p[0])
    i1, i4 = (0, 1) if pts[1][1] > pts[0][1] else (1, 0)
    i2, i3 = (2, 3) if pts[3][1] > pts[2][1] else (3, 2)
    return np.array([pts[i1], pts[i2], pts[i3], pts[i4]], np.float32), min(rect[1])


def _box_score(prob, box):
    h, w = prob.shape
    xmin, xmax = np.clip(int(np.floor(box[:, 0].min())), 0, w - 1), np.clip(int(np.ceil(box[:, 0].max())), 0, w - 1)
    ymin, ymax = np.clip(int(np.floor(box[:, 1].min())), 0, h - 1), np.clip(int(np.ceil(box[:, 1].max())), 0, h - 1)
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8)
    cv2.fillPoly(mask, [(box - [xmin, ymin]).astype(np.int32)], 1)
    return cv2.mean(prob[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def _unclip(box, ratio):
    import pyclipper
    x, y = box[:, 0], box[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    length = float(np.sum(np.linalg.norm(box - np.roll(box, 1, axis=0), axis=1)))
    off = pyclipper.PyclipperOffset()
    off.AddPath([tuple(p) for p in box.tolist()], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    return off.Execute(area * ratio / max(length, 1e-6))


def db_boxes(prob, thresh, box_thresh, unclip, max_candidates=1000, min_size=3):
    h, w = prob.shape
    contours, _ = cv2.findContours(((prob > thresh) * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours[:max_candidates]:
        pts, sside = _mini_box(c)
        if sside < min_size:
            continue
        score = _box_score(prob, pts)
        if score < box_thresh:
            continue
        exp = _unclip(pts, unclip)
        if len(exp) != 1:
            continue
        box, sside = _mini_box(np.array(exp[0], np.float32).reshape(-1, 1, 2))
        if sside < min_size + 2:
            continue
        box[:, 0] = np.clip(np.round(box[:, 0]), 0, w)
        box[:, 1] = np.clip(np.round(box[:, 1]), 0, h)
        if np.linalg.norm(box[0] - box[1]) <= 3 or np.linalg.norm(box[0] - box[3]) <= 3:
            continue
        out.append((box, score))
    return out


# ------------------------------------------------------------------ EasyOCR hook
def install(reader) -> str:
    """Put the chip detector in front of this EasyOCR reader when config.OCR_DETECTOR == "hailo".
    Returns "hailo" when installed, "" otherwise (the reader is then untouched)."""
    if getattr(config, "OCR_DETECTOR", "") != "hailo":
        return ""
    try:
        det = HailoTextDetector(config.HAILO_OCR_DET_HEF, getattr(config, "OCR_HAILO_MODE", "tile"))
        det.boxes(np.zeros((64, 64, 3), np.uint8))       # one small pass: a broken chip or file fails here
    except Exception as exc:
        logger.warning("Hailo text detector not usable (%s); text mode keeps its CPU detector", exc)
        return ""

    from easyocr.utils import diff, group_text_box, reformat_input
    cpu_detect = reader.detect                            # the bound CPU method, kept for the fallback

    def detect(img, min_size=20, text_threshold=0.7, low_text=0.4, link_threshold=0.4, canvas_size=2560,
               mag_ratio=1., slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5, width_ths=0.5, add_margin=0.1,
               reformat=True, optimal_num_chars=None, **rest):
        if reader.__dict__.get("_hailo_det") is None:
            return cpu_detect(img, min_size, text_threshold, low_text, link_threshold, canvas_size, mag_ratio,
                              slope_ths, ycenter_ths, height_ths, width_ths, add_margin, reformat, optimal_num_chars,
                              **rest)
        if reformat:
            img, _ = reformat_input(img)
        try:
            polys = det.boxes(img)
        except Exception as exc:
            logger.warning("Hailo text detector failed (%s); text mode goes back to the CPU detector", exc)
            det.close()
            reader._hailo_det = None
            return cpu_detect(img, min_size, text_threshold, low_text, link_threshold, canvas_size, mag_ratio,
                              slope_ths, ycenter_ths, height_ths, width_ths, add_margin, False, optimal_num_chars,
                              **rest)
        hl, fl = group_text_box(polys, slope_ths, ycenter_ths, height_ths, width_ths, add_margin,
                                optimal_num_chars is None)
        if min_size:
            hl = [i for i in hl if max(i[1] - i[0], i[3] - i[2]) > min_size]
            fl = [i for i in fl if max(diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size]
        return [hl], [fl]

    reader._hailo_det = det
    reader.detect = detect
    logger.info("Text detector: Hailo AI HAT (%s, %s)", os.path.basename(config.HAILO_OCR_DET_HEF), det.mode)
    return "hailo"
