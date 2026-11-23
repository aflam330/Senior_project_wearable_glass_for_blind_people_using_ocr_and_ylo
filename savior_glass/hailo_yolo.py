"""YOLOv8s on the Raspberry Pi AI HAT (Hailo-8), for object mode when OBJECT_BACKEND=hailo.

The HEF is Hailo's own compile of the COCO YOLOv8s (hailo-models package, /usr/share/hailo-models/yolov8s_h8.hef): the
same network and 80 classes as models/yolov8s.pt, with the box filtering (NMS, IoU 0.70) done on the chip. The frame is
letterboxed to 640x640 with grey (114) padding, as Ultralytics does, so the boxes map back to the frame the same way.
On the Pi 5 one picture takes about 12 ms on the chip, against about 450 ms for yolov8s.pt on the CPU.

The chip pipeline stays open while the object stays alive; close() releases it.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger("smart_glass.hailo_yolo")

# COCO class order of yolov8s.pt and of the Hailo model zoo's yolov8s
COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")


class HailoYolo:
    """detect(frame_bgr, conf) -> [(name, conf, x1, y1, x2, y2)] in frame pixels, highest confidence first."""

    def __init__(self, hef_path: str) -> None:
        from hailo_device import open_model
        self._model = open_model(hef_path)            # the AI HAT is shared with the other chip models (hailo_device.py)
        info = self._model.inputs[0]
        self._out_name = self._model.outputs[0].name
        self._size = int(info.shape[0])               # NHWC: 640 x 640 x 3

    def detect(self, frame: np.ndarray, conf: float) -> list[tuple[str, float, int, int, int, int]]:
        h, w = frame.shape[:2]
        s = self._size
        r = min(s / h, s / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        px, py = (s - nw) // 2, (s - nh) // 2
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        out = self._model.infer(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None])[self._out_name][0]
        found = []
        for cls_id, boxes in enumerate(out):
            for y1, x1, y2, x2, score in np.asarray(boxes, dtype=np.float32).reshape(-1, 5):
                if score < conf:
                    continue
                found.append((COCO_NAMES[cls_id], float(score),
                              int(np.clip((x1 * s - px) / r, 0, w)), int(np.clip((y1 * s - py) / r, 0, h)),
                              int(np.clip((x2 * s - px) / r, 0, w)), int(np.clip((y2 * s - py) / r, 0, h))))
        found.sort(key=lambda d: d[1], reverse=True)
        return found

    def close(self) -> None:
        self._model.close()
