"""Live algorithm panel for the test preview: runs every algorithm on the current camera picture and reports it.

For sighted testers only (preview.py). A background thread takes the newest frame a few times per second and runs:
  1. note detector        YOLOv8s Taka detector: box, denomination, confidence
  2. safe counterfeit     PRMVT on four views cut from the note (500 / 1,000 Taka): p(genuine) against the threshold
  3. capture guide        brightness and sharpness of the note crop against the validation thresholds
  4. watermark            learned localizer finds the window, MobileNetV2 scores it (needs the note held to a light)
  5. objects              YOLOv8 COCO detector
  6. emotion              face detector + expression classifier
OCR is not run live (it takes seconds); the panel shows the last text captured with ACTION.
Each entry carries the time it took, so the panel shows that the model really ran on this frame.
It shares app._model_lock with the button actions, so two threads never use a model at the same time.
"""
from __future__ import annotations

import logging
import threading
import time

import config

logger = logging.getLogger("smart_glass.live_diag")


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


class LiveDiagnostics:
    def __init__(self, app) -> None:
        self.app = app
        self.enabled = False
        self.state: dict = {}
        self._wm = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="live-diag")
        self._thread.start()

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        if not self.enabled:
            self.state = {}
        return self.enabled

    def close(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self.enabled or self.app._inferring.is_set():
                time.sleep(0.15)
                continue
            frame = self.app._camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            try:
                with self.app._model_lock:
                    self.state = self._run(frame)
            except Exception as exc:  # noqa: BLE001
                self.state = {"error": f"{type(exc).__name__}: {exc}"}
                logger.warning("live diagnostics failed: %s", exc)
            time.sleep(0.2)

    # ------------------------------------------------------------------
    def _run(self, frame) -> dict:
        st = {"at": time.time(), "frame": [int(frame.shape[1]), int(frame.shape[0])]}
        cur = self.app._modes[config.MODE_CURRENCY]
        if cur._yolo is None or (cur.safe_policy and cur._auth is None):
            cur.activate()

        # 1. note detector
        crop, top = None, None
        if cur._yolo is not None:
            t0 = time.perf_counter()
            res = cur._yolo.predict(frame, conf=0.25, verbose=False, imgsz=640)[0]
            boxes = []
            if res.boxes is not None and len(res.boxes):
                for i in res.boxes.conf.argsort(descending=True).tolist()[:3]:
                    x1, y1, x2, y2 = (int(v) for v in res.boxes.xyxy[i].tolist())
                    boxes.append({"name": str(res.names[int(res.boxes.cls[i])]), "conf": float(res.boxes.conf[i]), "box": [x1, y1, x2, y2]})
            st["detector"] = {"ms": _ms(t0), "boxes": boxes, "announce_conf": float(getattr(config, "CURRENCY_ANNOUNCE_CONF", 0.0))}
            if boxes:
                top = boxes[0]
                x1, y1, x2, y2 = top["box"]
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.shape[0] < 24 or crop.shape[1] < 24:
                    crop = None
        else:
            st["detector"] = {"missing": True}

        big_note = top is not None and top["name"] in getattr(config, "JAAL_SAFE_DENOMINATIONS", ())
        # 2. safe counterfeit check (500 / 1,000 Taka only)
        if crop is not None and big_note and cur._auth is not None:
            t0 = time.perf_counter()
            p = cur._safe_check(crop)
            st["safe"] = {"ms": _ms(t0), "p_genuine": p, "tau": float(config.JAAL_SAFE_TAU),
                          "verdict": None if p is None else ("likely genuine" if p > config.JAAL_SAFE_TAU else "check by hand")}
        else:
            st["safe"] = {"skipped": "needs a 500 or 1,000 Taka note in view" if cur._auth is not None else "model not loaded"}

        # 3. capture guide quality gate
        if crop is not None:
            from modes.capture_guide import assess
            t0 = time.perf_counter()
            status, q = assess(crop, float(config.CAPTURE_GUIDE_MIN_MEAN), float(config.CAPTURE_GUIDE_MIN_LAPVAR))
            st["guide"] = {"ms": _ms(t0), "status": status, "mean": q.get("mean"), "lap_var": q.get("lap_var"),
                           "min_mean": float(config.CAPTURE_GUIDE_MIN_MEAN), "min_lapvar": float(config.CAPTURE_GUIDE_MIN_LAPVAR)}
        else:
            st["guide"] = {"skipped": "no note in view"}

        # 4. watermark localizer + classifier
        if crop is not None and big_note:
            try:
                if self._wm is None:
                    from modes.watermark_check import WatermarkChecker
                    self._wm = WatermarkChecker()
                t0 = time.perf_counter()
                info = self._wm.inspect(crop, top["name"])
                x1, y1 = max(0, top["box"][0]), max(0, top["box"][1])
                quad = [[int(x1 + px), int(y1 + py)] for px, py in info["corners"]] if info.get("corners") is not None else None
                st["watermark"] = {"ms": _ms(t0), "p_genuine": info.get("prob"), "path": info.get("path"), "quad": quad,
                                   "threshold": float(getattr(config, "WATERMARK_CLEAR_THRESHOLD", 0.5))}
            except Exception as exc:  # noqa: BLE001
                st["watermark"] = {"skipped": f"not available: {exc}"[:80]}
        else:
            st["watermark"] = {"skipped": "needs a 500 or 1,000 Taka note in view"}

        # 5. objects
        obj = self.app._modes[config.MODE_OBJECT]
        if obj._model is None:
            obj.activate()
        if obj._model is not None:
            t0 = time.perf_counter()
            r = obj._model.predict(frame, conf=float(config.OBJECT_CONFIDENCE), verbose=False)[0]
            items = []
            if r.boxes is not None:
                for i in range(min(len(r.boxes), 6)):
                    x1, y1, x2, y2 = (int(v) for v in r.boxes.xyxy[i].tolist())
                    items.append({"name": str(r.names[int(r.boxes.cls[i])]), "conf": float(r.boxes.conf[i]), "box": [x1, y1, x2, y2]})
            st["objects"] = {"ms": _ms(t0), "boxes": items}
        else:
            st["objects"] = {"missing": True}

        # 6. emotion
        emo = self.app._modes[getattr(config, "MODE_EMOTION", -1)] if hasattr(config, "MODE_EMOTION") else None
        if emo is not None:
            if emo._detector is None:
                emo.activate()
            if emo._detector is not None:
                t0 = time.perf_counter()
                e = emo._detector.predict(frame)
                st["emotion"] = {"ms": _ms(t0), "face": e.get("face"), "label": e.get("label"), "prob": e.get("prob"), "backend": e.get("backend")}
            else:
                st["emotion"] = {"missing": True}

        # 7. OCR: last captured text only
        ocr = self.app._modes[config.MODE_OCR]
        st["ocr"] = {"loaded": ocr._reader is not None, "pipeline": getattr(config, "OCR_PIPELINE", ""), "last_text": getattr(ocr, "_stored_text", None)}
        return st
