"""
On-screen preview for sighted testers.

Shows the live camera picture, the current mode, what the glass just said,
and object / note boxes. Keyboard keys mirror the GPIO buttons so a test can
run without the button board:

  M = Mode   A or Space = Action   R = Read   + / - = Volume   Q or Esc = Quit
  L = live algorithm panel (live_diag.py): every model runs on the current picture and reports its output and time
  Voice study (STUDY_VOICE=1, voice_study.py): the name is asked at start; D = done, asks the yes / no questions;
  N = next participant; Y / N while a question is listening = answer by key

Enabled when a desktop is available (DISPLAY or WAYLAND_DISPLAY is set);
SHOW_PREVIEW=0 turns it off, SHOW_PREVIEW=1 forces it on. The blind user's
glass runs headless, so nothing here is on the normal code path.
"""
import collections
import logging
import os
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config

logger = logging.getLogger("smart_glass.preview")

WINDOW = "Smart Glass - test preview"
PANEL_W = 420
DIAG_W = 470
MODE_EN = {config.MODE_OCR: "TEXT (OCR)", config.MODE_OBJECT: "OBJECTS",
           config.MODE_CURRENCY: "CURRENCY", config.MODE_CLAUDE: "ONLINE",
           getattr(config, "MODE_EMOTION", 4): "EMOTION"}
_FONT_PATH = os.path.join(config.BASE_DIR, "ocr_bench", "fonts", "NotoSansBengali.ttf")


def enabled() -> bool:
    flag = os.environ.get("SHOW_PREVIEW", "auto").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    if flag in ("1", "true", "yes", "on"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _font(size):
    try:
        # RAQM layout shapes Bangla conjuncts correctly when libraqm is present
        return ImageFont.truetype(_FONT_PATH, size, layout_engine=ImageFont.Layout.RAQM)
    except Exception:  # noqa: BLE001
        try:
            return ImageFont.truetype(_FONT_PATH, size)
        except Exception:  # noqa: BLE001
            return ImageFont.load_default()


class Preview:
    """Must be created before app.start() so the button wrappers get registered."""

    def __init__(self, app) -> None:
        self._app = app
        self._said = collections.deque(maxlen=8)        # (time, text)
        self._presses = collections.deque(maxlen=5)     # (time, name, source)
        self._f_big, self._f, self._f_small = _font(26), _font(19), _font(15)
        self._open = False
        from live_diag import LiveDiagnostics
        self._diag = LiveDiagnostics(app)
        if os.environ.get("SHOW_ALGORITHMS", "0").strip() in ("1", "true", "on", "yes"):
            self._diag.enabled = True
        self._study = None
        if os.environ.get("STUDY_VOICE", "0").strip() in ("1", "true", "on", "yes"):
            from voice_study import VoiceStudy
            self._study = VoiceStudy(app)
            self._study_started = False

        # Record everything the glass says
        speak = app._tts.speak

        def _speak(text, *a, **kw):
            self._said.append((time.time(), str(text)))
            return speak(text, *a, **kw)
        app._tts.speak = _speak

        # Record GPIO presses (and keyboard presses, which call the same methods)
        for attr, name in (("_on_mode_press", "MODE"), ("_on_action_press", "ACTION"),
                           ("_on_read_press", "READ"), ("_on_vol_up", "VOL+"), ("_on_vol_down", "VOL-")):
            fn = getattr(app, attr)

            def _wrapped(fn=fn, name=name, source="button"):
                self._presses.append((time.time(), name, source))
                logger.info("%s pressed (%s)", name, source)
                return fn()
            setattr(app, attr, _wrapped)

    def _press(self, attr) -> None:
        getattr(self._app, attr)(source="keyboard")

    # ------------------------------------------------------------------

    def tick(self) -> bool:
        """Draw one frame and handle keys. Returns False when the user quits."""
        if not self._open:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, max(640, config.CAMERA_WIDTH) + PANEL_W, min(max(480, config.CAMERA_HEIGHT), 900))
            self._open = True

        app = self._app
        with app._mode_lock:
            mode_idx = app._current_mode
        if self._study is not None and not self._study_started and app._ready[mode_idx].is_set():
            self._study_started = True          # the app is up: ask the first participant's name
            self._study.new_participant()
        frame = app._camera.get_frame()
        if frame is None:
            frame = np.zeros((480, 640, 3), np.uint8)
        else:
            frame = frame.copy()
        if self._diag.enabled:
            self._draw_diag(frame)
        else:
            self._start_live(frame, mode_idx)
            self._draw_boxes(frame, mode_idx)   # boxes are in camera coordinates, so draw before any scaling
        # show the picture at its real size (so a sharper capture looks sharper), between 480 and 900 px high
        h, w = frame.shape[:2]
        dh = min(max(h, 480), 900)
        if dh != h:
            frame = cv2.resize(frame, (int(w * dh / h), dh), interpolation=cv2.INTER_AREA if dh < h else cv2.INTER_LINEAR)

        panel = Image.new("RGB", (PANEL_W, 480), (28, 30, 36))
        d = ImageDraw.Draw(panel)
        y = 10
        d.text((14, y), f"MODE: {MODE_EN.get(mode_idx, mode_idx)}", font=self._f_big, fill=(120, 220, 255))
        y += 38
        if app._inferring.is_set():
            status, col = "WORKING... (please wait)", (255, 200, 60)
        elif not app._ready[mode_idx].is_set():
            status, col = "Loading models...", (255, 200, 60)
        else:
            status, col = "Ready - press ACTION", (110, 230, 120)
        d.text((14, y), status, font=self._f, fill=col)
        y += 30
        d.text((14, y), f"Volume: {app._volume}%", font=self._f_small, fill=(200, 200, 200))
        y += 24
        if self._study is not None:
            listening = "LISTENING" in self._study.status
            for line in self._wrap(self._study.status, self._f_small, PANEL_W - 30)[:2]:
                d.text((14, y), line, font=self._f_small, fill=(255, 90, 90) if listening else (255, 200, 60))
                y += 19
            if listening:                      # live microphone level: the bar must move when the participant speaks
                d.rectangle((14, y + 2, 14 + PANEL_W - 40, y + 14), outline=(120, 120, 130))
                d.rectangle((14, y + 2, 14 + int((PANEL_W - 40) * self._study.level), y + 14), fill=(90, 220, 110))
                y += 20
            if self._study.last_heard:
                for line in self._wrap("heard: " + self._study.last_heard, self._f_small, PANEL_W - 30)[:2]:
                    d.text((14, y), line, font=self._f_small, fill=(200, 200, 200))
                    y += 19
        y += 4

        d.text((14, y), "Last presses:", font=self._f_small, fill=(160, 160, 170))
        y += 22
        now = time.time()
        for t, name, src in list(self._presses)[-3:][::-1]:
            d.text((24, y), f"{name}  ({src}, {now - t:.0f}s ago)", font=self._f_small, fill=(230, 230, 230))
            y += 20
        y = max(y, 196) + 6

        d.text((14, y), "Glass said:", font=self._f_small, fill=(160, 160, 170))
        y += 24
        for i, (t, text) in enumerate(list(self._said)[::-1]):
            col = (255, 255, 255) if i == 0 else (150, 150, 150)
            for line in self._wrap(text, self._f if i == 0 else self._f_small, PANEL_W - 30):
                if y > 440:
                    break
                d.text((24, y), line, font=self._f if i == 0 else self._f_small, fill=col)
                y += 28 if i == 0 else 21
            y += 4

        d.text((14, 452), "M mode  A action  R read  L algorithms  Q quit" + ("  D done  N next" if self._study is not None else ""),
               font=self._f_small, fill=(130, 130, 140))

        side = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
        if side.shape[0] < frame.shape[0]:   # pad the 480-px panel down to the picture height
            side = cv2.copyMakeBorder(side, 0, frame.shape[0] - side.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(36, 30, 28))
        cols = [frame, side]
        if self._diag.enabled:
            cols.append(self._diag_panel(frame.shape[0]))
        canvas = np.hstack(cols)
        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("m"), ord("M")):
            self._press("_on_mode_press")
        elif key in (ord("a"), ord("A"), ord(" ")):
            self._press("_on_action_press")
        elif key in (ord("r"), ord("R")):
            self._press("_on_read_press")
        elif key in (ord("+"), ord("=")):
            self._press("_on_vol_up")
        elif key in (ord("-"), ord("_")):
            self._press("_on_vol_down")
        elif self._study is not None and self._study._busy.locked() and key in (ord("y"), ord("Y"), ord("n"), ord("N")):
            self._study.key_answer = "yes" if key in (ord("y"), ord("Y")) else "no"
        elif self._study is not None and key in (ord("d"), ord("D")):
            self._study.finish()                # done using the glass: ask the questions
        elif self._study is not None and key in (ord("n"), ord("N")):
            self._study.new_participant()
        elif key in (ord("l"), ord("L")) or (self._study is None and key in (ord("d"), ord("D"))):
            on = self._diag.toggle()
            cv2.resizeWindow(WINDOW, max(640, config.CAMERA_WIDTH) + PANEL_W + (DIAG_W if on else 0),
                             min(max(480, config.CAMERA_HEIGHT), 900))
        elif key in (ord("q"), ord("Q"), 27):
            return False
        try:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return False   # window closed with the X button
        except cv2.error:
            return False
        return True

    def close(self) -> None:
        if self._open:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    # ------------------------------------------------------------------

    # ---- live boxes: the current mode's own detector on the live picture (config.PREVIEW_LIVE_BOXES) ----
    def _start_live(self, frame, mode_idx) -> None:
        if not getattr(config, "PREVIEW_LIVE_BOXES", False):
            return
        if mode_idx not in (config.MODE_CURRENCY, getattr(config, "MODE_EMOTION", -1)):
            return
        now = time.time()
        live = getattr(self, "_live", None) or {}
        if live.get("busy") or now - live.get("t", 0) < getattr(config, "PREVIEW_LIVE_INTERVAL_S", 1.0):
            return
        app = self._app
        if not app._ready[mode_idx].is_set() or app._inferring.is_set():
            return   # a model is still loading, or a button capture is running: leave the model to it
        self._live = {**live, "busy": True, "t": now}
        threading.Thread(target=self._live_run, args=(frame.copy(), mode_idx), daemon=True, name="preview-live").start()

    def _live_run(self, frame, mode_idx) -> None:
        app, boxes = self._app, []
        try:
            if not app._model_lock.acquire(blocking=False):
                return
            try:
                mode = app._modes[mode_idx]
                if mode_idx == config.MODE_CURRENCY and getattr(mode, "_yolo", None) is not None:
                    r = mode._yolo.predict(frame, conf=0.25, verbose=False, imgsz=640)[0]
                    if r.boxes is not None:
                        for i in range(len(r.boxes)):
                            x1, y1, x2, y2 = (int(v) for v in r.boxes.xyxy[i].tolist())
                            boxes.append((x1, y1, x2, y2, str(r.names[int(r.boxes.cls[i].item())]), float(r.boxes.conf[i].item())))
                elif mode_idx == getattr(config, "MODE_EMOTION", -1) and getattr(mode, "_detector", None) is not None:
                    res = mode._detector.predict(frame)
                    if res.get("face") is not None:
                        x, y, w, h = (int(v) for v in res["face"])
                        boxes.append((x, y, x + w, y + h, str(res.get("label", "")), float(res.get("prob") or 0)))
            finally:
                app._model_lock.release()
            self._live = {"busy": False, "t": time.time(), "mode": mode_idx, "boxes": boxes}
        except Exception as exc:  # noqa: BLE001
            logger.debug("Live boxes failed: %s", exc)
        finally:
            if getattr(self, "_live", {}).get("busy"):
                self._live = {**self._live, "busy": False}

    @staticmethod
    def _label(frame, text, x, y, colour) -> None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(th + 4, y)
        cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y + 2), colour, -1)
        cv2.putText(frame, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_boxes(self, frame, mode_idx) -> None:
        mode = self._app._modes[mode_idx]
        now = time.time()
        live = getattr(self, "_live", None) or {}
        fresh = live.get("mode") == mode_idx and now - live.get("t", 0) < 3 * getattr(config, "PREVIEW_LIVE_INTERVAL_S", 1.0)
        keep = getattr(config, "PREVIEW_CAPTURE_BOX_S", 15.0)
        if mode_idx == config.MODE_OCR:
            if now - getattr(mode, "last_boxes_t", 0) < keep:
                for x1, y1, x2, y2, conf, main in getattr(mode, "last_boxes", None) or []:
                    colour = (0, 220, 0) if main else (150, 150, 150)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2 if main else 1)
                    self._label(frame, ("text " if main else "") + f"{conf:.2f}", x1, y1 - 2, colour)
            return
        if mode_idx == config.MODE_CLAUDE:
            if now - getattr(mode, "last_sent_t", 0) < keep:
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (2, 2), (w - 3, h - 3), (255, 200, 0), 3)
                self._label(frame, "whole picture sent to the online model", 8, 22, (255, 200, 0))
            return
        if mode_idx == config.MODE_CURRENCY and fresh:
            for x1, y1, x2, y2, name, conf in live.get("boxes") or []:
                ok = conf >= getattr(config, "CURRENCY_ANNOUNCE_CONF", 0.0)
                colour = (0, 200, 255) if ok else (120, 120, 120)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                self._label(frame, f"{name} {conf:.2f}" + ("" if ok else " (too low)"), x1, y1 - 2, colour)
            return
        if mode_idx == getattr(config, "MODE_EMOTION", -1) and fresh:
            for x1, y1, x2, y2, label, prob in live.get("boxes") or []:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 255), 2)
                self._label(frame, f"{label} {prob:.2f}", x1, y1 - 2, (255, 120, 255))
            return
        if mode_idx == config.MODE_OBJECT:
            for x1, y1, x2, y2, name, conf in getattr(mode, "last_boxes", None) or []:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1 + 3, max(14, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif mode_idx == config.MODE_CURRENCY:
            for h in getattr(mode, "last_hits", None) or []:
                x, y, w, hh = (int(v) for v in h["bbox"])
                cv2.rectangle(frame, (x, y), (x + w, y + hh), (0, 200, 255), 2)
                label = f"{h.get('name', '')} {h.get('conf') or 0:.2f} {h.get('auth', '')}"
                cv2.putText(frame, label, (x + 3, max(14, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

        elif mode_idx == getattr(config, "MODE_EMOTION", -1):
            last = getattr(mode, "last", None) or {}
            if last.get("face"):
                x, y, w, hh = (int(v) for v in last["face"])
                cv2.rectangle(frame, (x, y), (x + w, y + hh), (255, 120, 255), 2)
                cv2.putText(frame, f"{last.get('label', '')} {last.get('prob') or 0:.2f}", (x + 3, max(14, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 255), 2)

    # ---- live algorithm panel (live_diag.py) ----
    def _draw_diag(self, frame) -> None:
        st = self._diag.state or {}
        for b in (st.get("objects") or {}).get("boxes") or []:
            x1, y1, x2, y2 = b["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 1)
            cv2.putText(frame, f"{b['name']} {b['conf']:.2f}", (x1 + 3, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
        for i, b in enumerate((st.get("detector") or {}).get("boxes") or []):
            x1, y1, x2, y2 = b["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 170, 255), 3 if i == 0 else 1)
            cv2.putText(frame, f"NOTE {b['name']} {b['conf']:.2f}", (x1 + 3, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 170, 255), 2)
        quad = (st.get("watermark") or {}).get("quad")
        if quad:
            cv2.polylines(frame, [np.int32(quad)], True, (255, 255, 0), 2)
            cv2.putText(frame, "watermark window", (quad[0][0], max(14, quad[0][1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        face = (st.get("emotion") or {}).get("face")
        if face:
            x, y, w, h = (int(v) for v in face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 120, 255), 2)
            cv2.putText(frame, f"{st['emotion'].get('label', '')} {st['emotion'].get('prob') or 0:.2f}", (x + 3, max(14, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 255), 2)

    def _diag_panel(self, height: int):
        st = self._diag.state or {}
        img = Image.new("RGB", (DIAG_W, height), (20, 24, 30))
        d = ImageDraw.Draw(img)
        OK, WARN, DIM, HEAD = (110, 230, 120), (255, 200, 60), (150, 150, 160), (120, 220, 255)
        y = 10
        d.text((12, y), "ALGORITHMS - live on this picture", font=self._f, fill=HEAD)
        y += 28
        age = time.time() - st.get("at", 0) if st.get("at") else None
        size = "x".join(str(v) for v in st.get("frame", [])) or "-"
        d.text((12, y), f"camera {size}   updated {age:.1f}s ago" if age is not None else "starting...", font=self._f_small, fill=DIM)
        y += 24

        def block(title, lines):
            nonlocal y
            if y > height - 40:
                return
            d.text((12, y), title, font=self._f_small, fill=HEAD)
            y += 20
            for text, col in lines:
                for line in self._wrap(text, self._f_small, DIAG_W - 34):
                    if y > height - 22:
                        return
                    d.text((24, y), line, font=self._f_small, fill=col)
                    y += 19
            y += 7

        if st.get("error"):
            block("ERROR", [(st["error"], WARN)])
        det = st.get("detector") or {}
        if det.get("missing"):
            block("1. Note detector (YOLOv8s)", [("model not loaded", WARN)])
        elif det:
            bx = det.get("boxes") or []
            lines = [(f"ran in {det['ms']} ms", DIM)]
            lines += [(f"{b['name']}  confidence {b['conf']:.2f}" + ("" if b["conf"] >= det["announce_conf"] else f"  (< {det['announce_conf']:.2f}: not announced)"),
                       OK if b["conf"] >= det["announce_conf"] else WARN) for b in bx] or [("no note in the picture", DIM)]
            block("1. Note detector (YOLOv8s)", lines)
        sf = st.get("safe") or {}
        if sf.get("skipped"):
            block("2. Safe counterfeit check (PRMVT)", [(sf["skipped"], DIM)])
        elif sf:
            p = sf.get("p_genuine")
            block("2. Safe counterfeit check (PRMVT)", [
                (f"ran in {sf['ms']} ms", DIM),
                (f"p(genuine) = {p:.4f}   threshold {sf['tau']:.4f}" if p is not None else "no score", (230, 230, 230)),
                (f"says: {sf.get('verdict')}", OK if sf.get("verdict") == "likely genuine" else WARN),
                ("never says 'counterfeit'", DIM)])
        gd = st.get("guide") or {}
        if gd.get("skipped"):
            block("3. Capture guide (brightness, sharpness)", [(gd["skipped"], DIM)])
        elif gd:
            block("3. Capture guide (brightness, sharpness)", [
                (f"brightness {gd['mean']:.0f}  (needs {gd['min_mean']:.0f})", OK if (gd['mean'] or 0) >= gd['min_mean'] else WARN),
                (f"sharpness {gd['lap_var']:.0f}  (needs {gd['min_lapvar']:.0f})", OK if (gd['lap_var'] or 0) >= gd['min_lapvar'] else WARN),
                (f"frame: {gd['status']}", OK if gd["status"] == "ok" else WARN)])
        wm = st.get("watermark") or {}
        if wm.get("skipped"):
            block("4. Watermark (localizer + MobileNetV2)", [(wm["skipped"], DIM)])
        elif wm:
            p = wm.get("p_genuine")
            block("4. Watermark (localizer + MobileNetV2)", [
                (f"ran in {wm['ms']} ms   window: {wm.get('path') or 'not found'}", DIM),
                (f"p(watermark clear) = {p:.3f}   threshold {wm['threshold']:.2f}" if p is not None else "window not found: hold the note straight", (230, 230, 230)),
                (("says: watermark clear" if p > wm["threshold"] else "says: not clear, check by hand") if p is not None else "", OK if p is not None and p > wm["threshold"] else WARN),
                ("only meaningful with the note held against a light", DIM)])
        ob = st.get("objects") or {}
        if ob.get("missing"):
            block("5. Objects (YOLOv8, COCO)", [("model not loaded", WARN)])
        elif ob:
            names = ", ".join(f"{b['name']} {b['conf']:.2f}" for b in ob.get("boxes") or []) or "nothing above the confidence threshold"
            block("5. Objects (YOLOv8, COCO)", [(f"ran in {ob['ms']} ms", DIM), (names, OK if ob.get("boxes") else DIM)])
        em = st.get("emotion") or {}
        if em.get("missing"):
            block("6. Emotion", [("model not loaded", WARN)])
        elif em:
            block("6. Emotion (face + expression)", [
                (f"ran in {em['ms']} ms   {em.get('backend', '')}", DIM),
                (f"{em.get('label')}  {em.get('prob') or 0:.2f}" if em.get("face") else "no face in the picture", OK if em.get("face") else DIM)])
        oc = st.get("ocr") or {}
        if oc:
            block("7. Text reading (EasyOCR, pipeline " + str(oc.get("pipeline")) + ")", [
                ("loaded; runs on ACTION in TEXT mode (takes seconds)" if oc.get("loaded") else "not loaded yet", DIM),
                (f"last text: {oc['last_text']}" if oc.get("last_text") else "no text captured yet", (230, 230, 230) if oc.get("last_text") else DIM)])
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _wrap(text, font, width):
        lines, cur = [], ""
        for word in text.split():
            trial = f"{cur} {word}".strip()
            if font.getlength(trial) <= width or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines[:6]
