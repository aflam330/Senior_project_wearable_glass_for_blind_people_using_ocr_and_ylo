"""
On-screen preview for sighted testers.

Shows the live camera picture, the current mode, what the glass just said,
and object / note boxes. Keyboard keys mirror the GPIO buttons so a test can
run without the button board:

  M = Mode   A or Space = Action   R = Read   + / - = Volume   Q or Esc = Quit

Enabled when a desktop is available (DISPLAY or WAYLAND_DISPLAY is set);
SHOW_PREVIEW=0 turns it off, SHOW_PREVIEW=1 forces it on. The blind user's
glass runs headless, so nothing here is on the normal code path.
"""
import collections
import logging
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config

logger = logging.getLogger("smart_glass.preview")

WINDOW = "Smart Glass - test preview"
PANEL_W = 420
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
            cv2.resizeWindow(WINDOW, 640 + PANEL_W, 480)
            self._open = True

        app = self._app
        with app._mode_lock:
            mode_idx = app._current_mode
        frame = app._camera.get_frame()
        if frame is None:
            frame = np.zeros((480, 640, 3), np.uint8)
        else:
            frame = cv2.resize(frame, (640, 480))
        self._draw_boxes(frame, mode_idx)

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
        y += 28

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

        d.text((14, 452), "Keys: M mode  A action  R read  +/- vol  Q quit",
               font=self._f_small, fill=(130, 130, 140))

        canvas = np.hstack([frame, cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)])
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

    def _draw_boxes(self, frame, mode_idx) -> None:
        mode = self._app._modes[mode_idx]
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
