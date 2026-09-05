"""Bangla currency detection — live bounding boxes on every frame,
with press-to-speak text-to-speech announcements of the detected note.

The detection model and inference are UNCHANGED; text-to-speech is a pure
output add-on and does not affect accuracy in any way.

Controls
--------
  SPACE : speak the name of the note currently detected on screen
  q/ESC : quit

Speech uses the Windows SAPI voice (offline) on a background thread, so it
never blocks the video loop and works reliably for every note, every press.
"""

import queue
import threading

import cv2
import numpy as np
import pythoncom
import win32com.client
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
CAMERA_INDEX = 0
CONF_THRESHOLD = 0.35
SKIP_FRAMES = 2      # run inference every N frames to keep preview smooth
SPEAK_CONF = 0.55    # only speak notes we're fairly confident about


# ──────────────────────────────────────────────────────────────────────────────
# Text-to-speech: dedicated background thread driving the Windows SAPI voice.
# Using SAPI directly (via win32com) speaks reliably every time, unlike the
# pyttsx3 run-loop which goes silent after the first utterance.
# ──────────────────────────────────────────────────────────────────────────────
class Speaker:
    def __init__(self, rate=1):
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._rate = rate
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        pythoncom.CoInitialize()
        voice = win32com.client.Dispatch("SAPI.SpVoice")
        voice.Rate = self._rate
        try:
            while not self._stop.is_set():
                try:
                    phrase = self._q.get(timeout=0.1)
                except queue.Empty:
                    continue
                if phrase is None:
                    break
                voice.Speak(phrase)   # synchronous, repeatable, reliable
        finally:
            pythoncom.CoUninitialize()

    def say(self, phrase):
        self._q.put(phrase)

    def stop(self):
        self._stop.set()
        self._q.put(None)


def spoken_name(class_name):
    """'10_taka' -> '10 taka'."""
    return class_name.replace("_", " ")


def brighten(frame, gain=3.0, gamma=0.5):
    f = frame.astype(np.float32) / 255.0
    f = np.clip(f * gain, 0, 1)
    f = np.power(f, gamma)
    return (f * 255).astype(np.uint8)


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    for _ in range(30):
        cap.read()

    cv2.namedWindow("Bangla Currency Detection", cv2.WINDOW_NORMAL)

    speaker = Speaker()
    current_notes = []   # class names detected in the most recent inference

    frame_count = 0
    last_annotated = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % SKIP_FRAMES == 0:
            inference_frame = brighten(frame) if frame.mean() < 80 else frame
            results = model.predict(inference_frame, conf=CONF_THRESHOLD, verbose=False)
            last_annotated = results[0].plot(img=frame.copy())

            # record which notes are currently visible (does not touch the model)
            notes = []
            for b in results[0].boxes:
                if float(b.conf.item()) >= SPEAK_CONF:
                    name = model.names[int(b.cls.item())]
                    if name not in notes:
                        notes.append(name)
            current_notes = notes

        display = last_annotated if last_annotated is not None else frame
        cv2.putText(
            display, "Press SPACE to speak note  |  'q' to quit",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
        cv2.imshow("Bangla Currency Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            # speak whatever is on screen right now
            if current_notes:
                for name in current_notes:
                    speaker.say(spoken_name(name))
            else:
                speaker.say("no note detected")

    speaker.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
