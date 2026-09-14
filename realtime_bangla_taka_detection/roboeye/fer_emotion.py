"""7-class facial emotion recognition with emotion-adaptive feedback.

Uses the public FER+ ONNX model when available (auto-downloaded), with an
OpenCV Haar cascade face detector. Falls back to a geometric smile/frown
heuristic if ONNX Runtime or the weights are missing.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .config import FER_LABELS, FER_ONNX, MODELS_DIR, ROOT

FERPLUS_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/"
    "emotion_ferplus/model/emotion-ferplus-8.onnx"
)

# OpenCV ships this cascade with the package.
_CASCADE_NAME = "haarcascade_frontalface_default.xml"


def _cascade_path() -> Path:
    data = Path(cv2.data.haarcascades) / _CASCADE_NAME
    return data


def ensure_fer_onnx() -> Path | None:
    if FER_ONNX.is_file() and FER_ONNX.stat().st_size > 10_000:
        return FER_ONNX
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(FERPLUS_URL, FER_ONNX)
        if FER_ONNX.stat().st_size > 10_000:
            return FER_ONNX
    except Exception:
        return None
    return None


class EmotionDetector:
    def __init__(self, download: bool = True):
        cascade = _cascade_path()
        self.face = cv2.CascadeClassifier(str(cascade)) if cascade.is_file() else None
        self.session = None
        self.input_name = None
        path = ensure_fer_onnx() if download else (FER_ONNX if FER_ONNX.is_file() else None)
        if path is not None:
            try:
                import onnxruntime as ort

                self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
            except Exception:
                self.session = None

    def _faces(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self.face is None:
            return []
        faces = self.face.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(48, 48))
        return [tuple(map(int, f)) for f in faces]

    def _ferplus(self, face_gray: np.ndarray) -> dict:
        face = cv2.resize(face_gray, (64, 64))
        x = face.astype(np.float32)[None, None, :, :]
        scores = self.session.run(None, {self.input_name: x})[0].reshape(-1)
        # FER+ uses a softmax over 8 emotions.
        exp = np.exp(scores - scores.max())
        prob = exp / exp.sum()
        idx = int(prob.argmax())
        label = FER_LABELS[idx] if idx < len(FER_LABELS) else "neutral"
        return {"label": label, "prob": float(prob[idx]), "probs": prob, "backend": "ferplus_onnx"}

    def _geometric(self, face_gray: np.ndarray) -> dict:
        """Crude 3-way fallback using mouth brightness vs upper-face brightness."""
        h, w = face_gray.shape
        upper = face_gray[: h // 2].mean()
        mouth = face_gray[int(h * 0.62) :, int(w * 0.25) : int(w * 0.75)].mean()
        delta = float(mouth - upper)
        if delta > 12:
            label, p = "happiness", min(0.55 + delta / 80.0, 0.8)
        elif delta < -8:
            label, p = "sadness", min(0.5 + abs(delta) / 80.0, 0.75)
        else:
            label, p = "neutral", 0.5
        return {"label": label, "prob": p, "probs": None, "backend": "geometric"}

    def predict(self, frame_bgr: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._faces(gray)
        if not faces:
            return {
                "label": "neutral",
                "prob": 0.0,
                "face": None,
                "backend": "no_face",
                "loaded": self.session is not None,
            }
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        roi = gray[y : y + h, x : x + w]
        result = self._ferplus(roi) if self.session is not None else self._geometric(roi)
        result["face"] = (x, y, w, h)
        result["loaded"] = self.session is not None
        return result


def tts_style_for_emotion(emotion: str) -> dict:
    """SAPI rate (-10..10) and a short prefix for emotion-adaptive speech."""
    emotion = (emotion or "neutral").lower()
    if emotion in {"sadness", "fear"}:
        return {"rate": -2, "prefix": "Take your time. "}
    if emotion == "anger":
        return {"rate": -1, "prefix": "It's alright. "}
    if emotion == "happiness":
        return {"rate": 1, "prefix": ""}
    if emotion == "surprise":
        return {"rate": 0, "prefix": ""}
    return {"rate": 0, "prefix": ""}
