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


class FER7Net:
    pass  # defined lazily in _fer7_ctor to keep onnxruntime-only installs working


def _fer7_module():
    import torch.nn as nn

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.45),
                nn.Linear(256, 7),
            )

        def forward(self, x):
            return self.head(self.features(x))

    return _Net


class EmotionDetector:
    """Haar face crop + FER+ and/or the trained 7-class CNN."""

    _TABLE_TO_PLUS = {
        0: "happiness",
        1: "sadness",
        2: "anger",
        3: "surprise",
        4: "fear",
        5: "disgust",
        6: "neutral",
    }

    def __init__(self, download: bool = True):
        cascade = _cascade_path()
        self.face = cv2.CascadeClassifier(str(cascade)) if cascade.is_file() else None
        self.session = None
        self.input_name = None
        self.fer7 = None
        self.cal_coef = None
        self.cal_intercept = None
        self.rgb_net = None
        self.rgb_size = 160
        self.blend = 0.55
        path = ensure_fer_onnx() if download else (FER_ONNX if FER_ONNX.is_file() else None)
        if path is not None:
            try:
                import onnxruntime as ort

                self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
            except Exception:
                self.session = None
        self._load_trained()

    def _load_trained(self) -> None:
        try:
            import torch
            from pathlib import Path as _P

            glass_models = _P(__file__).resolve().parents[2] / "savior_glass" / "models"
            for root in (glass_models, MODELS_DIR):
                pt = root / "emotion_fer7.pt"
                cal = root / "emotion_ferplus_cal.npz"
                rgb_pt = root / "emotion_faces.pt"
                if rgb_pt.is_file() and self.rgb_net is None:
                    bundle = torch.load(rgb_pt, map_location="cpu", weights_only=False)
                    from torchvision import models as _tv

                    arch = str(bundle.get("arch") or "mobilenet_v3_small")
                    if arch in {"tf_efficientnet_b2_hs", "tf_efficientnet_b2"}:
                        import timm
                        net = timm.create_model(
                            str(bundle.get("timm_name") or "tf_efficientnet_b2"),
                            pretrained=False,
                            num_classes=7,
                            global_pool=str(bundle.get("global_pool") or "max"),
                        )
                    elif arch == "efficientnet_v2_s":
                        net = _tv.efficientnet_v2_s(weights=None)
                        in_f = net.classifier[-1].in_features
                        net.classifier[-1] = torch.nn.Linear(in_f, 7)
                    elif arch == "convnext_tiny":
                        net = _tv.convnext_tiny(weights=None)
                        in_f = net.classifier[-1].in_features
                        net.classifier[-1] = torch.nn.Linear(in_f, 7)
                    elif "large" in arch:
                        net = _tv.mobilenet_v3_large(weights=None)
                        in_f = net.classifier[-1].in_features
                        net.classifier[-1] = torch.nn.Linear(in_f, 7)
                    else:
                        net = _tv.mobilenet_v3_small(weights=None)
                        in_f = net.classifier[-1].in_features
                        net.classifier[-1] = torch.nn.Linear(in_f, 7)
                    net.load_state_dict(bundle["state_dict"])
                    net.eval()
                    self.rgb_net = net
                    self.rgb_size = int(bundle.get("img_size", 160))
                if pt.is_file() and self.fer7 is None:
                    bundle = torch.load(pt, map_location="cpu", weights_only=False)
                    net = _fer7_module()()
                    net.load_state_dict(bundle["state_dict"])
                    net.eval()
                    self.fer7 = net
                    self.blend = float(bundle.get("blend_cnn_weight", 0.55))
                if cal.is_file() and self.cal_coef is None:
                    data = np.load(cal)
                    self.cal_coef = data["coef"]
                    self.cal_intercept = data["intercept"]
        except Exception:
            self.fer7 = None

    def _faces(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self.face is None:
            return []
        faces = self.face.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(48, 48))
        return [tuple(map(int, f)) for f in faces]

    def _ferplus(self, face_gray: np.ndarray) -> dict:
        face = cv2.resize(face_gray, (64, 64))
        x = face.astype(np.float32)[None, None, :, :]
        scores = self.session.run(None, {self.input_name: x})[0].reshape(-1)
        exp = np.exp(scores - scores.max())
        prob = exp / exp.sum()
        idx = int(prob.argmax())
        label = FER_LABELS[idx] if idx < len(FER_LABELS) else "neutral"
        return {"label": label, "prob": float(prob[idx]), "probs": prob, "backend": "ferplus_onnx"}

    def _calibrated7(self, plus_probs: np.ndarray) -> np.ndarray | None:
        if self.cal_coef is None:
            return None
        logits = plus_probs @ self.cal_coef.T + self.cal_intercept
        logits = logits - logits.max()
        e = np.exp(logits)
        return e / e.sum()

    def _rgb7(self, face_bgr: np.ndarray) -> np.ndarray | None:
        if self.rgb_net is None:
            return None
        import torch
        import torch.nn.functional as F

        rgb = cv2.cvtColor(cv2.resize(face_bgr, (self.rgb_size, self.rgb_size)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = (x - mean) / std
        xb = x.unsqueeze(0)
        xf = torch.flip(xb, dims=[3])
        with torch.no_grad():
            p = 0.5 * (F.softmax(self.rgb_net(xb), dim=1) + F.softmax(self.rgb_net(xf), dim=1))
        return p.numpy().reshape(-1)

    def _cnn7(self, face_gray: np.ndarray) -> np.ndarray | None:
        if self.fer7 is None:
            return None
        import torch
        import torch.nn.functional as F

        img = cv2.resize(face_gray, (48, 48)).astype(np.float32) / 255.0
        x = torch.from_numpy(img[None, None, :, :])
        xf = torch.from_numpy(np.fliplr(img).copy()[None, None, :, :])
        with torch.no_grad():
            p = 0.5 * (F.softmax(self.fer7(x), dim=1) + F.softmax(self.fer7(xf), dim=1))
        return p.numpy().reshape(-1)

    def _geometric(self, face_gray: np.ndarray) -> dict:
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
                "loaded": self.session is not None or self.fer7 is not None or self.rgb_net is not None,
            }
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        roi = gray[y : y + h, x : x + w]
        roi_bgr = frame_bgr[y : y + h, x : x + w]
        plus = self._ferplus(roi) if self.session is not None else None
        p_rgb = self._rgb7(roi_bgr)
        p_cnn = self._cnn7(roi)
        p_cal = self._calibrated7(plus["probs"]) if plus is not None else None
        if p_rgb is not None and p_cal is not None:
            mix = 0.75 * p_rgb + 0.25 * p_cal
            idx = int(mix.argmax())
            result = {
                "label": self._TABLE_TO_PLUS[idx],
                "prob": float(mix[idx]),
                "probs": mix,
                "backend": "rgb_mobilenet_blend",
            }
        elif p_rgb is not None:
            idx = int(p_rgb.argmax())
            result = {
                "label": self._TABLE_TO_PLUS[idx],
                "prob": float(p_rgb[idx]),
                "probs": p_rgb,
                "backend": "rgb_mobilenet",
            }
        elif p_cnn is not None and p_cal is not None:
            mix = self.blend * p_cnn + (1.0 - self.blend) * p_cal
            idx = int(mix.argmax())
            result = {
                "label": self._TABLE_TO_PLUS[idx],
                "prob": float(mix[idx]),
                "probs": mix,
                "backend": "fer7_blend",
            }
        elif p_cal is not None:
            idx = int(p_cal.argmax())
            result = {
                "label": self._TABLE_TO_PLUS[idx],
                "prob": float(p_cal[idx]),
                "probs": p_cal,
                "backend": "ferplus_calibrated",
            }
        elif p_cnn is not None:
            idx = int(p_cnn.argmax())
            result = {
                "label": self._TABLE_TO_PLUS[idx],
                "prob": float(p_cnn[idx]),
                "probs": p_cnn,
                "backend": "fer7_cnn",
            }
        elif plus is not None:
            result = plus
        else:
            result = self._geometric(roi)
        result["face"] = (x, y, w, h)
        result["loaded"] = True
        return result


def tts_style_for_emotion(emotion: str) -> dict:
    """Rate offset, spoken prefix, and a simplicity flag for adaptive feedback.

    confused/stressed (fear, sadness, anger, disgust) → slower + reassuring + simpler
    neutral → normal
    positive (happiness, surprise) → normal / slightly encouraging
    """
    emotion = (emotion or "neutral").lower()
    if emotion in {"sadness", "fear", "disgust", "confused", "stressed"}:
        return {
            "rate": -3,
            "prefix_en": "It's okay. Take your time. ",
            "prefix_bn": "ঠিক আছে। ধীরে শুনুন। ",
            "simplify": True,
            "band": "stressed",
        }
    if emotion == "anger":
        return {
            "rate": -2,
            "prefix_en": "It's alright. ",
            "prefix_bn": "রাগ করবেন না। ",
            "simplify": True,
            "band": "stressed",
        }
    if emotion == "happiness":
        return {
            "rate": 1,
            "prefix_en": "Good. ",
            "prefix_bn": "ভালো। ",
            "simplify": False,
            "band": "positive",
        }
    if emotion == "surprise":
        return {
            "rate": 0,
            "prefix_en": "",
            "prefix_bn": "",
            "simplify": False,
            "band": "positive",
        }
    return {
        "rate": 0,
        "prefix_en": "",
        "prefix_bn": "",
        "simplify": False,
        "band": "neutral",
    }


def _first_clause(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return stripped
    for sep in ("।", "!", "?", ".", ";", "\n"):
        if sep in stripped:
            head = stripped.split(sep, 1)[0].strip()
            if len(head) >= 8:
                return head + ("।" if sep == "।" else "")
    return stripped


def adapt_feedback(text: str, emotion: str, enabled: bool = True, bangla: bool = True) -> dict:
    """Rewrite an assistive utterance for the listener's current emotion.

    When adaptive feedback is OFF the original text is returned unchanged.
    """
    original = (text or "").strip()
    style = tts_style_for_emotion(emotion)
    if not enabled or not original:
        return {
            "text": original,
            "emotion": emotion or "neutral",
            "band": "off" if not enabled else style["band"],
            "adaptive": False,
            "rate": 0,
        }
    spoken = _first_clause(original) if style["simplify"] else original
    prefix = style["prefix_bn"] if bangla else style["prefix_en"]
    return {
        "text": prefix + spoken,
        "emotion": emotion or "neutral",
        "band": style["band"],
        "adaptive": True,
        "rate": style["rate"],
    }
