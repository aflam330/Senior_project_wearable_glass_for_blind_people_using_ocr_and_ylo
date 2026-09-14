"""Smoke-test every RoboEye module on a real JaalTaka crop + YOLO weights."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from roboeye.asr import parse_command
from roboeye.authenticity import AuthenticityClassifier, list_jaaltaka_notes
from roboeye.clip_zero_shot import PrototypeAuthenticator
from roboeye.config import YOLO_WEIGHTS
from roboeye.fer_emotion import EmotionDetector, tts_style_for_emotion
from roboeye.haptics import HapticEngine
from roboeye.pose import estimate_pose
from roboeye.vlm import NoteDescriber


def main():
    notes = list_jaaltaka_notes()
    print(f"JaalTaka notes: {len(notes)}")
    assert notes, "JaalTaka missing"
    note_dir, label = notes[0]
    img_path = next(p for p in note_dir.iterdir() if p.suffix.lower() == ".jpg")
    crop = cv2.imread(str(img_path))
    assert crop is not None
    print(f"sample {img_path} label={label} shape={crop.shape}")

    auth = AuthenticityClassifier()
    print("CNN+ViT", auth.predict([crop]))

    proto = PrototypeAuthenticator()
    if not proto.loaded:
        print("building prototypes (small)...")
        proto.build(max_notes_per_class=12, views_per_note=1)
    print("prototypes", proto.predict(crop), "backend", proto.backend)

    if YOLO_WEIGHTS.is_file():
        from ultralytics import YOLO

        yolo = YOLO(str(YOLO_WEIGHTS))
        r = yolo.predict(crop, conf=0.15, verbose=False)[0]
        print("YOLO names", yolo.names)
        print("YOLO boxes", len(r.boxes) if r.boxes is not None else 0)
        h, w = crop.shape[:2]
        xyxy = (0, 0, w, h)
        if r.boxes is not None and len(r.boxes):
            xyxy = tuple(int(v) for v in r.boxes.xyxy[0].cpu().numpy())
        pose = estimate_pose(crop, xyxy, "100_taka")
        print("pose", {k: pose[k] for k in pose if k not in {"quad", "tvec", "rvec"}})
    else:
        pose = {"ok": False, "needs_straighten": False}

    emo = EmotionDetector(download=False)
    print("FER", {k: emo.predict(crop)[k] for k in ("label", "backend", "loaded")})
    print("tts style", tts_style_for_emotion("sadness"))

    h = HapticEngine()
    h.play("detect", also_beep=False)
    print("haptic backend", h.backend)
    h.close()

    desc = NoteDescriber()
    print("VLM", desc.backend, desc.describe(crop, "100_taka", 0.9, "genuine", 0.8, pose))

    assert parse_command("What is this note?") == "describe"
    assert parse_command("is it real") == "authenticity"
    print("ASR parser ok")
    print("ALL MODULE SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
