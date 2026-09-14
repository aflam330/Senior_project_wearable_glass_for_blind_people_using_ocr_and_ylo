"""RoboEye live webcam: denomination + authenticity + FER + pose + TTS + haptics.

Controls
--------
  SPACE : speak denomination, authenticity, and pose prompt (emotion-adaptive)
  v     : listen for a voice command (“what is this note?”, “is it real?”)
  d     : VLM / structured description
  g     : toggle Grad-CAM overlay
  h     : replay last haptic pattern
  q/ESC : quit
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from roboeye.asr import VoiceCommands, parse_command
from roboeye.dual_head import DualHeadYOLO
from roboeye.fer_emotion import EmotionDetector
from roboeye.gradcam_live import LiveGradCAM
from roboeye.haptics import HapticEngine
from roboeye.pose import estimate_pose
from roboeye.speaker import Speaker
from roboeye.vlm import NoteDescriber

CAMERA_INDEX = 0
CONF_THRESHOLD = 0.35
SKIP_FRAMES = 3
SPEAK_CONF = 0.55


def brighten(frame, gain=3.0, gamma=0.5):
    f = frame.astype(np.float32) / 255.0
    f = np.clip(f * gain, 0, 1)
    f = np.power(f, gamma)
    return (f * 255).astype(np.uint8)


def spoken_name(class_name: str) -> str:
    return class_name.replace("_", " ")


def draw_hud(frame, detections, emotion, haptic_name, listening, caption, show_cam):
    h, w = frame.shape[:2]
    panel_w = 360
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (12, 16, 28), -1)
    frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)

    def put(y, text, scale=0.55, color=(0, 255, 180), thick=1):
        cv2.putText(frame, text, (w - panel_w + 12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    put(28, "RoboEye HUD", 0.7, (0, 255, 255), 2)
    if detections:
        d0 = detections[0]
        put(64, spoken_name(d0["name"]).upper(), 0.8, (0, 255, 0), 2)
        put(94, f"det {d0['conf']*100:.0f}%  auth {d0['genuine_prob']*100:.0f}%")
        color = (0, 255, 0) if d0["auth_label"] == "genuine" else ((0, 0, 255) if d0["auth_label"] == "counterfeit" else (0, 200, 255))
        put(122, d0["auth_label"].upper(), 0.7, color, 2)
        pose = d0.get("pose") or {}
        if pose.get("ok"):
            put(150, f"R {pose['roll']:+.0f}  P {pose['pitch']:+.0f}  Y {pose['yaw']:+.0f}")
            if pose.get("needs_straighten"):
                put(178, "STRAIGHTEN THE NOTE", 0.55, (0, 0, 255), 2)
            quad = pose.get("quad")
            if quad is not None:
                pts = quad.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
        x1, y1, x2, y2 = d0["xyxy"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    else:
        put(64, "NO NOTE", 0.7, (0, 180, 255), 2)

    emo = emotion.get("label", "neutral") if emotion else "neutral"
    put(220, f"emotion: {emo}", 0.55, (180, 220, 255))
    face = emotion.get("face") if emotion else None
    if face:
        x, y, fw, fh = face
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (255, 180, 0), 1)
    put(248, f"haptic: {haptic_name}", 0.5, (200, 200, 255))
    put(276, f"listen: {'ON' if listening else 'off'}  cam: {'ON' if show_cam else 'off'}", 0.45)
    if caption:
        y = 310
        for i in range(0, min(len(caption), 180), 36):
            put(y, caption[i : i + 36], 0.42, (220, 220, 220))
            y += 20
    cv2.putText(
        frame,
        "SPACE speak | v voice | d describe | g Grad-CAM | q quit",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return frame


def announce(speaker: Speaker, detections, emotion_label: str, describer: NoteDescriber, full: bool):
    if not detections:
        speaker.say("no note detected", emotion_label)
        return
    d0 = detections[0]
    if d0["conf"] < SPEAK_CONF:
        speaker.say("note visible but low confidence", emotion_label)
        return
    pose = d0.get("pose") or {}
    if full:
        text = describer.describe(
            d0.get("crop"),
            d0["name"],
            d0["conf"],
            d0["auth_label"],
            d0["genuine_prob"],
            pose,
        )
        speaker.say(text, emotion_label)
        return
    bits = [spoken_name(d0["name"])]
    if d0["auth_label"] == "counterfeit":
        bits.append("warning, possible counterfeit")
    elif d0["auth_label"] == "genuine":
        bits.append("appears genuine")
    if pose.get("needs_straighten"):
        bits.append("please straighten the note")
    speaker.say(". ".join(bits), emotion_label)


def main():
    dual = DualHeadYOLO(conf=CONF_THRESHOLD)
    speaker = Speaker()
    haptics = HapticEngine()
    emotion = EmotionDetector(download=True)
    describer = NoteDescriber()
    asr = VoiceCommands()
    cam = LiveGradCAM(yolo=dual.detector)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")
    for _ in range(20):
        cap.read()
    cv2.namedWindow("RoboEye", cv2.WINDOW_NORMAL)

    detections = []
    emo = {"label": "neutral", "face": None}
    caption = f"VLM={describer.backend}  ASR={'yes' if asr.available else 'keys'}  haptic={haptics.backend}"
    show_cam = False
    listening = False
    last_haptic_key = None
    last_haptic_t = 0.0
    frame_count = 0

    def on_asr(text, intent):
        nonlocal listening, caption
        listening = False
        caption = f"heard: {text or '(none)'}"
        if intent == "describe":
            announce(speaker, detections, emo.get("label", "neutral"), describer, True)
        elif intent == "authenticity":
            if detections:
                speaker.say(f"{detections[0]['auth_label']}", emo.get("label", "neutral"))
            else:
                speaker.say("no note detected", emo.get("label", "neutral"))
        elif intent == "help":
            speaker.say("Press space to speak. Say what is this note, or is it real.", emo.get("label", "neutral"))
        elif intent == "pose":
            pose = (detections[0].get("pose") if detections else None) or {}
            if pose.get("needs_straighten"):
                speaker.say("please straighten the note", emo.get("label", "neutral"))
            else:
                speaker.say("the note looks facing the camera", emo.get("label", "neutral"))
        elif intent == "emotion":
            speaker.say(f"you look {emo.get('label', 'neutral')}", "neutral")

    print("RoboEye live. SPACE=speak  v=voice  d=describe  g=Grad-CAM  q=quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % SKIP_FRAMES == 0:
            infer = brighten(frame) if frame.mean() < 80 else frame
            detections = dual.predict(infer)
            for d in detections:
                d["pose"] = estimate_pose(frame, d["xyxy"], d["name"])
            if frame_count % (SKIP_FRAMES * 4) == 0:
                emo = emotion.predict(frame)
            if detections:
                d0 = detections[0]
                key = (d0["name"], d0["auth_label"])
                now = time.time()
                if key != last_haptic_key or now - last_haptic_t > 2.5:
                    if d0.get("pose", {}).get("needs_straighten"):
                        haptics.play("straighten")
                    else:
                        haptics.for_auth(d0["auth_label"], d0["conf"])
                    last_haptic_key = key
                    last_haptic_t = now

        display = frame.copy()
        if show_cam and detections:
            display = cam.overlay(display, class_id=detections[0]["cls_id"])
        display = draw_hud(display, detections, emo, haptics.last_pattern, listening, caption, show_cam)
        cv2.imshow("RoboEye", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            announce(speaker, detections, emo.get("label", "neutral"), describer, False)
        elif key in (ord("d"), ord("D")):
            announce(speaker, detections, emo.get("label", "neutral"), describer, True)
        elif key in (ord("g"), ord("G")):
            show_cam = not show_cam
        elif key in (ord("h"), ord("H")):
            haptics.play(haptics.last_pattern if haptics.last_pattern != "idle" else "detect")
        elif key in (ord("v"), ord("V")):
            if asr.available:
                listening = True
                caption = "listening..."
                asr.listen_async(on_asr)
            else:
                caption = "ASR unavailable — use SPACE / d / typed keys"
                speaker.say("voice recognition is not available, use the keyboard", emo.get("label", "neutral"))

    speaker.stop()
    haptics.close()
    dual.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
