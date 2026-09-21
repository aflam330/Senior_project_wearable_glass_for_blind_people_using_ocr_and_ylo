"""Assistive extras: emotion, pose, adaptive TTS. EasyOCR is unchanged."""

from pathlib import Path
import sys

_ROBO = Path(__file__).resolve().parents[2] / "realtime_bangla_taka_detection"
if _ROBO.is_dir() and str(_ROBO) not in sys.path:
    sys.path.insert(0, str(_ROBO))

from roboeye.fer_emotion import EmotionDetector, adapt_feedback, tts_style_for_emotion
from roboeye.pose import estimate_pose

__all__ = [
    "EmotionDetector",
    "adapt_feedback",
    "tts_style_for_emotion",
    "estimate_pose",
]
