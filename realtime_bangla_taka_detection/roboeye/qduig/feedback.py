"""Novelty 9 — State-Conditioned Assistive Feedback Policy (SCAFP).

Deterministic map from (emotion, confidences, task state, failures, views,
pose quality) to TTS / haptic / request-next-view actions.

Three logged conditions:
  fixed, confidence-aware, confidence+emotion-aware.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Condition = Literal["fixed", "confidence_aware", "confidence_emotion_aware"]


@dataclass
class FeedbackState:
    emotion: str
    emotion_conf: float
    detection_conf: float
    auth_conf: float
    auth_uncertainty: float
    task_state: str
    recent_failures: int
    num_views: int
    pose_quality: float


@dataclass
class FeedbackAction:
    tts_speed: float
    verbosity: str
    repetition: int
    confirmation: bool
    haptic_intensity: float
    haptic_pattern: str
    request_next_view: bool
    condition: str
    reason: str


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def scafp(state: FeedbackState, condition: Condition) -> FeedbackAction:
    """Fully deterministic. Same inputs always produce the same action."""
    if condition == "fixed":
        return FeedbackAction(
            tts_speed=1.0,
            verbosity="normal",
            repetition=1,
            confirmation=False,
            haptic_intensity=0.5,
            haptic_pattern="detect",
            request_next_view=state.num_views < 6 and state.task_state == "authenticating",
            condition=condition,
            reason="fixed_schedule",
        )

    low_auth = state.auth_conf < 0.75 or state.auth_uncertainty > 0.45
    fail_streak = state.recent_failures >= 2
    request = bool(
        state.task_state == "authenticating"
        and state.num_views < 6
        and (low_auth or state.detection_conf < 0.50 or state.pose_quality < 0.40)
    )

    if condition == "confidence_aware":
        speed = 0.85 if low_auth or fail_streak else 1.05
        verb = "verbose" if low_auth or fail_streak else "brief"
        reps = 2 if fail_streak else 1
        confirm = low_auth or fail_streak
        intensity = 0.75 if low_auth else 0.45
        pattern = "low_conf" if low_auth else ("genuine" if state.auth_conf >= 0.75 else "unknown")
        return FeedbackAction(
            tts_speed=speed,
            verbosity=verb,
            repetition=reps,
            confirmation=confirm,
            haptic_intensity=intensity,
            haptic_pattern=pattern,
            request_next_view=request,
            condition=condition,
            reason="confidence_and_failures",
        )

    # confidence + emotion
    distressed = state.emotion.lower() in {"fear", "sadness", "anger", "disgust"} and state.emotion_conf >= 0.45
    speed = 0.75 if distressed else (0.85 if low_auth else 1.05)
    verb = "verbose" if distressed or low_auth else "brief"
    reps = 2 if distressed or fail_streak else 1
    confirm = distressed or low_auth or fail_streak
    intensity = 0.35 if distressed else (0.75 if low_auth else 0.45)
    if distressed:
        pattern = "unknown"
        reason = "emotion_distress_slows_and_repeats"
    elif low_auth:
        pattern = "low_conf"
        reason = "low_auth_confidence"
    else:
        pattern = "genuine"
        reason = "stable_confidence"
    if distressed:
        request = request or (state.num_views < 3 and state.task_state == "authenticating")
    return FeedbackAction(
        tts_speed=speed,
        verbosity=verb,
        repetition=reps,
        confirmation=confirm,
        haptic_intensity=intensity,
        haptic_pattern=pattern,
        request_next_view=request,
        condition=condition,
        reason=reason,
    )


def log_feedback(state: FeedbackState, action: FeedbackAction) -> dict:
    return {"state": asdict(state), "action": asdict(action)}
