"""Guided back-lit capture for the watermark check (the interaction the user study measures).

After the denomination is announced for a 500 / 1,000 Taka note, the glass asks the user to hold the
note up to a light. It then looks at frames for up to CAPTURE_GUIDE_TIMEOUT_S seconds and rejects
  no note     -> "নোটটি ক্যামেরার সামনে ধরুন"   (hold the note in front of the camera)
  dark        -> "নোটটি আলোর আরও কাছে ধরুন"    (hold it closer to the light)
  blurry      -> "নোটটি স্থির রাখুন"            (hold it still)
  no window   -> "নোটটি আলোর সামনে সোজা করে ধরুন" (hold it straight against the light)
Only the first frame that passes runs the watermark model. The answer is the existing one:
"জলছাপ স্পষ্ট" above the validation threshold, otherwise "check by hand". It never says "counterfeit".
If no frame passes in time: "আসল কিনা হাতে যাচাই করুন".

Thresholds (CAPTURE_GUIDE_MIN_MEAN, CAPTURE_GUIDE_MIN_LAPVAR in config) are the 2nd percentile of
JaalTaka VALIDATION back-lit photos (realtime_bangla_taka_detection/results/capture_guide/thresholds.json).
They are a starting point: the glass camera needs its own calibration before the study.

Every guided check appends one JSON line to logs/study_events.jsonl (participant id from the
STUDY_PARTICIPANT environment variable): prompts spoken, frames rejected by reason, outcome, seconds.
That file gives time per note and how often the glass asked for a hand check.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from typing import Callable, Optional

import cv2
import numpy as np

import config

logger = logging.getLogger("smart_glass.capture_guide")

PROMPTS = {
    "start": "নোটটি আলোর দিকে তুলে ধরুন",  # hold the note up to the light
    "no_note": "নোটটি ক্যামেরার সামনে ধরুন",
    "dark": "নোটটি আলোর আরও কাছে ধরুন",
    "blurry": "নোটটি স্থির রাখুন",
    "no_window": "নোটটি আলোর সামনে সোজা করে ধরুন",
    "timeout": "ছবি পরিষ্কার হয়নি। আসল কিনা হাতে যাচাই করুন",  # no usable frame: check by hand
}


def frame_quality(crop: np.ndarray) -> dict:
    """Mean grey level and Laplacian variance on the note crop scaled to 400 px wide."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (400, max(1, round(g.shape[0] * 400 / g.shape[1]))), interpolation=cv2.INTER_AREA)
    return {"mean": float(g.mean()), "lap_var": float(cv2.Laplacian(g, cv2.CV_64F).var())}


def assess(crop: Optional[np.ndarray], min_mean: float, min_lapvar: float) -> tuple[str, dict]:
    if crop is None or crop.shape[0] < 40 or crop.shape[1] < 40:
        return "no_note", {}
    q = frame_quality(crop)
    if q["mean"] < min_mean:
        return "dark", q
    if q["lap_var"] < min_lapvar:
        return "blurry", q
    return "ok", q


class CaptureGuide:
    def __init__(self, checker, log_path: Optional[str] = None):
        self.checker = checker
        self.min_mean = float(getattr(config, "CAPTURE_GUIDE_MIN_MEAN", 0.0))
        self.min_lapvar = float(getattr(config, "CAPTURE_GUIDE_MIN_LAPVAR", 0.0))
        self.timeout = float(getattr(config, "CAPTURE_GUIDE_TIMEOUT_S", 10.0))
        self.prompt_gap = float(getattr(config, "CAPTURE_GUIDE_PROMPT_GAP_S", 2.5))
        self.log_path = log_path or os.path.join(config.BASE_DIR, "logs", "study_events.jsonl")
        self.condition = getattr(config, "STUDY_CONDITION", "guided")
        if self.condition == "unguided":  # study baseline: no frame rejection
            self.min_mean = self.min_lapvar = 0.0

    def run(self, denom: str, get_frame: Callable[[], Optional[np.ndarray]],
            locate: Callable[[np.ndarray], Optional[np.ndarray]], speak: Callable[[str], None],
            clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> dict:
        """Guide until one frame passes, then score it. locate(frame) returns the note crop or None."""
        t0 = clock()
        speak(PROMPTS["start"])
        last_prompt, prompts, rejected, frames = t0, ["start"], {}, 0
        prob, outcome, q = None, "timeout", {}
        while clock() - t0 < self.timeout:
            frame = get_frame()
            if frame is None:
                sleep(0.05)
                continue
            frames += 1
            crop = locate(frame)
            status, q = assess(crop, self.min_mean, self.min_lapvar)
            if status == "ok":
                prob = self.checker.genuine_prob(crop, denom)
                if prob is not None:
                    outcome = "likely_genuine" if prob > getattr(config, "WATERMARK_CLEAR_THRESHOLD", 0.5) else "check_by_hand"
                    break
                status = "no_window"
            rejected[status] = rejected.get(status, 0) + 1
            if clock() - last_prompt >= self.prompt_gap:
                speak(PROMPTS[status])
                prompts.append(status)
                last_prompt = clock()
            sleep(0.05)
        answer = self.checker.sentence(prob) if prob is not None else PROMPTS["timeout"]
        speak(answer)
        event = {"at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                 "participant": os.environ.get("STUDY_PARTICIPANT", ""), "condition": self.condition, "denomination": denom,
                 "outcome": outcome, "hand_check_requested": outcome != "likely_genuine",
                 "watermark_prob": prob, "seconds": round(clock() - t0, 3), "frames": frames,
                 "rejected": rejected, "prompts": prompts, "final_quality": q,
                 "thresholds": {"min_mean": self.min_mean, "min_lapvar": self.min_lapvar}}
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Study log not written: %s", exc)
        return event
