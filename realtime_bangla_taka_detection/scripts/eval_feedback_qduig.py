"""Deterministic SCAFP feedback conditions. No human outcomes unless a study exists."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

from roboeye.qduig.artifacts import init_run, save_json
from roboeye.qduig.feedback import FeedbackState, log_feedback, scafp

EMOTION_JSON = WORKSPACE / "savior_glass" / "results" / "emotion_rafdb.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(ROOT / "results" / "qduig" / "feedback"))
    args = p.parse_args()
    out = Path(args.output_dir)
    init_run(out, {"name": "scafp"}, 42, "adaptive feedback", "Logged policy only. HUMAN_STUDY outcomes NOT_MEASURED.")

    emotion = json.loads(EMOTION_JSON.read_text(encoding="utf-8")) if EMOTION_JSON.is_file() else None
    grid = []
    for emotion_name in ("neutral", "happiness", "fear", "sadness"):
        for auth in (0.55, 0.80, 0.95):
            for views in (1, 3, 6):
                state = FeedbackState(
                    emotion=emotion_name,
                    emotion_conf=0.7 if emotion_name != "neutral" else 0.4,
                    detection_conf=0.85,
                    auth_conf=auth,
                    auth_uncertainty=1.0 - auth,
                    task_state="authenticating",
                    recent_failures=1 if auth < 0.6 else 0,
                    num_views=views,
                    pose_quality=0.6,
                )
                for cond in ("fixed", "confidence_aware", "confidence_emotion_aware"):
                    grid.append(log_feedback(state, scafp(state, cond)))
    save_json(out / "feedback_logs.json", grid)
    save_json(
        out / "emotion_preserved.json",
        {
            "source": str(EMOTION_JSON) if emotion else None,
            "accuracy": emotion.get("accuracy") if emotion else None,
            "macro_f1": emotion.get("macro_f1") if emotion else None,
            "n_test": emotion.get("n_test") if emotion else None,
            "retrain": False,
            "note": "Preserved measured RAF-DB result. Not replaced.",
        },
    )
    save_json(out / "human_study.json", {"HUMAN_STUDY": "NOT_MEASURED", "reason": "participants_unavailable"})
    print("feedback logs", len(grid), "human study NOT_MEASURED")


if __name__ == "__main__":
    main()
