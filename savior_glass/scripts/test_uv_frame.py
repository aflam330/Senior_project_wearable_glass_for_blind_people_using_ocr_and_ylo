"""A currency press with the note under the UV lamp answers with the UV fibre check; an ordinary press does not.

Runs the real CurrencyMode (detector, UV check) on the UV photographs in data set/uv_notes, turned landscape and cut to
a 640x480 camera frame as the glass would see them, and on ordinary note photos. Checks the rules and writes how often each
answer came: results/uv_check/uv_frame_app.json. The UV photos are phone photographs, not the C270 webcam.

    python scripts/test_uv_frame.py
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import config  # noqa: E402
from modes.currency_mode import CurrencyMode, is_uv_frame  # noqa: E402
from uv_frame_rule import camera_frame  # noqa: E402  (the same webcam-style frame as the rule)

DATA = HERE.parent.parent / "data set"
OUT = HERE.parent / "results" / "uv_check" / "uv_frame_app.json"


def read(p: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def main() -> None:
    mode = CurrencyMode()
    mode.activate()
    assert mode._yolo is not None, "currency detector did not load"
    split = {r["file"]: r["split"] for r in csv.DictReader(open(DATA / "uv_notes" / "boxes.csv", encoding="utf-8"))}
    counts: dict = collections.defaultdict(collections.Counter)
    value: collections.Counter = collections.Counter()
    for p in sorted((DATA / "uv_notes").glob("*/*.png")):
        frame = camera_frame(read(p))
        assert is_uv_frame(frame), p.name
        text = mode.process_frame(frame)
        assert mode.uv_answered and not mode.wants_guided_watermark(), p.name
        assert "জাল" not in text, text                       # never "counterfeit"
        assert "ইউভি" in text, text                          # every answer is about the UV check
        assert "টাকার নোট" not in text, text                 # the value is not spoken under the lamp
        top = (mode.last_hits or [{}])[0]
        counts[split.get(p.name, "?")][top.get("auth", "no_note")] += 1
        # would the value have been right? (why it is not spoken)
        if top:
            sure = "sure" if top["conf"] >= config.CURRENCY_ANNOUNCE_CONF else "unsure"
            value[f"{sure}_{'right' if top['name'] == p.parent.name + '_taka' else 'wrong'}"] += 1
        else:
            value["no_box"] += 1
    ordinary = sorted((DATA / "currency_yolo_data" / "valid" / "images").glob("*"))[::100]
    for p in ordinary:
        frame = camera_frame(read(p))
        text = mode.process_frame(frame)
        assert not mode.uv_answered and "ইউভি" not in text, (p.name, text)
    res = {"uv_photos_by_split": {k: dict(v) for k, v in counts.items()},
           "detector_value_under_uv": dict(value), "ordinary_photos_checked": len(ordinary),
           "detector": Path(config.CURRENCY_YOLO_PATH).name, "announce_conf": config.CURRENCY_ANNOUNCE_CONF,
           "frame": "photo turned landscape, centre cut to 4:3, 640x480", "not_measured": "the C270 webcam under the lamp; counterfeit notes"}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print("uv frame: all checks pass")


if __name__ == "__main__":
    main()
