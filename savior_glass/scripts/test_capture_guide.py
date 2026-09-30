"""Capture guide logic without a camera: fake clock, fake frames, fake watermark checker.

Checks: dark / blurry / missing-note frames are rejected with the right spoken prompt, only a good
frame reaches the watermark model, a timeout ends in "check by hand", nothing ever says "counterfeit"
(জাল), and one study-log line is written per check.
Run: python scripts/test_capture_guide.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modes.capture_guide import PROMPTS, CaptureGuide  # noqa: E402

rng = np.random.default_rng(0)
GOOD = rng.integers(60, 255, (300, 600, 3), dtype=np.uint8)            # bright, sharp texture
DARK = (GOOD * 0.2).astype(np.uint8)
BLURRY = np.full((300, 600, 3), 160, np.uint8)                         # no detail at all


class FakeChecker:
    def __init__(self, p):
        self.p, self.calls = p, 0

    def genuine_prob(self, crop, denom):
        self.calls += 1
        assert crop.mean() > 60, "a rejected frame reached the watermark model"
        return self.p

    def sentence(self, p):
        return "জলছাপ স্পষ্ট।" if p is not None and p > 0.5 else "জলছাপ স্পষ্ট নয়। আসল কিনা হাতে যাচাই করুন।"


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += max(s, 1.0)  # one frame per simulated second


def run(frames, p, log):
    said, clock, it = [], Clock(), iter(frames)
    g = CaptureGuide(FakeChecker(p), log_path=log)
    g.min_mean, g.min_lapvar, g.timeout, g.prompt_gap = 73.87, 106.75, 10.0, 0.0
    ev = g.run("500_taka", lambda: next(it, None), lambda f: f if f is not None and f.shape[0] > 10 else None,
               said.append, clock=clock, sleep=clock.sleep)
    return ev, said, g.checker


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "study.jsonl")
        tiny = np.zeros((5, 5, 3), np.uint8)
        ev, said, chk = run([tiny, DARK, BLURRY, GOOD], 0.9, log)
        assert ev["outcome"] == "likely_genuine" and chk.calls == 1, ev
        assert ev["rejected"] == {"no_note": 1, "dark": 1, "blurry": 1}, ev["rejected"]
        assert said[:4] == [PROMPTS["start"], PROMPTS["no_note"], PROMPTS["dark"], PROMPTS["blurry"]], said
        ev2, said2, _ = run([GOOD], 0.1, log)
        assert ev2["outcome"] == "check_by_hand" and ev2["hand_check_requested"]
        ev3, said3, chk3 = run([DARK] * 30, 0.9, log)
        assert ev3["outcome"] == "timeout" and chk3.calls == 0 and said3[-1] == PROMPTS["timeout"]
        assert not any("জাল" in s for s in said + said2 + said3), "said counterfeit"
        lines = [json.loads(x) for x in open(log, encoding="utf-8")]
        assert len(lines) == 3 and {x["outcome"] for x in lines} == {"likely_genuine", "check_by_hand", "timeout"}
    print("capture guide: all checks pass")


if __name__ == "__main__":
    main()
