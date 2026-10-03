"""Self-test for field_log.py and field_report.py (temporary folder; nothing is written to logs/).

Checks: the cost of one log() call in the app thread, 10,000 events from 4 threads all written,
system samples written, frames saved and capped, labels.csv created, accuracy computed once labels are filled.
Run: python scripts/test_field_log.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from field_log import FieldLogger  # noqa: E402
import field_report  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        lg = FieldLogger(d, sys_interval_s=0.2, save_frames=True, max_frames=3, settings={"test": True})
        frame = np.zeros((480, 640, 3), np.uint8)
        # 1. cost in the calling thread
        n = 2000
        t0 = time.perf_counter()
        for i in range(n):
            lg.log("result", "currency", latency_ms=300.0 + i % 50, result="পাঁচশত টাকার নোট। সম্ভবত আসল",
                   denomination="500_taka", confidence=0.93, verdict="likely_genuine", score=0.9997, press_event=i)
        per_call_us = (time.perf_counter() - t0) / n * 1e6
        # 2. concurrency
        def burst():
            for _ in range(2000):
                lg.log("object_announce", "object", latency_ms=200.0, result="চেয়ার")
        ths = [threading.Thread(target=burst) for _ in range(4)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        # 3. frames (capped at 3) and an OCR answer
        press = [lg.log("action_press", "ocr", frame=frame) for _ in range(5)]
        lg.log("result", "ocr", latency_ms=1900.0, result="লেখা সংরক্ষণ করা হয়েছে", lang="bn", ocr_text="জরুরি বিভাগ",
               press_event=press[0])
        lg.log("result", "currency", latency_ms=310.0, result="এক হাজার টাকার নোট। আসল কিনা হাতে যাচাই করুন",
               denomination="1000_taka", confidence=0.88, verdict="check_by_hand", score=0.42, press_event=press[1])
        time.sleep(0.7)
        lg.close()
        run = Path(lg.dir)
        ev = list(csv.DictReader(open(run / "events.csv", encoding="utf-8-sig")))
        sysr = list(csv.DictReader(open(run / "system.csv", encoding="utf-8-sig")))
        assert len(ev) == n + 8000 + 5 + 2 + 1, len(ev)  # + session_end
        assert lg.dropped == 0
        assert len(sysr) >= 2
        assert len(list((run / "frames").glob("*.jpg"))) == 3
        assert json.loads((run / "session.json").read_text(encoding="utf-8"))["events_logged"] == len(ev)
        # 4. report + labels
        s = field_report.summarise(run)
        assert s["accuracy"]["labelled_rows"] == 0 and (run / "labels.csv").exists()
        rows = list(csv.DictReader(open(run / "labels.csv", encoding="utf-8-sig")))
        for r in rows:  # test labels in the temp folder only
            if r["mode"] == "currency":
                r["true_denomination"] = r["denomination"] if r["denomination"] == "500_taka" else "500_taka"
                r["true_label"] = "genuine" if r["verdict"] == "likely_genuine" else "counterfeit"
            if r["mode"] == "ocr":
                r["true_text"] = "জরুরি বিভাগ"
        with open(run / "labels.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=field_report.LABEL_COLS)
            w.writeheader()
            w.writerows(rows)
        a = field_report.summarise(run)["accuracy"]
        assert a["denomination"]["n"] == n + 1 and a["denomination"]["correct"] == n, a
        assert a["counterfeit_safety"]["counterfeit_passed_as_genuine"] == 0, a
        assert a["ocr"]["mean_cer"] == 0.0, a
        print(f"field log: all checks pass. log() costs {per_call_us:.1f} µs per call in the app thread; "
              f"{len(ev)} rows, {len(sysr)} system samples, 0 dropped")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
