"""Time the Pi's currency path of the code checked out now, the same way as scripts/benchmark_pi5.py.

    python time_current_currency.py <out.json>

100 calls after 3 warm-up calls on the same note pictures, with benchmark_pi5.py's own timer: currency mode end to end
(the detector config.py picks, then the safe-check), and the hand-held full-precision detector (best_handheld.pt) alone.
Written for the 29 October run: a349cfe moved the Pi from best_handheld_int8.onnx to best_handheld.pt after it.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "savior_glass" / "scripts"))
sys.path.insert(0, str(REPO / "savior_glass"))
import benchmark_pi5 as B  # noqa: E402
import config  # noqa: E402
from modes.currency_mode import CurrencyMode  # noqa: E402
from ultralytics import YOLO  # noqa: E402

notes = B.inputs()["notes"]
commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
out = {"commit": commit, "temp_c_start": B.cpu_temp_c(), "throttled_start": B.throttled(),
       "currency_yolo_path": Path(config.CURRENCY_YOLO_PATH).name, "announce_conf": config.CURRENCY_ANNOUNCE_CONF}
m = CurrencyMode()
m.activate()
out["currency_mode_full"] = B.time_fn(m.process_frame, notes, 3, 100)
print("currency_mode_full", out["currency_mode_full"], flush=True)
det = YOLO(str(REPO / "realtime_bangla_taka_detection" / "models" / "best_handheld.pt"), task="detect")
out["taka_yolo_handheld_pt"] = B.time_fn(lambda f: det.predict(f, imgsz=640, conf=0.25, verbose=False), notes, 3, 100)
print("taka_yolo_handheld_pt", out["taka_yolo_handheld_pt"], flush=True)
out["throttled_end"] = B.throttled()
Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
