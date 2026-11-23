"""Run the glass app itself (main.SmartGlass, headless like the glass) and press its buttons from a script.

The presses go through the app's own button handlers (_on_mode_press / set_mode, _on_action_press), with the real
camera, speech and models, so the field log (logs/field/<date>_<time>/) records them like a session by hand. Used after a
benchmark run to time the app on the code as it is, e.g. with AI_HAT=1.

    AI_HAT=1 python app_session_driver.py [presses per mode]       (run from savior_glass/)
"""
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("SHOW_PREVIEW", "0")
GLASS = Path(__file__).resolve().parents[2] / "savior_glass"
sys.path.insert(0, str(GLASS))
os.chdir(GLASS)

import config  # noqa: E402
import main as app_main  # noqa: E402
import utils  # noqa: E402

PRESSES = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def wait_idle(app, limit=120.0):
    """Until the current ACTION has been answered (the app's own busy flag) and speech is quiet."""
    t0 = time.time()
    time.sleep(0.3)
    while app._inferring.is_set() and time.time() - t0 < limit:
        time.sleep(0.1)


def main():
    utils.setup_logging()
    app = app_main.SmartGlass()
    runner = threading.Thread(target=app.run, name="app", daemon=True)
    runner.start()
    time.sleep(3)
    plan = [(config.MODE_CURRENCY, "currency"), (config.MODE_OCR, "text"), (config.MODE_OBJECT, "objects"),
            (config.MODE_EMOTION, "emotion")]
    for idx, name in plan:
        if app._current_mode != idx:
            app.set_mode(idx)
        app._ready[idx].wait(180)
        print(f"== {name}: ready", flush=True)
        time.sleep(2)
        for i in range(PRESSES):
            app._on_action_press()
            wait_idle(app)
            print(f"   {name}: ACTION {i + 1} answered", flush=True)
            time.sleep(4 if name != "objects" else 2)
        if name == "objects":
            time.sleep(30)      # object mode also scans and announces on its own every OBJECT_SCAN_INTERVAL
    app._running.clear()
    runner.join(60)
    print("session done", flush=True)


if __name__ == "__main__":
    main()
