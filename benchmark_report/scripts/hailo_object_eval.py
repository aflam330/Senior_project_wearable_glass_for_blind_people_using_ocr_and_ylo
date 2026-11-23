"""Object mode on the AI HAT. python hailo_object_eval.py <out.json>
1) default (CPU) path of the new object_mode.py against the committed one: same boxes and announcements;
2) Hailo against CPU: same objects, and the time per picture."""
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
import types

import cv2

GLASS = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "savior_glass")
OUT = os.path.abspath(sys.argv[1])
sys.path.insert(0, GLASS)
os.chdir(GLASS)
import config  # noqa: E402
import modes  # noqa: E402,F401  (package, for the relative import of base_mode)

A = "/home/memo/.venvs/savior_glass/lib/python3.13/site-packages/ultralytics/assets/"
REPO = os.path.dirname(GLASS)
paths = [A + "bus.jpg", A + "zidane.jpg"] + sorted(
    os.path.join(REPO, "benchmark_report/2026-11-23/raw", f) for f in os.listdir(os.path.join(REPO, "benchmark_report/2026-11-23/raw"))
    if f.endswith(".png"))
frames = [(os.path.basename(p), cv2.resize(cv2.imread(p), (640, 480))) for p in paths if cv2.imread(p) is not None]
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)          # real frames from the glass webcam (kept in memory only)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
t_end = time.time() + 2
while time.time() < t_end:
    cap.read()
for i in range(10):
    ok, f = cap.read()
    if ok:
        frames.append((f"webcam_{i}", f))
    time.sleep(0.5)
cap.release()

# the committed object_mode.py, loaded as modes.object_mode_old
old_src = subprocess.run(["git", "show", "HEAD:savior_glass/modes/object_mode.py"], capture_output=True, text=True,
                         cwd=GLASS).stdout
old = types.ModuleType("modes.object_mode_old")
old.__package__ = "modes"
exec(compile(old_src, "object_mode_old.py", "exec"), old.__dict__)
from modes import object_mode as new  # noqa: E402

def run(m, frame):
    m._last_announced.clear()
    m._force_announce = True
    said = m.process_frame(frame)
    return said, getattr(m, "last_boxes", [])

o, n = old.ObjectMode(), new.ObjectMode()
o.activate(); n.activate()
same = 0
for name, f in frames:
    a, b = run(o, f), run(n, f)
    ok = a[0] == b[0] and [x[:4] + (x[4],) for x in a[1]] == [x[:4] + (x[4],) for x in b[1]] and \
        all(abs(x[5] - y[5]) < 1e-6 for x, y in zip(a[1], b[1]))
    same += ok
    if not ok:
        print("DIFF", name, a, b)
default_same = same
print(f"default CPU path: {same} of {len(frames)} pictures give the same boxes and announcement as the committed code")

config.OBJECT_BACKEND = "hailo"
h = new.ObjectMode(); h.activate()
assert h._hailo is not None, "Hailo did not load"
rows = []
for name, f in frames:
    cpu = n.detect(f); hai = h.detect(f)
    cn = sorted({d[0] for d in cpu if d[1] >= config.OBJECT_CLASS_CONFIDENCE.get(d[0], 0)})
    hn = sorted({d[0] for d in hai if d[1] >= config.OBJECT_CLASS_CONFIDENCE.get(d[0], 0)})
    rows.append((name, cn, hn))
    print(f"{name:40s} CPU {cn}  HAILO {hn}")
print("same object list on", sum(c == hh for _, c, hh in rows), "of", len(rows), "pictures")

def timeit(fn, k=40):
    for _, f in frames[:3]:
        fn(f)
    ms = []
    for i in range(k):
        f = frames[i % len(frames)][1]
        t = time.perf_counter(); fn(f); ms.append((time.perf_counter() - t) * 1000)
    ms.sort()
    return round(statistics.median(ms), 1), round(ms[int(0.95 * (len(ms) - 1))], 1)

c_ms, h_ms = timeit(n.detect), timeit(h.detect, 200)
print("CPU   detect ms (median, p95):", c_ms)
print("HAILO detect ms (median, p95):", h_ms)
json.dump({"n": len(frames), "default_same": default_same, "same": sum(c == hh for _, c, hh in rows),
           "cpu_ms_median": c_ms[0], "cpu_ms_p95": c_ms[1], "hailo_ms_median": h_ms[0], "hailo_ms_p95": h_ms[1],
           "pictures": [{"name": nm, "cpu": c, "hailo": hh} for nm, c, hh in rows]}, open(OUT, "w"), indent=1)
h.cleanup()
