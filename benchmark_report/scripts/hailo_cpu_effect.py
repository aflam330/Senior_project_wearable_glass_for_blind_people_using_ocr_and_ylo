"""Run from savior_glass/. Does having the AI HAT models open in the process slow the CPU ONNX models? Same model, same frames, three blocks."""
import sys, time, statistics, glob, os, json
sys.path.insert(0, "."); import config, cv2, numpy as np
from ultralytics import YOLO
path = os.path.join(os.path.dirname(config.BASE_DIR), "realtime_bangla_taka_detection", "models", "best_int8.onnx")
m = YOLO(path, task="detect")
frames = [cv2.resize(cv2.imread(p), (640, 480)) for p in sorted(glob.glob("../benchmark_report/2026-11-23/raw/*.png"))[:4]]
def block(n=40):
    for f in frames: m.predict(f, verbose=False, imgsz=640)
    t = []
    for i in range(n):
        t0 = time.perf_counter(); m.predict(frames[i % len(frames)], verbose=False, imgsz=640); t.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(t), 1)
res = {"without_ai_hat": block()}
from hailo_yolo import HailoYolo; from hailo_text_detector import HailoTextDetector
y = HailoYolo(config.HAILO_YOLO_HEF); d = HailoTextDetector(config.HAILO_OCR_DET_HEF)
y.detect(frames[0], 0.25); d.boxes(frames[0])
res["ai_hat_models_open_idle"] = block()
for f in frames: y.detect(f, 0.25)
res["ai_hat_just_used"] = block()
y.close(); d.close()
res["after_closing_ai_hat"] = block()
print(json.dumps(res))
