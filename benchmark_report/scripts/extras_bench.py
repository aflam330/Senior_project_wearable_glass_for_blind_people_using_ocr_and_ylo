"""The timings scripts/benchmark_pi5.py cannot give on this device, because its datasets are not on the Pi.

  emotion_with_face   the emotion detector on a picture that has a face in it (benchmark_pi5 feeds a black frame
                      when RAF-DB is absent, so only the face detector runs there)
  object_yolov8n/s    both everyday-object detectors, with the settings of ObjectMode.process_frame
  speech_*            espeak-ng turning a sentence into audio (written to a pipe, not played)
  camera              time to open the camera and the interval between frames (skipped if no camera answers)

    python extras_bench.py <out.json> [phase.txt] [emotion object speech camera]

Name one or more parts to run only those; their results replace the same parts in an existing out.json.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
GLASS = HERE.parents[2] / "savior_glass"
sys.path.insert(0, str(GLASS))
sys.path.insert(0, str(GLASS / "scripts"))

import cv2  # noqa: E402

import config  # noqa: E402
from benchmark_pi5 import camera_like, cpu_temp_c, inputs, time_fn  # noqa: E402

ITERS, WARMUP = 30, 3


def main() -> None:
    out = Path(sys.argv[1])
    phase = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    def set_phase(name):
        if phase:
            phase.write_text(name)

    only = set(sys.argv[3:])
    res, skipped = {}, {}

    def wanted(part):
        return not only or part in only

    # a real face: the astronaut picture that ships with scikit-image (an EasyOCR dependency)
    set_phase("extras_emotion")
    try:
        if not wanted("emotion"):
            raise StopIteration
        from skimage import data
        from assistive import EmotionDetector
        face = camera_like(cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out.parent / "emotion_input_face.png"), face)
        t0 = time.perf_counter()
        det = EmotionDetector(download=False)
        load_ms = (time.perf_counter() - t0) * 1000.0
        first = det.predict(face)
        r = time_fn(det.predict, [face], WARMUP, ITERS)
        r.update(load_ms=round(load_ms, 1), face_found=first.get("face") is not None, backend=first.get("backend"),
                 label=first.get("label"), face_detector=getattr(det, "face_kind", None),
                 strong_classifier=getattr(det, "rgb_net", None) is not None)
        res["emotion_with_face"] = r
        print("emotion_with_face", r["median_ms"], "ms", r["backend"], r["label"], flush=True)
    except StopIteration:
        pass
    except Exception as exc:  # noqa: BLE001
        skipped["emotion_with_face"] = f"{type(exc).__name__}: {exc}"[:300]

    set_phase("extras_object")
    scene = inputs()["scenes"]
    for name in ("yolov8n", "yolov8s") if wanted("object") else ():
        path = GLASS / "models" / f"{name}.pt"
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            from ultralytics import YOLO
            t0 = time.perf_counter()
            model = YOLO(str(path))
            load_ms = (time.perf_counter() - t0) * 1000.0
            r = time_fn(lambda f, m=model: m.predict(f, conf=config.OBJECT_CONFIDENCE, verbose=False, stream=False),
                        scene, WARMUP, ITERS)
            r["load_ms"] = round(load_ms, 1)
            res[f"object_{name}"] = r
            print(f"object_{name}", r["median_ms"], "ms", flush=True)
        except Exception as exc:  # noqa: BLE001
            skipped[f"object_{name}"] = f"{type(exc).__name__}: {exc}"[:300]

    set_phase("extras_speech")
    sentences = {
        "speech_bn_short": ("পাঁচশত টাকার নোট। আসল কিনা হাতে যাচাই করুন", config.ESPEAK_VOICE_BN),
        "speech_en_short": ("The text says: EXIT 12 Bus Stop Main Entrance", config.ESPEAK_VOICE_EN),
        "speech_bn_page": ("লেখা আছে: সরকারি প্রাথমিক বিদ্যালয়। জুতা খুলে প্রবেশ করুন। ঢাকা মেডিকেল কলেজ। রক্ত পরীক্ষা কেন্দ্র। "
                           "শিশুদের খেলার মাঠ। ভাড়া দেওয়া হবে", config.ESPEAK_VOICE_BN),
    }
    for name, (text, voice) in sentences.items() if wanted("speech") else ():
        try:
            ms, audio_s = [], None
            for _ in range(8):
                t0 = time.perf_counter()
                wav = subprocess.run(["espeak-ng", "-v", voice, "-s", str(config.ESPEAK_SPEED), "-p", str(config.ESPEAK_PITCH),
                                      "-g", str(config.ESPEAK_WORD_GAP), "--stdout", text], capture_output=True, check=True).stdout
                ms.append((time.perf_counter() - t0) * 1000.0)
                audio_s = max(0, len(wav) - 44) / (22050 * 2)  # espeak-ng writes 22.05 kHz 16-bit mono
            res[name] = {"median_ms": round(statistics.median(ms), 1), "max_ms": round(max(ms), 1), "iters": len(ms),
                         "characters": len(text), "audio_seconds": round(audio_s, 2), "voice": voice}
            print(name, res[name]["median_ms"], "ms for", res[name]["audio_seconds"], "s of audio", flush=True)
        except Exception as exc:  # noqa: BLE001
            skipped[name] = f"{type(exc).__name__}: {exc}"[:300]

    set_phase("extras_camera")
    try:
        if not wanted("camera"):
            raise StopIteration
        t0 = time.perf_counter()
        cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_V4L2)   # the same settings as utils.Camera._open_cv
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"camera index {config.CAMERA_INDEX} gave no frame")
        open_ms = (time.perf_counter() - t0) * 1000.0
        for _ in range(10):
            cap.read()
        gaps, last = [], time.perf_counter()
        for _ in range(90):
            cap.read()
            now = time.perf_counter()
            gaps.append((now - last) * 1000.0)
            last = now
        cap.release()
        gaps.sort()
        res["camera"] = {"open_to_first_frame_ms": round(open_ms, 1), "frame_interval_ms_median": round(statistics.median(gaps), 2),
                         "frame_interval_ms_p95": round(gaps[int(0.95 * len(gaps))], 2),
                         "fps_median": round(1000.0 / statistics.median(gaps), 1), "frames": len(gaps),
                         "size": f"{frame.shape[1]}x{frame.shape[0]}", "index": config.CAMERA_INDEX}
        print("camera", res["camera"], flush=True)
    except StopIteration:
        pass
    except Exception as exc:  # noqa: BLE001
        skipped["camera"] = f"{type(exc).__name__}: {exc}"[:300]

    if only and out.is_file():
        old = json.loads(out.read_text(encoding="utf-8"))
        res = {**old.get("results", {}), **res}
        skipped = {**{k: v for k, v in old.get("skipped", {}).items() if k not in res}, **skipped}
    out.write_text(json.dumps({"results": res, "skipped": skipped, "cpu_temp_c_end": cpu_temp_c()}, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    set_phase("idle")
    print("wrote", out, "skipped", skipped)


if __name__ == "__main__":
    main()
