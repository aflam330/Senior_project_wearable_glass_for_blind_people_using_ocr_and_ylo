"""Text mode with the text found on the AI HAT (OCR_DETECTOR=hailo) against the Pi's CPU detector (8-bit CRAFT), on
real photos from the glass camera with typed reference text (~/Desktop/glass_ocr/glass_set, 50 shots of 14 pages).

One OCRMode, one letter recogniser; only reader.detect is switched between the two detectors, so any difference in
the text comes from the detector alone.
  1. find-text time on every shot, both detectors (the picture text mode reads: camera width 640, enlarged to 1,200);
  2. the full read (OCRMode.process_frame, the plain path) on one shot per page, both detectors: time and character
     error (Unicode NFC, whitespace collapsed, as glass_ocr/eval_glass.py scores).

python hailo_ocr_glass_eval.py <out.json>   (run from savior_glass/)
"""
import json
import os
import re
import statistics as st
import sys
import time
import unicodedata
from pathlib import Path

os.environ["OCR_DETECTOR"] = "hailo"
GLASS = Path(__file__).resolve().parents[2] / "savior_glass"
sys.path.insert(0, str(GLASS))
os.chdir(GLASS)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402

SET = Path(os.environ.get("GLASS_SET", str(Path.home() / "Desktop/glass_ocr/glass_set")))


def norm(s):
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[#*_>`]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def edit_distance(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    ar = np.arange(len(b) + 1)
    enc = {c: i for i, c in enumerate(set(a) | set(b))}
    bv = np.array([enc[c] for c in b])
    prev = ar.copy()
    for i, ca in enumerate(a, 1):
        tmp = np.empty_like(prev)
        tmp[0] = i
        tmp[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (bv != enc[ca]))
        prev = np.minimum.accumulate(tmp - ar) + ar
    return int(prev[-1])


def cer(hyp, ref):
    h, r = norm(hyp), norm(ref)
    return 100.0 * edit_distance(h, r) / max(1, len(r))


def app_frame(path):
    """The shot as the glass camera gives it to text mode (CAMERA_WIDTH, 640 by default)."""
    img = cv2.imread(str(path))
    w = config.CAMERA_WIDTH
    return cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_AREA)


def main():
    out = Path(sys.argv[1])
    from modes.ocr_mode import OCRMode
    m = OCRMode()
    m.activate()
    r = m._reader
    hailo_detect = r.__dict__.get("detect")
    assert hailo_detect is not None and r.__dict__.get("_hailo_det") is not None, "Hailo text detector did not install"
    cpu_detect = type(r).detect.__get__(r)            # EasyOCR's own detect with the Pi's CPU detector (8-bit CRAFT)
    cpu_name = f"{type(r.detector).__name__} ({config.OCR_HAILO_CPU_FALLBACK})"
    spent = {"t": 0.0}

    def use(fn):
        def timed(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                spent["t"] += time.perf_counter() - t0
        r.detect = timed

    shots = sorted(p for p in SET.iterdir() if (p / "page.jpg").is_file() and (p / "expected.txt").is_file())
    frames = {p.name: app_frame(p / "page.jpg") for p in shots}
    truth = {p.name: (p / "expected.txt").read_text(encoding="utf-8") for p in shots}

    # 1. find-text time, every shot (the CLAHE picture _read_v2 gives the detector)
    def det_input(f):
        h, w = f.shape[:2]
        rw = int(getattr(config, "OCR_READ_WIDTH", 1200))
        if w < rw:
            f = cv2.resize(f, (rw, int(h * rw / w)), interpolation=cv2.INTER_CUBIC)
        return cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))

    kw = dict(getattr(config, "OCR_V2_PARAMS", {}))
    det_kw = {k: kw[k] for k in ("text_threshold", "low_text", "link_threshold", "mag_ratio", "slope_ths") if k in kw}
    det_kw.update(canvas_size=getattr(config, "OCR_CANVAS_SIZE", 2560), add_margin=float(getattr(config, "OCR_ADD_MARGIN", 0.1)),
                  width_ths=0.7, height_ths=0.7)
    find = {"cpu": [], "hailo": []}
    nbox = {"cpu": [], "hailo": []}
    for name in ("hailo", "cpu"):
        fn = hailo_detect if name == "hailo" else cpu_detect
        fn(det_input(frames[shots[0].name]), **det_kw)   # warm-up
        for p in shots:
            g = det_input(frames[p.name])
            t0 = time.perf_counter()
            hl, fl = fn(g, **det_kw)
            find[name].append((time.perf_counter() - t0) * 1000)
            nbox[name].append(len(hl[0]) + len(fl[0]))
        print(name, "find-text median ms", round(st.median(find[name]), 1), flush=True)

    # 2. full read, one shot per page
    first = {}
    for p in shots:
        first.setdefault(p.name.split("_")[0], p.name)
    rows = []
    for page, shot in sorted(first.items()):
        row = {"shot": shot, "ref_chars": len(norm(truth[shot]))}
        for name in ("hailo", "cpu"):
            use(hailo_detect if name == "hailo" else cpu_detect)
            m.speak_line = None
            spent["t"] = 0.0
            t0 = time.perf_counter()
            m.process_frame(frames[shot])
            total = time.perf_counter() - t0
            text = m._stored_text or ""
            row[name] = {"total_s": round(total, 2), "find_s": round(spent["t"], 2), "cer": round(cer(text, truth[shot]), 2),
                         "text": text}
        print(shot, "hailo", row["hailo"]["total_s"], "s", row["hailo"]["cer"], "% | cpu", row["cpu"]["total_s"], "s",
              row["cpu"]["cer"], "%", flush=True)
        rows.append(row)

    def summ(name):
        c = [x[name]["cer"] for x in rows]
        return {"cer_mean": round(st.mean(c), 2), "cer_median": round(st.median(c), 2),
                "total_s_median": round(st.median(x[name]["total_s"] for x in rows), 2),
                "find_s_median": round(st.median(x[name]["find_s"] for x in rows), 2),
                "find_ms_all_shots_median": round(st.median(find[name]), 1),
                "find_ms_all_shots_p95": round(sorted(find[name])[int(0.95 * (len(find[name]) - 1))], 1),
                "boxes_median": st.median(nbox[name])}

    res = {"set": str(SET), "shots": len(shots), "pages": len(rows), "camera_width": config.CAMERA_WIDTH,
           "read_width": getattr(config, "OCR_READ_WIDTH", 1200), "hailo_mode": config.OCR_HAILO_MODE,
           "cpu_detector": cpu_name, "summary": {"hailo": summ("hailo"), "cpu": summ("cpu")},
           "find_ms": find, "boxes": nbox, "pages_read": rows}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res["summary"], indent=1))
    m.cleanup()


if __name__ == "__main__":
    main()
