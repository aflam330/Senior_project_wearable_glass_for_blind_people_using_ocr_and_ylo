"""The Raspberry Pi copy (8-bit ONNX) of the re-trained hand-held detector (models/best_nstu.pt = epoch 9 of
runs/nstu_weak/seed42, chosen in eval_nstu_weak.py).

models/best_nstu_int8.onnx is made by scripts/quantize_int8.py (calibration on synthetic validation pictures, the
procedure of the deployed best_handheld_int8.onnx). Its announce threshold is chosen by the open-set rule on the
VALIDATION pool of eval_nstu_weak.py; then, once: NSTU test (one image per original) and Bangla Money, on the CPU,
with the deployed 8-bit detector (best_handheld_int8.onnx at its deployed threshold 0.80) for comparison.

    python scripts/eval/eval_nstu_int8.py
Output: results/nstu_weak/int8.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval import eval_closeup_ft as CF  # noqa: E402
from scripts.eval import eval_open_set as OS  # noqa: E402
from scripts.eval.eval_nstu_weak import val_pool  # noqa: E402

OUT = ROOT / "results" / "nstu_weak" / "int8.json"
MODELS = {"deployed_int8": ("models/best_handheld_int8.onnx", 0.80), "nstu_int8": ("models/best_nstu_int8.onnx", None)}


def main():
    from ultralytics import YOLO
    CF.DEV = "cpu"
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"date": "2026-11-23"}
    pool = val_pool()
    for name, (path, tau) in MODELS.items():
        if name in res:
            continue
        model = YOLO(str(ROOT / path), task="detect")
        r = {"weights": path, "size_mb": round((ROOT / path).stat().st_size / 1e6, 1)}
        if tau is None:
            rows = OS.score_items(model, pool)
            sweep = [OS.tally(rows, t) for t in OS.GRID]
            best = max(sweep, key=lambda s: (s["utility_correct_minus_wrong"], -s["tau"]))
            tau, r["validation_at_tau"] = best["tau"], best
        r["tau"] = tau
        nt = CF.predict(model, CF.nstu("test"))
        r["nstu_test"] = CF.tally(nt, tau)
        r["bangla_money"] = CF.tally(CF.predict(model, CF.bangla_money()), tau)
        import cv2
        img = cv2.imread(str(CF.nstu("test")[0][0]))
        for _ in range(3):
            model.predict(img, imgsz=640, verbose=False, device="cpu")
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            model.predict(img, imgsz=640, verbose=False, device="cpu")
            ts.append((time.perf_counter() - t0) * 1000)
        r["laptop_cpu_latency_ms_median"] = round(sorted(ts)[10], 1)
        res[name] = r
        OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(name, json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
