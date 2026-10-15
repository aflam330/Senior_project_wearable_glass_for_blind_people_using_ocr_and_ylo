"""The Raspberry Pi copy (8-bit ONNX) of the hand-held detector: calibration, threshold, one test read.

Two 8-bit versions of models/best_handheld.pt (epoch 8 of the close-up fine-tune, chosen in eval_closeup_ft.py):
  int8_scene     calibrated on 200 synthetic VALIDATION scene pictures (the procedure used for best_int8.onnx)
  int8_closeups  calibrated on 100 training close-ups + 100 training scenes; layers kept in full precision chosen on
                 300 validation pictures (scripts/quantize_int8.py, minmax)
For each, the announce threshold is chosen by the open-set rule of eval_open_set.py on its validation pool (all NSTU
validation photos as known notes, 600 demonetized notes and 300 coins as unknown; maximise correct - wrong - unknown
announced). The version with the higher validation utility is the Pi model. Then, once: NSTU test (one image per
original) and Bangla Money (1,536 known, 101 one-taka unknown), at its threshold.

    python scripts/eval/eval_handheld_int8.py
Output: results/closeup_ft/handheld_int8.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from ultralytics import YOLO  # noqa: E402

import eval_closeup_ft as E  # noqa: E402
from scripts.eval import eval_open_set as OS  # noqa: E402

OUT = ROOT / "results" / "closeup_ft" / "handheld_int8.json"
MODELS = {"int8_scene": "models/best_handheld_int8.onnx", "int8_closeups": "models/best_handheld_int8_minmax.onnx"}


def pool():
    rng = random.Random(42)
    val = []
    for folder in sorted(p for p in (E.NSTU / "validation").iterdir() if p.is_dir()):
        val += [(p, folder.name) for p in OS.imgs(folder)]
    coin_root = E.EXT / "Large Scale BDT DB 2026 Kaggle" / "bd_coin"
    for name, n in (("demonetized_note", 600), ("1_Taka", 100), ("2_Taka", 100), ("5_Taka", 100)):
        files = OS.imgs(coin_root / name)
        rng.shuffle(files)
        val += [(p, None) for p in files[:n]]
    return val


def main():
    res = json.loads(OUT.read_text()) if OUT.exists() else {"date": "2026-10-15", "validation": {}}
    val = pool()
    for name, path in MODELS.items():
        if name in res["validation"]:
            continue
        rows = OS.score_items(YOLO(str(ROOT / path), task="detect"), val)
        sweep = [OS.tally(rows, t) for t in OS.GRID]
        best = max(sweep, key=lambda r: (r["utility_correct_minus_wrong"], -r["tau"]))
        res["validation"][name] = {"model": path, "tau": best["tau"], "at_tau": best}
        OUT.write_text(json.dumps(res, indent=1))
        print(name, best["tau"], best["utility_correct_minus_wrong"], flush=True)
    chosen = max(res["validation"], key=lambda n: res["validation"][n]["at_tau"]["utility_correct_minus_wrong"])
    res["chosen"] = chosen
    if "test" not in res:
        v = res["validation"][chosen]
        model = YOLO(str(ROOT / v["model"]), task="detect")
        res["test"] = {"model": v["model"], "tau": v["tau"], "nstu_test": E.tally(E.predict(model, E.nstu("test")), v["tau"]),
                       "bangla_money": E.tally(E.predict(model, E.bangla_money()), v["tau"])}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("chosen", "test")}, indent=1))


if __name__ == "__main__":
    main()
