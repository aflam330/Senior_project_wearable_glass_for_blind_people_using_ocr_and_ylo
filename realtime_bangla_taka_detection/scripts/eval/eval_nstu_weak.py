"""Choose and test the hand-held detector re-trained with NSTU train photos (scripts/dataset/make_nstu_weak.py,
scripts/train/train_nstu_weak.py) against the deployed hand-held detector (models/best_handheld.pt).

Rule, fixed before any candidate was scored (2026-11-23), the same as eval_closeup_ft.py:
  Candidates: the deployed hand-held detector and every epoch checkpoint of runs/nstu_weak/seed42.
  Constraint: synthetic VALIDATION mAP@0.5 at least the deployed one's minus 0.005 (whole notes in scenes kept).
  Threshold and score, VALIDATION only: the pool of eval_open_set.py (all NSTU validation photos as known notes,
    600 demonetised notes and 300 coins as unknown, seed 42); for each candidate the announce threshold on the grid
    0.25-0.95 that maximises correct - wrong - unknown announced, and that utility is its score.
  The admissible candidate with the highest score is chosen (the deployed one on a tie).
  Then, once, for the deployed and the chosen detector at their own thresholds: NSTU Recognition test (one image per
    original; NSTU test was not read before), Bangla Money (1,536 known notes, 101 one-taka notes as unknown) and the
    synthetic test split (mAP). Model size and laptop GPU latency are recorded; Pi 5 latency needs the Pi.

    python scripts/eval/eval_nstu_weak.py select
    python scripts/eval/eval_nstu_weak.py test
Output: results/nstu_weak/eval.json
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval import eval_closeup_ft as CF  # noqa: E402
from scripts.eval import eval_open_set as OS  # noqa: E402

RUN = ROOT / "runs" / "nstu_weak" / "seed42"
RES = ROOT / "results" / "nstu_weak" / "eval.json"
DEPLOYED = ROOT / "models" / "best_handheld.pt"


def candidates():
    c = [("deployed_handheld", DEPLOYED)]
    for p in sorted((RUN / "weights").glob("epoch*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0])):
        c.append((p.stem, p))
    if (RUN / "weights" / "last.pt").exists():
        c.append(("last", RUN / "weights" / "last.pt"))
    return c


def val_pool():
    rng = random.Random(42)                                # the same draw as eval_open_set.py
    val = []
    for folder in sorted(p for p in (CF.NSTU / "validation").iterdir() if p.is_dir()):
        val += [(p, folder.name) for p in OS.imgs(folder)]
    coin_root = CF.EXT / "Large Scale BDT DB 2026 Kaggle" / "bd_coin"
    for name, n in (("demonetized_note", 600), ("1_Taka", 100), ("2_Taka", 100), ("5_Taka", 100)):
        files = OS.imgs(coin_root / name)
        rng.shuffle(files)
        val += [(p, None) for p in files[:n]]
    return val


def main():
    from ultralytics import YOLO
    RES.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(RES.read_text(encoding="utf-8")) if RES.exists() else {"date": "2026-11-23", "validation": {}}
    if sys.argv[1] == "select":
        pool = val_pool()
        for name, w in candidates():
            if name in res["validation"]:
                continue
            m50, m5095 = CF.val_map(w)
            rows = OS.score_items(YOLO(str(w)), pool)
            sweep = [OS.tally(rows, t) for t in OS.GRID]
            best = max(sweep, key=lambda r: (r["utility_correct_minus_wrong"], -r["tau"]))
            res["validation"][name] = {"weights": str(Path(w).relative_to(ROOT)), "synthetic_val_map50": m50,
                                       "synthetic_val_map50_95": m5095, "tau": best["tau"], "at_tau": best}
            RES.write_text(json.dumps(res, indent=1), encoding="utf-8")
            print(name, m50, best, flush=True)
        V = res["validation"]
        base = V["deployed_handheld"]["synthetic_val_map50"]
        ok = {n: v for n, v in V.items() if v["synthetic_val_map50"] >= base - 0.005}
        res["chosen"] = max(ok, key=lambda n: (ok[n]["at_tau"]["utility_correct_minus_wrong"], n == "deployed_handheld"))
        res["admissible"] = sorted(ok)
        RES.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("chosen:", res["chosen"])
    elif sys.argv[1] == "test":
        res.setdefault("test", {})
        for name in dict.fromkeys(["deployed_handheld", res["chosen"]]):
            if name in res["test"]:
                continue
            v = res["validation"][name]
            w = ROOT / v["weights"]
            model = YOLO(str(w))
            nt = CF.predict(model, CF.nstu("test"))
            t = {"tau": v["tau"], "nstu_test": CF.tally(nt, v["tau"])}
            bm = CF.predict(model, CF.bangla_money())
            t["bangla_money"] = CF.tally(bm, v["tau"])
            m = YOLO(str(w)).val(data=str(ROOT / "data" / "data.yaml"), split="test", imgsz=640, batch=16, verbose=False,
                                 plots=False, device=CF.DEV, workers=2)
            t["synthetic_test_map50"], t["synthetic_test_map50_95"] = round(float(m.box.map50), 4), round(float(m.box.map), 4)
            t["model_size_mb"] = round(w.stat().st_size / 1e6, 1)
            import cv2
            img = cv2.imread(str(CF.nstu("test")[0][0]))
            for _ in range(3):
                model.predict(img, imgsz=640, verbose=False, device=CF.DEV)
            ts = []
            for _ in range(30):
                t0 = time.perf_counter()
                model.predict(img, imgsz=640, verbose=False, device=CF.DEV)
                ts.append((time.perf_counter() - t0) * 1000)
            t["laptop_latency_ms_median"] = round(sorted(ts)[15], 1)
            res["test"][name] = t
            RES.write_text(json.dumps(res, indent=1), encoding="utf-8")
            print(name, json.dumps(t), flush=True)


if __name__ == "__main__":
    main()
