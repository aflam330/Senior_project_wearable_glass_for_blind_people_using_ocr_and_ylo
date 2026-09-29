"""Unknown-note rejection for the Taka detector: confidence threshold chosen on validation only.

Rule (fixed before scoring test):
  score  = confidence of the top YOLO box (predict at conf 0.05)
  answer = announce the top box's class if score >= tau, else say nothing / "not sure"
  tau    = grid 0.25..0.95 step 0.05, maximising (correct - wrong) on the validation pool,
           where any announcement for an unknown image counts as wrong.
Validation pool (never used to train or select the detector):
  known   NSTU-BDTAKA Recognition/validation (9 denominations)
  unknown Large Scale BDT DB demonetized_note (600 random) + 1/2/5-Taka coins (100 each)
Test (scored once with the chosen tau):
  known   Bangla Money Training folders, 8 denominations (1,536)
  unknown Bangla Money "1" folder (1-taka notes, not a detector class; 101)

Output: results/open_set/{rows_val,rows_test,summary}.json
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT.parent / "data set for comparison"
OUT = ROOT / "results" / "open_set"
IMG = {".jpg", ".jpeg", ".png", ".bmp"}
GRID = [round(0.25 + 0.05 * i, 2) for i in range(15)]


def imgs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG)


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def score_items(model: YOLO, items: list[tuple[Path, str | None]]) -> list[dict]:
    rows = []
    for path, truth in items:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        r = model.predict(bgr, conf=0.05, verbose=False, imgsz=640)[0]
        if r.boxes is None or len(r.boxes) == 0:
            top, conf = None, 0.0
        else:
            i = int(r.boxes.conf.argmax())
            top, conf = r.names[int(r.boxes.cls[i])], float(r.boxes.conf[i])
        rows.append({"path": str(path.relative_to(ROOT.parent)), "truth": truth, "pred": top, "conf": conf})
    return rows


def tally(rows: list[dict], tau: float) -> dict:
    known = [r for r in rows if r["truth"] is not None]
    unknown = [r for r in rows if r["truth"] is None]
    k_ans = [r for r in known if r["pred"] is not None and r["conf"] >= tau]
    k_ok = sum(r["pred"] == r["truth"] for r in k_ans)
    u_ans = sum(r["pred"] is not None and r["conf"] >= tau for r in unknown)
    return {
        "tau": tau,
        "known_n": len(known), "known_correct": k_ok, "known_wrong": len(k_ans) - k_ok,
        "known_abstain": len(known) - len(k_ans),
        "known_correct_rate": k_ok / len(known) if known else None,
        "known_wrong_rate_of_answered": (len(k_ans) - k_ok) / len(k_ans) if k_ans else None,
        "unknown_n": len(unknown), "unknown_announced": u_ans,
        "unknown_announced_rate": u_ans / len(unknown) if unknown else None,
        "unknown_announced_wilson95": wilson(u_ans, len(unknown)),
        "utility_correct_minus_wrong": k_ok - (len(k_ans) - k_ok) - u_ans,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(ROOT / "models" / "best.pt"))
    rng = random.Random(42)

    val = []
    nstu = EXT / "NSTU-BDTAKA dataset Mendeley" / "data" / "Recognition" / "validation"
    for folder in sorted(p for p in nstu.iterdir() if p.is_dir()):
        val += [(p, folder.name) for p in imgs(folder)]  # folder names are "<n>_taka"
    coin_root = EXT / "Large Scale BDT DB 2026 Kaggle" / "bd_coin"
    for name, n in (("demonetized_note", 600), ("1_Taka", 100), ("2_Taka", 100), ("5_Taka", 100)):
        files = imgs(coin_root / name)
        rng.shuffle(files)
        val += [(p, None) for p in files[:n]]

    test = []
    bm = EXT / "Bangla Money dataset Kaggle" / "bangla_banknote_v2" / "Training"
    for folder in sorted(p for p in bm.iterdir() if p.is_dir()):
        truth = None if folder.name == "1" else f"{folder.name}_taka"
        test += [(p, truth) for p in imgs(folder)]

    rows_val = score_items(model, val)
    (OUT / "rows_val.json").write_text(json.dumps(rows_val, indent=1), encoding="utf-8")
    sweep = [tally(rows_val, t) for t in GRID]
    best = max(sweep, key=lambda s: (s["utility_correct_minus_wrong"], -s["tau"]))
    tau = best["tau"]
    print("chosen tau on validation:", tau, flush=True)

    rows_test = score_items(model, test)  # scored once, after tau is fixed
    (OUT / "rows_test.json").write_text(json.dumps(rows_test, indent=1), encoding="utf-8")
    summary = {
        "rule": __doc__.split("Validation pool")[0].strip(),
        "val_sweep": sweep,
        "chosen_tau": tau,
        "test_at_app_default_0.25": tally(rows_test, 0.25),
        "test_at_chosen_tau": tally(rows_test, tau),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("chosen_tau", "test_at_app_default_0.25", "test_at_chosen_tau")}, indent=2))


if __name__ == "__main__":
    main()
