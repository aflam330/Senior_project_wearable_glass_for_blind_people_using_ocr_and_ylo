"""Score the glass's deployed jaal (counterfeit) verdict on whole-note photographs.

Runs savior_glass CurrencyMode.detect_live exactly as the app does (Taka YOLO, then PRMVT on
the top box crop as one view, genuine if p >= 0.5) and records the verdict per image.

Sets (none was used to train or select PRMVT):
  counterfeit_ds  data set/Bangladeshi Counterfeit Currency Image Dataset (500/1000 BDT, genuine + counterfeit)
  bangla_money    data set for comparison/.../bangla_banknote_v2/Training (genuine, 8 denominations)
  banglataka      data set/Bangladeshi_Paper_Currency_Raw, 50 per class, seed 42 (genuine)

Output: results/jaal_deployed/rows.json (one row per image) and summary.json.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[3]
GLASS = ROOT / "savior_glass"
sys.path.insert(0, str(GLASS))
sys.path.insert(0, str(ROOT / "realtime_bangla_taka_detection"))

from modes.currency_mode import CurrencyMode  # noqa: E402

OUT = ROOT / "realtime_bangla_taka_detection" / "results" / "jaal_deployed"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def collect() -> list[tuple[str, Path, str]]:
    items = []
    cf = ROOT / "data set" / "Bangladeshi Counterfeit Currency Image Dataset" / "Denomination-wise Distribution"
    for folder in sorted(p for p in cf.iterdir() if p.is_dir()):
        truth = "counterfeit" if "Counterfeit" in folder.name else "genuine"
        items += [("counterfeit_ds", p, truth) for p in images(folder)]
    bm = ROOT / "data set for comparison" / "Bangla Money dataset Kaggle" / "bangla_banknote_v2" / "Training"
    for folder in sorted(p for p in bm.iterdir() if p.is_dir()):
        if folder.name == "1":  # 1 taka is outside the detector's classes
            continue
        items += [("bangla_money", p, "genuine") for p in images(folder)]
    raw = ROOT / "data set" / "Bangladeshi_Paper_Currency_Raw" / "Bangladeshi_Paper_Currency_Raw"
    rng = random.Random(42)
    for folder in sorted(p for p in raw.iterdir() if p.is_dir()):
        files = images(folder)
        rng.shuffle(files)
        items += [("banglataka", p, "genuine") for p in files[:50]]
    return items


def summarize(rows: list[dict]) -> dict:
    out = {}
    for ds in sorted({r["dataset"] for r in rows}):
        for truth in ("genuine", "counterfeit"):
            rs = [r for r in rows if r["dataset"] == ds and r["truth"] == truth]
            if not rs:
                continue
            judged = [r for r in rs if r["verdict"] in ("genuine", "counterfeit")]
            wrong = [r for r in judged if r["verdict"] != truth]
            out[f"{ds}/{truth}"] = {
                "n_images": len(rs),
                "no_note_detected": sum(r["verdict"] == "none" for r in rs),
                "no_verdict": sum(r["verdict"] == "unknown" for r in rs),
                "judged": len(judged),
                "wrong_verdict": len(wrong),
                "wrong_verdict_rate_of_judged": len(wrong) / len(judged) if judged else None,
                "wrong_verdict_wilson95": wilson(len(wrong), len(judged)),
            }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mode = CurrencyMode()
    mode.verdict_enabled = True  # score the verdict even though the app ships with it off
    mode._load_yolo()
    mode._load_auth()
    if mode._auth_kind != "qduig":
        raise SystemExit(f"deployed PRMVT not loaded (got {mode._auth_kind})")
    items = collect()
    print("images", len(items), flush=True)
    rows = []
    t0 = time.time()
    for i, (ds, path, truth) in enumerate(items):
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        hits = mode.detect_live(bgr)
        if not hits:
            verdict, prob, name, conf = "none", None, None, None
        else:
            h = hits[0]
            verdict, prob, name, conf = h["auth"], h["genuine_prob"], h["name"], h["conf"]
        rows.append({
            "dataset": ds, "path": str(path.relative_to(ROOT)), "truth": truth,
            "verdict": verdict, "genuine_prob": prob, "yolo_class": name, "yolo_conf": conf,
        })
        if (i + 1) % 200 == 0:
            print(i + 1, f"{time.time() - t0:.0f}s", flush=True)
    (OUT / "rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    summary = {
        "pipeline": "savior_glass CurrencyMode.detect_live; PRMVT prefix_ft seed42 on top YOLO crop, 1 view, threshold 0.5",
        "note": "No image here was used to train or select PRMVT. Photos are not grouped by physical note.",
        "sets": summarize(rows),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
