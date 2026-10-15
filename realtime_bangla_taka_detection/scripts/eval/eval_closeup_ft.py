"""Choose and test the detector fine-tuned with hand-held close-ups (scripts/dataset/make_closeups.py).

Rule, fixed before training finished (2026-10-14):
  Candidates: the deployed detector (models/best.pt) and every epoch checkpoint of runs/closeup_ft/seed42.
  Constraint: mAP@0.5 on the synthetic VALIDATION split (currency_yolo_data/valid) at least the deployed detector's
    minus 0.005, so whole notes in scenes are not given up.
  Score: on NSTU-BDTAKA Recognition/VALIDATION (hand-held close-ups, one image per original photo), the glass's own
    announce rule (top box, confidence >= 0.60, CURRENCY_ANNOUNCE_CONF): correct answers minus wrong answers.
  The best-scoring candidate that meets the constraint is chosen. Then, once:
    NSTU Recognition/test (one image per original), Bangla Money (1,536 known notes, 101 one-taka notes as unknown),
    and the synthetic test split (mAP), for the chosen model and the deployed one.

Added before any test read (2026-10-15), after the epoch scores showed that the fine-tuned detector also names many
more WRONG values on hand-held photos: the announce threshold is re-chosen for the chosen model by the same rule and
the same validation pool that set 0.60 for the deployed one (scripts/eval/eval_open_set.py: all NSTU validation
photos as known notes, 600 demonetized notes and 300 coins as unknown, grid 0.25-0.95, maximise correct - wrong
- unknown announced). The test then reports the chosen model at that threshold and at 0.60.

    python scripts/eval/eval_closeup_ft.py select
    python scripts/eval/eval_closeup_ft.py tau
    python scripts/eval/eval_closeup_ft.py test
Output: results/closeup_ft/closeup_ft.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT.parent / "data set for comparison"
NSTU = EXT / "NSTU-BDTAKA dataset Mendeley" / "data" / "Recognition"
BM = EXT / "Bangla Money dataset Kaggle" / "bangla_banknote_v2" / "Training"
RUN = ROOT / "runs" / "detect" / "runs" / "closeup_ft" / "seed42"   
OUT = ROOT / "results" / "closeup_ft"
OUT.mkdir(parents=True, exist_ok=True)
RES = OUT / "closeup_ft.json"
TAU = 0.60
IMG = {".jpg", ".jpeg", ".png"}
DEV = os.environ.get("EVAL_DEVICE") or None          # "cpu" while the GPU is training


def oid(name: str) -> str:
    return re.split(r"_(?:jpg|jpeg|png)\.rf\.", name)[0]


def nstu(split: str):
    items = {}
    for folder in sorted(p for p in (NSTU / split).iterdir() if p.is_dir()):
        truth = folder.name                                   # e.g. "500_taka"
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in IMG:
                items.setdefault((truth, oid(p.name)), (p, truth))
    return list(items.values())


def bangla_money():
    out = []
    for folder in sorted(p for p in BM.iterdir() if p.is_dir()):
        truth = None if folder.name == "1" else f"{folder.name}_taka"
        out += [(p, truth) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMG]
    return out


def predict(model, items):
    rows = []
    for path, truth in items:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        r = model.predict(bgr, conf=0.05, imgsz=640, verbose=False, device=DEV)[0]
        if r.boxes is None or len(r.boxes) == 0:
            rows.append((truth, None, 0.0))
        else:
            i = int(r.boxes.conf.argmax())
            rows.append((truth, r.names[int(r.boxes.cls[i])], float(r.boxes.conf[i])))
    return rows


def tally(rows, tau):
    known = [r for r in rows if r[0] is not None]
    unknown = [r for r in rows if r[0] is None]
    ans = [r for r in known if r[1] is not None and r[2] >= tau]
    ok = sum(r[1] == r[0] for r in ans)
    out = {"known": len(known), "correct": ok, "wrong": len(ans) - ok, "no_answer": len(known) - len(ans),
           "correct_rate": round(ok / max(len(known), 1), 4), "correct_minus_wrong": ok - (len(ans) - ok),
           "found_any_box": round(sum(r[1] is not None for r in known) / max(len(known), 1), 4)}
    if unknown:
        u = sum(r[1] is not None and r[2] >= tau for r in unknown)
        out.update({"unknown": len(unknown), "unknown_announced": u, "unknown_announced_rate": round(u / len(unknown), 4)})
    return out


def val_map(weights):
    from ultralytics import YOLO
    m = YOLO(str(weights)).val(data=str(ROOT / "data" / "data.yaml"), split="val", imgsz=640, batch=16, verbose=False, plots=False, device=DEV, workers=2)
    return round(float(m.box.map50), 4), round(float(m.box.map), 4)


def candidates():
    c = [("deployed", ROOT / "models" / "best.pt")]
    for p in sorted((RUN / "weights").glob("epoch*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0])):
        c.append((p.stem, p))
    if (RUN / "weights" / "last.pt").exists():
        c.append(("last", RUN / "weights" / "last.pt"))
    return c


def main():
    from ultralytics import YOLO
    res = json.loads(RES.read_text(encoding="utf-8")) if RES.exists() else {"date": "2026-10-14", "tau": TAU, "validation": {}}
    if sys.argv[1] == "select":
        items = nstu("validation")
        for name, w in candidates():
            if name in res["validation"]:
                continue
            m50, m5095 = val_map(w)
            rows = predict(YOLO(str(w)), items)
            res["validation"][name] = {"weights": str(w.relative_to(ROOT)), "synthetic_val_map50": m50, "synthetic_val_map50_95": m5095,
                                       "nstu_val": tally(rows, TAU), "nstu_val_at_0.25": tally(rows, 0.25)}
            RES.write_text(json.dumps(res, indent=1), encoding="utf-8")
            print(name, m50, res["validation"][name]["nstu_val"], flush=True)
        base = res["validation"]["deployed"]["synthetic_val_map50"]
        ok = {n: v for n, v in res["validation"].items() if v["synthetic_val_map50"] >= base - 0.005}
        best = max(ok, key=lambda n: (ok[n]["nstu_val"]["correct_minus_wrong"], n == "deployed"))
        res["chosen"] = best
        RES.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print("chosen:", best)
    elif sys.argv[1] == "tau":
        import random
        sys.path.insert(0, str(ROOT))
        from scripts.eval import eval_open_set as OS
        rng = random.Random(42)                                # the same draw as eval_open_set.py
        val = []
        for folder in sorted(p for p in (NSTU / "validation").iterdir() if p.is_dir()):
            val += [(p, folder.name) for p in OS.imgs(folder)]
        coin_root = EXT / "Large Scale BDT DB 2026 Kaggle" / "bd_coin"
        for name, n in (("demonetized_note", 600), ("1_Taka", 100), ("2_Taka", 100), ("5_Taka", 100)):
            files = OS.imgs(coin_root / name)
            rng.shuffle(files)
            val += [(p, None) for p in files[:n]]
        out = {}
        for name in dict.fromkeys(["deployed", res["chosen"]]):
            rows = OS.score_items(YOLO(str(ROOT / res["validation"][name]["weights"])), val)
            sweep = [OS.tally(rows, t) for t in OS.GRID]
            best = max(sweep, key=lambda r: (r["utility_correct_minus_wrong"], -r["tau"]))
            out[name] = {"tau": best["tau"], "at_tau": best, "at_0.60": OS.tally(rows, 0.60)}
            print(name, "tau", best["tau"], best, flush=True)
        res["tau"] = out
        RES.write_text(json.dumps(res, indent=1), encoding="utf-8")
    elif sys.argv[1] == "test":
        chosen = res["chosen"]
        res.setdefault("test", {})
        for name in dict.fromkeys(["deployed", chosen]):
            if name in res["test"]:
                continue
            w = ROOT / res["validation"][name]["weights"]
            model = YOLO(str(w))
            tau = res.get("tau", {}).get(name, {}).get("tau", TAU)
            nt = predict(model, nstu("test"))
            t = {"tau": tau, "nstu_test": tally(nt, tau), "nstu_test_at_0.60": tally(nt, TAU)}
            bm = predict(model, bangla_money())
            t["bangla_money"] = tally(bm, tau)
            t["bangla_money_at_0.60"] = tally(bm, TAU)
            t["bangla_money_at_0.25"] = tally(bm, 0.25)
            m = YOLO(str(w)).val(data=str(ROOT / "data" / "data.yaml"), split="test", imgsz=640, batch=16, verbose=False, plots=False, device=DEV, workers=2)
            t["synthetic_test_map50"], t["synthetic_test_map50_95"] = round(float(m.box.map50), 4), round(float(m.box.map), 4)
            res["test"][name] = t
            RES.write_text(json.dumps(res, indent=1), encoding="utf-8")
            print(name, json.dumps(t), flush=True)


if __name__ == "__main__":
    main()
