"""Two note detectors together: the earlier hand-held detector (good on whole notes) and the NSTU re-trained one (good
on hand-held close-ups). Rule fixed on 2026-11-23, before any test number of a cascade was seen:

  Device configurations (each model at the threshold already chosen for it on validation):
    laptop  H = best_handheld.pt 0.75,       N = best_nstu.pt 0.85
    pi      H = best_handheld.pt 0.75 (the Pi default since 29 October), N = best_nstu_int8.onnx 0.90
  Candidates per device: H only, N only, H then N (N is asked only when H is not sure), N then H.
  Validation: (1) the open-set pool of eval_nstu_weak.py (NSTU validation photos + 900 unknown demonetised notes and
  coins), utility = correct - wrong - unknown announced; (2) whole notes: 600 synthetic VALIDATION scenes
  (currency_yolo_data/valid, seed 0), the largest labelled note is the truth, correct rate.
  Choice: the candidate with the highest (1) among those whose (2) is at least H's minus 0.005.
  Then, once: NSTU test (one image per original), Bangla Money (1,536 known + 101 unknown) and, as an extra check that
  is NOT independent (its notes were the templates of the synthetic training set), 225 whole flat BanglaTaka photos.

    python scripts/eval/eval_detector_cascade.py
Output: results/nstu_weak/cascade.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval import eval_closeup_ft as CF  # noqa: E402
from scripts.eval.eval_nstu_weak import val_pool  # noqa: E402

OUT = ROOT / "results" / "nstu_weak" / "cascade.json"
CACHE = ROOT / "results" / "nstu_weak" / "cascade_cache.json"
DS = ROOT.parent / "data set"
MODELS = {"handheld": ("models/best_handheld.pt", 0.75), "nstu": ("models/best_nstu.pt", 0.85),
          "nstu_int8": ("models/best_nstu_int8.onnx", 0.90)}
DEVICES = {"laptop": ("handheld", "nstu"), "pi": ("handheld", "nstu_int8")}
NAMES = ["2_taka", "5_taka", "10_taka", "20_taka", "50_taka", "100_taka", "200_taka", "500_taka", "1000_taka"]


def synthetic_val(n=600):
    imgs = sorted((DS / "currency_yolo_data" / "valid" / "images").glob("*.jpg"))
    random.Random(0).shuffle(imgs)
    out = []
    for p in imgs:
        lab = DS / "currency_yolo_data" / "valid" / "labels" / f"{p.stem}.txt"
        rows = [l.split() for l in lab.read_text().splitlines() if l.strip()] if lab.exists() else []
        if not rows:
            continue
        big = max(rows, key=lambda r: float(r[3]) * float(r[4]))
        out.append((p, NAMES[int(big[0])]))
        if len(out) == n:
            break
    return out


def flat_notes():
    R = DS / "Bangladeshi_Paper_Currency_Raw" / "Bangladeshi_Paper_Currency_Raw"
    items = []
    for d in sorted(R.iterdir()):
        if d.is_dir():
            fs = sorted(d.glob("*.jpg"))
            random.Random(0).shuffle(fs)
            items += [(p, f"{d.name}_taka") for p in fs[:25]]
    return items


def top(model, items, device):
    out = []
    for p, truth in items:
        r = model.predict(cv2.imread(str(p)), conf=0.05, imgsz=640, verbose=False, device=device)[0]
        if r.boxes is None or not len(r.boxes):
            out.append([truth, None, 0.0])
        else:
            i = int(r.boxes.conf.argmax())
            out.append([truth, r.names[int(r.boxes.cls[i])], float(r.boxes.conf[i])])
    return out


def answer(rows_a, tau_a, rows_b=None, tau_b=None):
    """Cascade: A's answer if A is sure, else B's if B is sure, else no answer. Returns [(truth, answer or None)]."""
    out = []
    for i, (t, n, c) in enumerate(rows_a):
        if n is not None and c >= tau_a:
            out.append((t, n))
        elif rows_b is not None and rows_b[i][1] is not None and rows_b[i][2] >= tau_b:
            out.append((t, rows_b[i][1]))
        else:
            out.append((t, None))
    return out


def tally(ans):
    known = [(t, a) for t, a in ans if t is not None]
    unk = [(t, a) for t, a in ans if t is None]
    ok = sum(a == t for t, a in known)
    wrong = sum(a is not None and a != t for t, a in known)
    r = {"known": len(known), "correct": ok, "wrong": wrong, "no_answer": len(known) - ok - wrong,
         "correct_rate": round(ok / max(len(known), 1), 4)}
    if unk:
        u = sum(a is not None for _, a in unk)
        r.update(unknown=len(unk), unknown_announced=u, utility=ok - wrong - u)
    return r


def main():
    from ultralytics import YOLO
    sets = {"val_pool": val_pool(), "synthetic_val": synthetic_val(), "nstu_test": CF.nstu("test"),
            "bangla_money": CF.bangla_money(), "flat_notes": flat_notes()}
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    for m, (w, _) in MODELS.items():
        for s, items in sets.items():
            key = f"{m}|{s}"
            if key in cache:
                continue
            dev = "cpu" if w.endswith(".onnx") else None
            cache[key] = top(YOLO(str(ROOT / w), task="detect"), items, dev)
            CACHE.write_text(json.dumps(cache))
            print("scored", key, flush=True)
    res = {"rule": __doc__.split("Choice:")[1].split("Then")[0].strip(), "devices": {}}
    for dev, (h, n) in DEVICES.items():
        th, tn = MODELS[h][1], MODELS[n][1]
        cands = {"H only": (h, th, None, None), "N only": (n, tn, None, None), "H then N": (h, th, n, tn), "N then H": (n, tn, h, th)}

        def run(c, s):
            a, ta, b, tb = cands[c]
            return tally(answer(cache[f"{a}|{s}"], ta, cache[f"{b}|{s}"] if b else None, tb))
        val = {c: {"val_pool": run(c, "val_pool"), "synthetic_val": run(c, "synthetic_val")} for c in cands}
        floor = val["H only"]["synthetic_val"]["correct_rate"] - 0.005
        ok = [c for c in cands if val[c]["synthetic_val"]["correct_rate"] >= floor]
        chosen = max(ok, key=lambda c: (val[c]["val_pool"]["utility"], c == "H only"))
        test = {c: {s: run(c, s) for s in ("nstu_test", "bangla_money", "flat_notes")} for c in dict.fromkeys(["H only", chosen])}
        res["devices"][dev] = {"models": {"H": [h, th], "N": [n, tn]}, "validation": val, "admissible": ok,
                               "chosen": chosen, "test": test}
        print(dev, "chosen", chosen, json.dumps(test), flush=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
