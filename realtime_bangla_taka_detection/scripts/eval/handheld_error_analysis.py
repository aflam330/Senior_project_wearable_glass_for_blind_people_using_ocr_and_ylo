"""Error analysis of the hand-held Taka detector on NSTU-BDTAKA Recognition (test locked: this script only READS it
with an already-chosen model and threshold; nothing here selects anything on test).

For a detector (default models/best_handheld.pt, announce threshold 0.75 as on the glass):
  * class-mapping check: model.names against the NSTU folder names and data/data.yaml
  * confusion matrix (truth x answer, plus "no answer"), one image per original photo (as in eval_closeup_ft.py)
  * per-denomination recall, precision when answering, F1, and abstention rate
  * error factors: brightness, sharpness, portrait/landscape, how much of the frame the box fills, whether the box
    touches the frame edge (note cut off), and denomination confusions; error rate per factor bin
  * the 88 test originals that also occur in NSTU train are reported separately (they are not removed silently)
  * a montage of wrong answers and of abstentions

    python scripts/eval/handheld_error_analysis.py [--weights models/best_handheld.pt] [--tau 0.75] [--split test]
Output: results/handheld_analysis/<weights-stem>_<split>.json, confusion .csv/.png, failures_*.jpg
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NSTU = ROOT.parent / "data set for comparison" / "NSTU-BDTAKA dataset Mendeley" / "data" / "Recognition"
OUT = ROOT / "results" / "handheld_analysis"
IMG = {".jpg", ".jpeg", ".png"}
ORDER = ["2_taka", "5_taka", "10_taka", "20_taka", "50_taka", "100_taka", "200_taka", "500_taka", "1000_taka"]


def oid(name: str) -> str:
    return re.split(r"_(?:jpg|jpeg|png)\.rf\.", name)[0]


def nstu(split: str):
    items = {}
    for folder in sorted(p for p in (NSTU / split).iterdir() if p.is_dir()):
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in IMG:
                items.setdefault((folder.name, oid(p.name)), (p, folder.name))
    return items


def prf(cm, labels):
    out = {}
    for c in labels:
        tp = cm[c][c]
        n_true = sum(cm[c].values())
        answered_as_c = sum(cm[t][c] for t in cm)
        abst = cm[c]["no_answer"]
        rec = tp / n_true if n_true else None
        prec = tp / answered_as_c if answered_as_c else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else 0.0
        out[c] = {"n": n_true, "correct": tp, "recall": rec, "precision_when_answering": prec, "f1": f1,
                  "abstain_rate": abst / n_true if n_true else None,
                  "wrong": n_true - tp - abst}
    return out


def bins(rows, key, edges, labels):
    res = {}
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        rs = [r for r in rows if lo <= r[key] < hi]
        if rs:
            res[lab] = {"n": len(rs), "correct": sum(r["outcome"] == "correct" for r in rs) / len(rs),
                        "wrong": sum(r["outcome"] == "wrong" for r in rs) / len(rs),
                        "no_answer": sum(r["outcome"] == "no_answer" for r in rs) / len(rs)}
    return res


def montage(rows, path, title_fn, n=24):
    tiles = []
    for r in rows[:n]:
        im = cv2.imread(str(r["path"]))
        h, w = im.shape[:2]
        s = 220 / max(h, w)
        im = cv2.resize(im, (int(w * s), int(h * s)))
        if r.get("box"):
            x1, y1, x2, y2 = (int(v * s) for v in r["box"])
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
        tile = np.full((260, 230, 3), 255, np.uint8)
        tile[:im.shape[0], :im.shape[1]] = im
        cv2.putText(tile, title_fn(r), (3, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return
    while len(tiles) % 6:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
    cv2.imwrite(str(path), grid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "best_handheld.pt"))
    ap.add_argument("--tau", type=float, default=0.75)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(a.weights)
    names = {int(k): v for k, v in model.names.items()}
    folders = sorted(p.name for p in (NSTU / a.split).iterdir() if p.is_dir())
    mapping = {"model_names": names, "nstu_folders": folders,
               "every_folder_is_a_model_class": all(f in names.values() for f in folders),
               "model_classes_not_in_nstu": sorted(set(names.values()) - set(folders))}
    items = nstu(a.split)
    train_ids = set(nstu("train")) if a.split != "train" else set()      # (denomination, original id)

    rows = []
    for (truth, o), (p, _) in items.items():
        bgr = cv2.imread(str(p))
        r = model.predict(bgr, conf=0.05, imgsz=640, verbose=False, device=a.device)[0]
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        row = {"path": str(p), "truth": truth, "original": o, "in_train_too": (truth, o) in train_ids,
               "brightness": float(gray.mean()), "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
               "portrait": int(h > w), "box": None, "pred": None, "conf": 0.0, "box_fill": 0.0, "touches_edge": 0}
        if r.boxes is not None and len(r.boxes):
            i = int(r.boxes.conf.argmax())
            x1, y1, x2, y2 = (float(v) for v in r.boxes.xyxy[i].tolist())
            row.update(box=[x1, y1, x2, y2], pred=names[int(r.boxes.cls[i])], conf=float(r.boxes.conf[i]),
                       box_fill=(x2 - x1) * (y2 - y1) / (w * h),
                       touches_edge=int(min(x1, y1) < 3 or x2 > w - 3 or y2 > h - 3))
        ans = row["pred"] if row["conf"] >= a.tau else None
        row["answer"] = ans or "no_answer"
        row["outcome"] = "no_answer" if ans is None else ("correct" if ans == truth else "wrong")
        rows.append(row)

    def summarise(rs):
        cm = {t: Counter() for t in ORDER if any(r["truth"] == t for r in rs)}
        for r in rs:
            cm[r["truth"]][r["answer"]] += 1
        oc = Counter(r["outcome"] for r in rs)
        return cm, {"n": len(rs), "correct": oc["correct"], "wrong": oc["wrong"], "no_answer": oc["no_answer"],
                    "correct_rate": oc["correct"] / len(rs),
                    "precision_when_answering": oc["correct"] / max(oc["correct"] + oc["wrong"], 1),
                    "found_any_box": sum(r["box"] is not None for r in rs) / len(rs)}

    cm, overall = summarise(rows)
    _, overall_clean = summarise([r for r in rows if not r["in_train_too"]])
    per_class = prf(cm, list(cm))
    wrong_pairs = Counter((r["truth"], r["answer"]) for r in rows if r["outcome"] == "wrong").most_common(12)
    factors = {
        "brightness": bins(rows, "brightness", [0, 80, 120, 160, 256], ["<80", "80-120", "120-160", ">=160"]),
        "sharpness": bins(rows, "sharpness", [0, 50, 150, 500, 1e12], ["<50", "50-150", "150-500", ">=500"]),
        "orientation": bins(rows, "portrait", [0, 1, 2], ["landscape", "portrait"]),
        "box_fill": bins(rows, "box_fill", [0, 0.3, 0.6, 0.85, 1.01], ["<30%", "30-60%", "60-85%", ">=85%"]),
        "box_touches_edge": bins(rows, "touches_edge", [0, 1, 2], ["inside", "touches edge"]),
    }
    # crude attribution of each wrong answer / abstention to the most likely factor (first that applies)
    def cause(r):
        if r["box"] is None:
            return "no note found"
        if r["brightness"] < 80 or r["brightness"] > 200:
            return "poor lighting"
        if r["sharpness"] < 50:
            return "blur"
        if r["touches_edge"] and r["box_fill"] > 0.85:
            return "note fills / overflows frame (cropping)"
        return "none of the above (likely domain shift / look-alike denomination)"
    attribution = {o: dict(Counter(cause(r) for r in rows if r["outcome"] == o)) for o in ("wrong", "no_answer")}

    stem = f"{Path(a.weights).stem}_{a.split}"
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ORDER + ["no_answer"]
    with (OUT / f"{stem}_confusion.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["truth \\ answer"] + cols)
        for t in cm:
            w.writerow([t] + [cm[t][c] for c in cols])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = np.array([[cm[t][c] for c in cols] for t in cm], float)
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=140)
    ax.imshow(M / M.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j]:
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] / M[i].sum() > 0.5 else "black")
    ax.set_xticks(range(len(cols)), [c.replace("_taka", "") for c in cols], rotation=0)
    ax.set_yticks(range(len(cm)), [t.replace("_taka", "") for t in cm])
    ax.set_xlabel("answer (tau = %.2f)" % a.tau)
    ax.set_ylabel("true denomination")
    ax.set_title(f"{Path(a.weights).name} on NSTU {a.split} (n = {len(rows)})")
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}_confusion.png")
    wrong = sorted((r for r in rows if r["outcome"] == "wrong"), key=lambda r: -r["conf"])
    montage(wrong, OUT / f"{stem}_failures_wrong.jpg",
            lambda r: f"{r['truth'][:-5]} -> {r['answer'][:-5]} {r['conf']:.2f}")
    abst = [r for r in rows if r["outcome"] == "no_answer"]
    montage(abst[::max(1, len(abst) // 24)], OUT / f"{stem}_failures_no_answer.jpg",
            lambda r: f"{r['truth'][:-5]}: top {str(r['pred'])[:-5]} {r['conf']:.2f}")

    res = {"weights": a.weights, "tau": a.tau, "split": a.split, "class_mapping": mapping,
           "preprocessing": "imgsz 640 at training (runs/.../closeup_ft args.yaml) and in the app (currency_mode.py)",
           "overall": overall, "overall_excluding_originals_also_in_train": overall_clean,
           "n_originals_also_in_train": sum(r["in_train_too"] for r in rows), "per_class": per_class,
           "confusion": {t: dict(cm[t]) for t in cm}, "top_confusions": [[t, p, n] for (t, p), n in wrong_pairs],
           "factors": factors, "attribution": attribution,
           "rows": [{k: v for k, v in r.items()} for r in rows]}
    (OUT / f"{stem}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("class_mapping", "overall", "overall_excluding_originals_also_in_train",
                                          "top_confusions", "factors", "attribution")}, indent=1))
    for c in ("200_taka", "500_taka", "1000_taka"):
        print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in per_class.get(c, {}).items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
