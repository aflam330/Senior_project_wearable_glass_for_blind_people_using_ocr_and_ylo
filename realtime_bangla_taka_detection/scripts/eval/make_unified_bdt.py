"""Unified Bangladeshi Taka benchmark manifest (JaalTaka-Seq + denomination + open-set + external counterfeit).

Tasks and leakage-aware splits (no new images; paths only):
  counterfeit_multiview  JaalTaka, 1,390 notes x 6 views; print-disjoint split (results/serial_split) and the
                         original note-disjoint seed-42 split (results/camva/splits)
  counterfeit_wholenote  Counterfeit Currency Image Dataset (500/1000 BDT), test only; grouped by burst / series
  denomination           BanglaTaka raw (train source only), NSTU-BDTAKA Recognition (its own train/val/test, 88
                         leaked test originals flagged), Bangla Money Training (test only)
  detection              NSTU-BDTAKA Detection (its own splits)
  open_set_unknown       Large Scale BDT DB 2026 (coins + demonetized notes), Bangla Money 1-taka (test unknowns)
Output: results/unified_bdt/manifest.json and summary.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
sys.path.insert(0, str(ROOT))
from roboeye.camva.notes import SPLIT_DIR, load_splits  # noqa: E402

IMG = {".jpg", ".jpeg", ".png", ".bmp"}
OUT = ROOT / "results" / "unified_bdt"


def imgs(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG) if folder.is_dir() else []


def rel(p: Path) -> str:
    return str(p.relative_to(WORK)).replace("\\", "/")


def main() -> None:
    rows = []
    _, rec = load_splits(SPLIT_DIR)
    ser = {s: json.loads((ROOT / "results/serial_split" / f"{s}_note_ids.json").read_text(encoding="utf-8")) for s in ("train", "val", "test")}
    std = {s: json.loads((SPLIT_DIR / f"{s}_note_ids.json").read_text(encoding="utf-8")) for s in ("train", "val", "test")}
    inv = lambda d: {n: s for s, ns in d.items() for n in ns}
    ser_i, std_i = inv(ser), inv(std)
    for nid, r in rec.items():
        rows.append({"task": "counterfeit_multiview", "dataset": "JaalTaka", "unit": nid, "label": "genuine" if int(r["label"]) else "counterfeit",
                     "n_images": len(r["view_paths"]), "split_print_disjoint": ser_i.get(nid), "split_note_disjoint": std_i.get(nid)})
    cf = WORK / "data set/Bangladeshi Counterfeit Currency Image Dataset/Denomination-wise Distribution"
    for folder in sorted(p for p in cf.iterdir() if p.is_dir()):
        for p in imgs(folder):
            rows.append({"task": "counterfeit_wholenote", "dataset": "CounterfeitCurrencyImageDataset", "unit": rel(p),
                         "label": "counterfeit" if "Counterfeit" in folder.name else "genuine", "denomination": folder.name.split()[0],
                         "augmented": p.name.startswith("augmented"), "split": "test"})
    raw = WORK / "data set/Bangladeshi_Paper_Currency_Raw/Bangladeshi_Paper_Currency_Raw"
    for folder in sorted(p for p in raw.iterdir() if p.is_dir()):
        for p in imgs(folder):
            rows.append({"task": "denomination", "dataset": "BanglaTaka", "unit": rel(p), "label": folder.name, "split": "train_source"})
    nstu = WORK / "data set for comparison/NSTU-BDTAKA dataset Mendeley/data"
    for split in ("train", "validation", "test"):
        for folder in sorted((nstu / "Recognition" / split).glob("*")) if (nstu / "Recognition" / split).is_dir() else []:
            for p in imgs(folder):
                rows.append({"task": "denomination", "dataset": "NSTU-BDTAKA-Recognition", "unit": rel(p), "label": folder.name, "split": split})
        for p in imgs(nstu / "Detection" / split / "images"):
            rows.append({"task": "detection", "dataset": "NSTU-BDTAKA-Detection", "unit": rel(p), "label": "Taka", "split": split})
    bm = WORK / "data set for comparison/Bangla Money dataset Kaggle/bangla_banknote_v2/Training"
    for folder in sorted(p for p in bm.iterdir() if p.is_dir()):
        task = "open_set_unknown" if folder.name == "1" else "denomination"
        for p in imgs(folder):
            rows.append({"task": task, "dataset": "BanglaMoney", "unit": rel(p), "label": folder.name, "split": "test"})
    ls = WORK / "data set for comparison/Large Scale BDT DB 2026 Kaggle"
    for p in imgs(ls):
        rows.append({"task": "open_set_unknown", "dataset": "LargeScaleBDT2026", "unit": rel(p), "label": p.parent.name, "split": "val_unknown"})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
    summ = {"units_total": len(rows),
            "by_task_dataset": {f"{t}/{d}": c for (t, d), c in Counter((r["task"], r["dataset"]) for r in rows).items()},
            "images_total": sum(r.get("n_images", 1) for r in rows),
            "counterfeit_labelled_units": sum(1 for r in rows if r["task"].startswith("counterfeit")),
            "jaaltaka_print_disjoint_test": sum(1 for r in rows if r.get("split_print_disjoint") == "test")}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
