"""Corruption eval for PRMVT, NDAL, and the CNN+ViT baseline.

Each corruption is decoded once at 6 views, then sliced to 1..6 views for all three models.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from roboeye.camva.data import NoteViewDataset, collate_notes
from roboeye.camva.engine import load_baseline
from roboeye.camva.notes import load_splits
from roboeye.config import DEVICE
from roboeye.qduig.config_io import load_config
from roboeye.qduig.engine import load_qduig
from roboeye.qduig.metrics_ext import full_binary_report
from scripts.train.train_novel import BUILDERS, _module

PAPER = ROOT.parent / "paper_evidence"
OUT = ROOT / "results" / "robustness" / "top_seed42.json"
CORRUPTIONS = [
    ("gaussian_blur", 3.0),
    ("motion_blur", 5.0),
    ("low_light", 0.35),
    ("brightness", 1.6),
    ("contrast", 1.8),
    ("glare", 0.65),
    ("occlusion", 0.2),
    ("occlusion", 0.55),
    ("jpeg", 30.0),
    ("rotation", 20.0),
    ("perspective", 0.1),
    ("scale", 0.7),
    ("sensor_noise", 12.0),
]


def _stack(test, records, corruption, severity):
    ds = NoteViewDataset(test, records, n_views=6, train=False, corruption=corruption, corruption_severity=severity)
    views, labels = [], []
    for i in range(len(ds)):
        item = ds[i]
        views.append(item["views"])
        labels.append(int(item["label"]))
    return torch.stack(views, dim=0), np.array(labels, dtype=np.int64)


def _acc_from_probs(y, p) -> float:
    return float(full_binary_report(y, p)["accuracy"])


def _run_models(views, y, ndal, prmvt, baseline, batch=16):
    n = views.size(0)
    out = {}
    for k in range(1, 7):
        chunk = views[:, :k]
        mask = torch.ones(n, k, dtype=torch.long)
        probs = {name: [] for name in ("ndal", "prmvt", "baseline")}
        for start in range(0, n, batch):
            v = chunk[start:start + batch].to(DEVICE)
            m = mask[start:start + batch].to(DEVICE)
            with torch.inference_mode():
                if ndal is not None:
                    probs["ndal"].append(ndal(v, m)["prob"].detach().cpu().numpy())
                if prmvt is not None:
                    probs["prmvt"].append(prmvt(v, m)["prob"].detach().cpu().numpy())
                if baseline is not None:
                    logit = baseline(v)
                    probs["baseline"].append(torch.softmax(logit, dim=1)[:, 1].detach().cpu().numpy())
        for name, parts in probs.items():
            if parts:
                out[(name, k)] = _acc_from_probs(y, np.concatenate(parts))
    return out


def main() -> None:
    splits, records = load_splits()
    test = splits["test"]
    ndal_path = ROOT / "results" / "novel_v2" / "ndal" / "seed42" / "checkpoint.pt"
    prmvt_path = ROOT / "results" / "qduig" / "prefix_ft" / "seed42" / "checkpoint.pt"
    base_path = ROOT / "results" / "camva" / "checkpoints" / "baseline_cnnvit_seed42.pt"
    ndal = prmvt = baseline = None
    if ndal_path.is_file():
        ndal = BUILDERS["ndal"]().to(DEVICE)
        blob = torch.load(ndal_path, map_location=DEVICE)
        _module(ndal).load_state_dict(blob["model"])
        ndal.eval()
    if prmvt_path.is_file():
        prmvt = load_qduig(prmvt_path, load_config(ROOT / "configs" / "proposed_prefix_ft.yaml"))
    if base_path.is_file():
        baseline = load_baseline(base_path)
    print("loading clean views", flush=True)
    clean_views, y = _stack(test, records, None, 0.0)
    clean = _run_models(clean_views, y, ndal, prmvt, baseline)
    del clean_views
    rows = []
    done = set()
    if OUT.is_file():
        previous = json.loads(OUT.read_text(encoding="utf-8"))
        rows = previous.get("rows", [])
        done = {(row["corruption"], row["severity"], row["k"]) for row in rows}
        if previous.get("clean"):
            clean = {tuple(key.split("_")): value for key, value in previous["clean"].items()}
            clean = {(name, int(k)): value for name, k, value in ((key[0], key[1], value) for key, value in previous["clean"].items())}
    for name, severity in CORRUPTIONS:
        if all((name, severity, k) in done for k in range(1, 7)):
            print(f"skip {name} {severity}", flush=True)
            continue
        print(f"loading {name} {severity}", flush=True)
        views, y = _stack(test, records, name, severity)
        measured = _run_models(views, y, ndal, prmvt, baseline)
        del views
        rows = [row for row in rows if not (row["corruption"] == name and row["severity"] == severity)]
        for k in range(1, 7):
            entry = {"corruption": name, "severity": severity, "k": k}
            for model in ("ndal", "prmvt", "baseline"):
                acc = measured.get((model, k))
                entry[model] = acc
                base_clean = clean.get((model, k))
                entry[f"{model}_drop"] = None if acc is None or base_clean is None else base_clean - acc
            rows.append(entry)
            print(f"{name} {severity} {k}view { {m: entry.get(m) for m in ('ndal', 'prmvt', 'baseline')} }", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({
            "clean": {f"{a}_{k}": v for (a, k), v in clean.items()},
            "rows": rows,
        }, indent=2), encoding="utf-8")
    lines = [
        "# Robustness, seed 42, test split",
        "",
        "Drop is clean accuracy minus corrupted accuracy at the same view count. Thirteen corruptions were measured, including the twelve named in the plan plus sensor noise.",
        "Source: `realtime_bangla_taka_detection/results/robustness/top_seed42.json`.",
        "",
        "| corruption | severity | views | PRMVT | drop | NDAL | drop | baseline | drop |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def cell(key: str) -> str:
            value = row.get(key)
            return "NOT_MEASURED" if value is None else f"{value:.4f}"
        lines.append(
            f"| {row['corruption']} | {row['severity']} | {row['k']} | {cell('prmvt')} | {cell('prmvt_drop')} | {cell('ndal')} | {cell('ndal_drop')} | {cell('baseline')} | {cell('baseline_drop')} |"
        )
    (PAPER / "ROBUSTNESS_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(PAPER / "ROBUSTNESS_RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
