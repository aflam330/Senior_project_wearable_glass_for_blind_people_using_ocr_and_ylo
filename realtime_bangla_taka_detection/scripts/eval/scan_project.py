"""Full project scan: compile Python, load model files, inventory datasets, results and claims.

Writes results/scan/scan.json and paper_evidence/SCAN_COMPLETE.md.
Skips venv/, .git/, runs/ and cache/. Nothing is modified outside those two outputs.
"""
from __future__ import annotations

import json
import os

import sys
import time
import traceback
from collections import Counter
from pathlib import Path

WS = Path(__file__).resolve().parents[3]
ROOT = WS / "realtime_bangla_taka_detection"
OUT = ROOT / "results" / "scan"
SKIP_DIRS = {"venv", ".venv", ".git", "__pycache__", "runs", "cache", "node_modules"}
DATA_DIRS = [WS / "data set", WS / "data set for comparison"]
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_EXT = {".pt", ".pts", ".onnx", ".torchscript", ".pth"}


def walk(top: Path, skip_data: bool = True):
    for dirpath, dirnames, filenames in os.walk(top):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS
                       and not (skip_data and Path(dirpath, d) in DATA_DIRS)]
        for f in filenames:
            yield Path(dirpath, f)


def compile_all() -> dict:
    ok, bad = 0, []
    for p in walk(WS):
        if p.suffix == ".py":
            try:  # compile in memory; py_compile cannot target os.devnull on Windows
                compile(p.read_bytes(), str(p), "exec", dont_inherit=True)
                ok += 1
            except (SyntaxError, ValueError) as exc:
                bad.append({"file": str(p.relative_to(WS)), "error": f"{type(exc).__name__}: {exc}"[:300]})
    return {"compiled_ok": ok, "failed": bad}


def load_models() -> list[dict]:
    import torch
    rows = []
    for p in walk(WS):
        if p.suffix.lower() not in MODEL_EXT:
            continue
        row = {"file": str(p.relative_to(WS)), "mb": round(p.stat().st_size / 1e6, 2)}
        t0 = time.time()
        try:
            if p.suffix.lower() == ".onnx":
                import onnxruntime as ort
                s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
                row["kind"] = "onnx"
                row["inputs"] = [(i.name, i.shape) for i in s.get_inputs()]
            elif p.suffix.lower() in {".torchscript", ".pts"}:
                torch.jit.load(str(p), map_location="cpu")
                row["kind"] = "torchscript"
            else:
                blob = torch.load(str(p), map_location="cpu", weights_only=False)
                if isinstance(blob, dict):
                    keys = list(blob.keys())
                    row["kind"] = "dict"
                    row["keys"] = keys[:8]
                    for k in ("config", "cfg", "model_cfg", "kind", "name", "arch"):
                        if k in blob and isinstance(blob[k], (str, int, float)):
                            row[k] = blob[k]
                    if "model" in blob and hasattr(blob["model"], "yaml"):
                        row["kind"] = "ultralytics"
                        row["nc"] = blob["model"].yaml.get("nc")
                else:
                    row["kind"] = type(blob).__name__
            row["loads"] = True
        except Exception as exc:
            row["loads"] = False
            row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        row["seconds"] = round(time.time() - t0, 2)
        rows.append(row)
    return rows


def inventory_datasets() -> list[dict]:
    out = []
    for base in DATA_DIRS:
        if not base.is_dir():
            continue
        for ds in sorted(p for p in base.iterdir() if p.is_dir()):
            n_img, ext, per_dir, n_label = 0, Counter(), Counter(), 0
            for dirpath, dirnames, filenames in os.walk(ds):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                rel = str(Path(dirpath).relative_to(ds))
                for f in filenames:
                    e = os.path.splitext(f)[1].lower()
                    ext[e] += 1
                    if e in IMG:
                        n_img += 1
                        per_dir[rel] += 1
                    elif e in {".txt", ".json", ".csv", ".xml", ".yaml"}:
                        n_label += 1
            leaves = dict(sorted(per_dir.items()))
            out.append({
                "dataset": str(ds.relative_to(WS)), "images": n_img, "annotation_like_files": n_label,
                "extensions": dict(ext.most_common(8)),
                "n_image_folders": len(leaves),
                "image_folders": leaves if len(leaves) <= 60 else {"(many)": len(leaves)},
            })
    return out


def results_and_claims() -> dict:
    metrics = [p for p in walk(WS) if p.name == "test_metrics.json"]
    reg = json.loads((WS / "paper_evidence" / "CLAIM_REGISTRY.json").read_text(encoding="utf-8"))
    status = Counter(str(c.get("status")) for c in reg["claims"])
    missing = []
    for c in reg["claims"]:
        art = c.get("source_artifact") or ""
        if str(c.get("status", "")).startswith("VERIFIED") or c.get("status") == "NOT_MET":
            if art and not any((b / art.replace("\\", "/")).exists() for b in (WS, ROOT)):
                missing.append(c["claim_id"])
    return {"n_test_metrics_json": len(metrics), "claims": len(reg["claims"]),
            "claim_status": dict(status), "claims_missing_artifact": missing}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    scan = {"workspace": str(WS)}
    for name, fn in (("python", compile_all), ("models", load_models),
                     ("datasets", inventory_datasets), ("results", results_and_claims)):
        print("scanning", name, flush=True)
        try:
            scan[name] = fn()
        except Exception:
            scan[name] = {"scan_error": traceback.format_exc()[-800:]}
    scan["seconds"] = round(time.time() - t0, 1)
    (OUT / "scan.json").write_text(json.dumps(scan, indent=1, default=str), encoding="utf-8")
    print("done", scan["seconds"], "s")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
