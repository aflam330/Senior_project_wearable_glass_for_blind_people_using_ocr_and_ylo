"""Automatic result collection: one function per evaluation script. Each builds ONE experiment-level record from the
numbers the script just computed and drops it in research_results/inbox/ (see results_db.record).

Records made here start as 'partially_verified': the numbers are real, but nobody has looked at them yet.
`python research_results/results_db.py review EXP_xxxx --by <name>` marks a record verified after a look.
Nothing here stores per-image or per-frame data.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import results_db as DB

AUTO = {"method": "recorded automatically by the evaluation script; not yet reviewed"}


def _group(gid, name, purpose=""):
    return {"id": gid, "name": name, "purpose": purpose}


def ocr_bench(result: dict, path, meta: dict) -> None:
    """savior_glass/ocr_bench/bench.py save(): one record per method x phrase set (3 image seeds inside)."""
    if "overall" not in result or not result.get("overall"):
        return
    stem = Path(path).stem
    st, _, method = stem.partition("_")
    m = {"cer": result["overall"]["cer"], "wer": result["overall"]["wer"], "exact_match": result["overall"]["exact"]}
    for lang in ("bn", "en"):
        if result.get(lang):
            m["cer_bangla" if lang == "bn" else "cer_english"] = result[lang]["cer"]
    for cond, blk in (result.get("by_cond") or {}).items():
        if blk.get("all"):
            m[f"cer_{cond}"] = blk["all"]["cer"]
    rec = {"experiment_name": f"OCR {method} on {st}", "task": "ocr", "mode": "ocr", "status": "partially_verified", "verification": dict(AUTO),
           "metrics": m, "comparison_group": _group("OCR_BENCHMARK", "OCR pipelines on the shaped Bangla + English benchmark"),
           "comparison": {"variant": method},
           "dataset": {"name": "OCR benchmark (HarfBuzz-shaped Bangla + English)", "split": st, "samples": result["overall"]["n"],
                       "real_or_synthetic": "synthetic text on clean / distorted pages and real COCO photos"},
           "configuration": {k: v for k, v in (meta or {}).items() if k in ("config", "choices", "prep", "method", "task", "seed")},
           "runs": {"count": len(result.get("per_seed_cer", [])), "metric": "cer per image seed", "values": result.get("per_seed_cer", [])},
           "evidence": [{"type": "evaluation_output", "file": str(path), "selector": "overall, bn, en, by_cond"}]}
    if result.get("latency_ms_median"):
        rec["performance"] = {"latency_ms_median": result["latency_ms_median"]}
    DB.record(rec)


def pi5_benchmark(payload: dict, path) -> None:
    """savior_glass/scripts/benchmark_pi5.py: one record per timed module, one for the sustained run."""
    h = payload.get("host", {})
    hw = {"device": h.get("device_model") or h.get("hostname", ""), "hostname": h.get("hostname", ""), "raspberry_pi_5": h.get("raspberry_pi_5"),
          "cpu": f"{h.get('machine', '')}, {h.get('cpu_count', '')} cores"}
    env = {"os": h.get("os", ""), "python": h.get("python", ""),
           "framework_versions": {k: h[k] for k in ("torch", "ultralytics", "onnxruntime", "easyocr", "cv2") if k in h}}
    gid = "PI5_LATENCY" if h.get("raspberry_pi_5") else "HOST_LATENCY"
    for name, x in (payload.get("latency") or {}).items():
        perf = {k2: x[k1] for k1, k2 in (("median_ms", "latency_ms_median"), ("p95_ms", "latency_ms_p95"), ("mean_ms", "latency_ms_mean"),
                                         ("fps_median", "fps"), ("rss_mb_after", "memory_rss_mb_after"), ("cpu_temp_c_after", "cpu_temp_c_after"),
                                         ("load_ms", "load_ms")) if x.get(k1) is not None}
        DB.record({"experiment_name": f"{'Pi 5' if h.get('raspberry_pi_5') else 'Host'} latency: {name}", "task": "system_performance", "mode": name,
                   "status": "partially_verified", "verification": dict(AUTO), "performance": perf, "raw_measurements": {"iterations": x.get("iters")},
                   "comparison_group": _group(gid, "Latency per module (benchmark_pi5.py)"), "comparison": {"variant": name}, "hardware": hw,
                   "software_environment": env, "evidence": [{"type": "benchmark_output", "file": str(path), "selector": f"latency.{name}"}]})
    s = payload.get("sustained")
    if s:
        DB.record({"experiment_name": f"{'Pi 5' if h.get('raspberry_pi_5') else 'Host'} sustained mixed run", "task": "system_performance",
                   "status": "partially_verified", "verification": dict(AUTO),
                   "performance": {k: s[k] for k in ("minutes", "calls", "max_cpu_temp_c") if s.get(k) is not None} | (
                       {"peak_rss_mb": payload["peak_rss_mb"]} if payload.get("peak_rss_mb") is not None else {}),
                   "raw_measurements": {"final_throttled": s.get("final_throttled"), "per_call_csv": s.get("csv")},
                   "comparison_group": _group(gid, "Latency per module (benchmark_pi5.py)"), "comparison": {"variant": "sustained run"},
                   "hardware": hw, "software_environment": env,
                   "evidence": [{"type": "benchmark_output", "file": str(path), "selector": "sustained"}]})


def field_run(summary: dict, run_dir) -> None:
    """savior_glass/scripts/field_report.py --record: one record per real-time test run on the device.
    Field logs are not committed (they can hold photos), so the small summary.json is copied to research_results/evidence/."""
    run_dir = Path(run_dir)
    dst = DB.HERE / "evidence" / "field" / run_dir.name
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "summary.json", dst / "summary.json")
    perf = {}
    for key, v in (summary.get("latency_ms") or {}).items():
        tag = key.replace(":", "_")
        perf[f"{tag}_median_ms"] = v["median"]
        perf[f"{tag}_p95_ms"] = v["p95"]
        perf[f"{tag}_n"] = v["n"]
    sysb = summary.get("system") or {}
    for k in ("max_temp_c", "min_mem_available_mb", "max_app_rss_mb", "max_cpu_busy_pct"):
        if sysb.get(k) is not None:
            perf[k] = sysb[k]
    acc = summary.get("accuracy") or {}
    metrics, human = {}, {}
    if "denomination" in acc:
        metrics.update({"denomination_accuracy": acc["denomination"]["accuracy"], "denomination_n": acc["denomination"]["n"]})
    if "counterfeit_safety" in acc:
        metrics.update(acc["counterfeit_safety"])
    if "ocr" in acc:
        metrics.update({"ocr_mean_cer": acc["ocr"]["mean_cer"], "ocr_n": acc["ocr"]["n"]})
    labelled = bool(acc.get("labelled_rows"))
    rec = {"experiment_name": f"Field run {run_dir.name}", "task": "field_test", "mode": "real-time use on the device",
           "status": "partially_verified", "verification": {"method": "recorded by field_report.py from the run's CSV logs"
                                                            + ("; accuracy from hand-filled labels.csv" if labelled else "; no ground-truth labels yet")},
           "performance": perf, "configuration": summary.get("settings") or {},
           "raw_measurements": {"events": summary.get("events"), "duration_min": summary.get("duration_min"),
                                "throttled_values": sysb.get("throttled_values"), "labelled_rows": acc.get("labelled_rows", 0),
                                "currency_answers": summary.get("currency_answers"), "watermark_checks": summary.get("watermark_checks"),
                                "errors": len(summary.get("errors") or [])},
           "comparison_group": _group("FIELD_RUNS", "Real-time test runs on the device"), "comparison": {"variant": summary.get("device") or "device"},
           "hardware": {"device": summary.get("device") or ""}, "timestamp": summary.get("start"),
           "test_conditions": {"environment": "", "lighting": "", "notes": "fill in when reviewing"},
           "evidence": [{"type": "field_run_summary", "file": str(dst / "summary.json")}]}
    if metrics:
        rec["metrics"] = metrics
    if human:
        rec["human_feedback"] = human
    DB.record(rec)


def watermark_localizer(res: dict, path) -> None:
    """realtime_bangla_taka_detection/scripts/train/train_watermark_localizer.py: one record per seed."""
    c, loc = res["test_classification"], res["test_localization"]
    DB.record({"experiment_name": f"Learned watermark localizer + MobileNetV2, seed {res['seed']}", "task": "counterfeit_detection",
               "mode": "currency: watermark check", "status": "partially_verified", "verification": dict(AUTO), "run": {"seed": res["seed"]},
               "experiment_group": "WATERMARK_LOCALIZER",
               "metrics": {"accuracy_all_test_notes": c["localizer_all_test_notes"]["accuracy"], "mean_corner_err_frac": loc["mean_corner_err_frac"],
                           "median_quad_iou": loc["median_quad_iou"], "best_val_mean_corner_err": res["best_val_mean_corner_err"]},
               "comparison_group": _group("WATERMARK_CHECK", "Watermark-window check on unseen prints"),
               "comparison": {"variant": "Learned localizer + MobileNetV2"}, "dataset": {"name": str(res.get("split", "")), "split": "test"},
               "evidence": [{"type": "evaluation_output", "file": str(path), "selector": "test_localization, test_classification"}]})


def mode_switch(res: dict, path) -> None:
    """savior_glass/scripts/measure_mode_switch.py."""
    perf = {k: v for k, v in res.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    DB.record({"experiment_name": f"Currency mode switch timing on {res.get('host', 'host')}", "task": "system_performance", "mode": "currency",
               "status": "partially_verified", "verification": dict(AUTO), "performance": perf,
               "configuration": {"watermark_check": res.get("watermark_check"), "raspberry_pi_5": res.get("raspberry_pi_5")},
               "comparison_group": _group("DEVICE_BEHAVIOUR", "Glass interaction behaviour"), "comparison": {"variant": f"mode switch on {res.get('host', '')}"},
               "evidence": [{"type": "benchmark_output", "file": str(path)}]})


def generic(name: str, task: str, path, metrics: dict | None = None, performance: dict | None = None, **fields) -> None:
    """For any other script: record one experiment with its result file.
    Example: hooks.generic("ResNet-50 fine-tune, seed 42", "counterfeit_detection", out_path, metrics={"accuracy_k6": acc}, run={"seed": 42})"""
    rec = {"experiment_name": name, "task": task, "status": "partially_verified", "verification": dict(AUTO),
           "evidence": [{"type": "evaluation_output", "file": str(path)}], **fields}
    if metrics:
        rec["metrics"] = metrics
    if performance:
        rec["performance"] = performance
    DB.record(rec)


def call(kind: str, *args, **kwargs) -> None:
    """Entry point used by scripts; never raises."""
    try:
        globals()[kind](*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[research_results] {kind} not recorded: {exc}")
