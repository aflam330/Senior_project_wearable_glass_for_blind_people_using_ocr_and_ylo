"""Self-test of the automatic collection hooks, fed with real saved result files. The inbox and master file are
redirected to a temporary folder, so nothing in research_results/ changes. Run: python research_results/test_hooks.py"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(r"E:\Final SP")
sys.path.insert(0, str(REPO))
from research_results import hooks, results_db as DB

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    DB.INBOX = d / "inbox"
    real_master = DB.MASTER
    DB.MASTER = d / "experiments.json"
    DB.EXPORTS = d / "exports"
    real_here = DB.HERE
    # 1. OCR benchmark hook, fed with a real saved result
    p = REPO / "savior_glass/results/ocr_bench/test_final.json"
    res = json.loads(p.read_text(encoding="utf-8"))
    hooks.call("ocr_bench", res, p, res.get("meta", {}))
    # 2. Pi benchmark hook, fed with the real Pi run
    p2 = REPO / "savior_glass/results/pi5_benchmark_memo_20261001T111325Z.json"
    hooks.call("pi5_benchmark", json.loads(p2.read_text(encoding="utf-8")), p2)
    # 3. localizer and mode-switch hooks
    p3 = REPO / "realtime_bangla_taka_detection/results/watermark_localizer/seed42.json"
    hooks.call("watermark_localizer", json.loads(p3.read_text(encoding="utf-8")), p3)
    p4 = REPO / "savior_glass/results/mode_switch_after.json"
    hooks.call("mode_switch", json.loads(p4.read_text(encoding="utf-8")), p4)
    # 4. field run hook: build a summary from the field-log self-test style data (temp run folder)
    run = d / "20261003_000000_testhost"
    run.mkdir()
    summ = {"run": run.name, "start": DB.now_iso(), "device": "test device", "events": {"result": 3}, "duration_min": 1.0,
            "latency_ms": {"currency:result": {"n": 3, "median": 300.0, "p95": 320.0, "max": 330.0}}, "settings": {"OCR_PIPELINE": "v2"},
            "system": {"max_temp_c": 60.0, "throttled_values": [], "min_mem_available_mb": 5000.0, "max_app_rss_mb": 900.0, "max_cpu_busy_pct": 80.0},
            "currency_answers": {"n": 3}, "watermark_checks": {"n": 0}, "errors": [], "accuracy": {"labelled_rows": 0}}
    (run / "summary.json").write_text(json.dumps(summ), encoding="utf-8")
    DB.HERE = d / "rr"          # so the copied field summary lands in the temp folder, not the repo
    DB.REPO = d
    hooks.call("field_run", summ, run)
    DB.HERE, DB.REPO = real_here, REPO
    files = sorted(DB.INBOX.glob("*.json"))
    recs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    print("inbox records:", len(recs))
    by = {}
    for r in recs:
        by.setdefault(r["task"], []).append(r)
    print({k: len(v) for k, v in by.items()})
    assert all(r["status"] == "partially_verified" for r in recs)
    ocr = next(r for r in recs if r["task"] == "ocr")
    assert ocr["metrics"]["cer"] == res["overall"]["cer"] and ocr["evidence"][0]["file"] == "savior_glass/results/ocr_bench/test_final.json"
    assert ocr["evidence"][0]["sha256"] and ocr["versioning"]["code_commit"]
    pi = [r for r in recs if r["experiment_name"].startswith("Pi 5 latency")]
    assert len(pi) == 15 and all(r["hardware"]["raspberry_pi_5"] for r in pi)
    # merge everything except the field record (its evidence lives in the temp repo)
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        if r["task"] == "field_test":
            f.unlink()
    st = DB.collect(verbose=False)
    db = DB.load()
    print("collected:", len(st["added"]), "rejected:", st["rejected"], "problems:", DB.validate(db))
    assert len(st["added"]) == 19 and not st["rejected"] and not DB.validate(db)
    print("largest record bytes:", max(len(json.dumps(r)) for r in db["experiments"]))
    print("example field record keys:", sorted(next(r for r in recs if r["task"] == "field_test").keys()))
print("hooks: all checks pass; real master untouched:", real_master.exists())
