"""Self-test of the research evidence database. Works on a temporary copy; the real master file is not touched.

Checks: bad records are rejected (missing fields, bad status, NaN, text in metrics, planned with metrics, verified
without evidence), ids never repeat, the same evidence is not recorded twice, a changed evidence file becomes a new
linked record (the old one stays), inbox collection, review, claims that point at missing or unverified experiments,
CSV export, and that existing records are byte-for-byte unchanged after new ones are added.
Run: python research_results/test_results_db.py
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research_results import results_db as DB  # noqa: E402


def rejected(db, rec) -> bool:
    try:
        DB.append(db, rec)
    except ValueError:
        return True
    return False


def main() -> None:
    real_path = DB.MASTER
    real_hash = DB.sha256(real_path) if real_path.exists() else None
    real = json.loads(real_path.read_text(encoding="utf-8")) if real_path.exists() else DB.empty_db()
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        repo = d / "repo"
        (repo / "research_results").mkdir(parents=True)
        (repo / "results").mkdir()
        shutil.copy2(DB.HERE / "schema.json", repo / "research_results" / "schema.json")
        DB.REPO, DB.HERE = repo, repo / "research_results"
        DB.MASTER, DB.INBOX, DB.EXPORTS = DB.HERE / "experiments.json", DB.HERE / "inbox", DB.HERE / "exports"
        evf = repo / "results" / "eval_a.json"
        evf.write_text(json.dumps({"accuracy": 0.9}), encoding="utf-8")
        ev = lambda: [{"type": "evaluation_output", "file": "results/eval_a.json", "sha256": DB.sha256(evf)}]
        good = {"experiment_name": "Model A on set T", "task": "classification", "status": "verified", "timestamp": DB.now_iso(),
                "metrics": {"accuracy": 0.9}, "evidence": ev(), "verification": {"method": "test"},
                "comparison_group": {"id": "CMP_T", "name": "test group"}, "comparison": {"variant": "A"}, "run": {"seed": 1}}
        db = DB.empty_db()
        # --- bad records are rejected
        assert rejected(db, {**good, "status": "great"}), "invalid status accepted"
        assert rejected(db, {**good, "metrics": {"accuracy": float("nan")}}), "NaN accepted"
        assert rejected(db, {**good, "metrics": {"accuracy": "94%"}}), "text metric accepted"
        assert rejected(db, {**good, "timestamp": "yesterday"}), "bad timestamp accepted"
        assert rejected(db, {k: v for k, v in good.items() if k != "task"}), "missing task accepted"
        assert rejected(db, {**good, "evidence": []}), "verified without evidence accepted"
        assert rejected(db, {**good, "verification": {}}), "verified without a verification method accepted"
        assert rejected(db, {**good, "evidence": [{"file": "results/not_there.json"}]}), "missing evidence file accepted"
        assert rejected(db, {"experiment_name": "Plan", "task": "x", "status": "planned", "metrics": {"accuracy": 1.0}}), "planned with metrics accepted"
        assert db["experiments"] == [], "a rejected record was stored"
        # --- append-only ids, no duplicate evidence
        a = DB.append(db, good)
        assert a == "EXP_0001"
        assert DB.append(db, good) is None, "same evidence recorded twice"
        b = DB.append(db, {**good, "experiment_name": "Model B on set T", "comparison": {"variant": "B"}, "metrics": {"accuracy": 0.8}})
        assert b == "EXP_0002"
        snapshot = json.dumps(db["experiments"][0], sort_keys=True)
        # --- a changed evidence file becomes a new record linked to the old one
        evf.write_text(json.dumps({"accuracy": 0.95}), encoding="utf-8")
        c = DB.append(db, {**good, "metrics": {"accuracy": 0.95}, "evidence": ev(), "run": {"seed": 2}})
        assert c == "EXP_0003" and db["experiments"][2]["run"]["rerun_of"] == "EXP_0001"
        assert json.dumps(db["experiments"][0], sort_keys=True) == snapshot, "an old record was modified"
        DB.save(db)
        assert len(DB.evidence_changes(DB.load())) == 2, "changed evidence not noticed"  # EXP_0001 and EXP_0002 point at the old content
        # --- failed and planned records are kept
        assert DB.append(db, {"experiment_name": "Crashed run", "task": "classification", "status": "failed", "notes": ["out of memory"]})
        assert DB.append(db, {"experiment_name": "Future run", "task": "classification", "status": "planned"})
        DB.save(db)
        # --- inbox: automatic records start partially_verified; collect merges them
        f2 = repo / "results" / "eval_b.json"
        f2.write_text("{}", encoding="utf-8")
        out = DB.record({"experiment_name": "Auto run", "task": "classification", "status": "partially_verified", "metrics": {"accuracy": 0.7},
                         "evidence": [{"type": "evaluation_output", "file": str(f2)}]})
        assert out and out.exists()
        (DB.INBOX / "bad.json").write_text(json.dumps({"experiment_name": "bad", "task": "x", "status": "verified"}), encoding="utf-8")
        st = DB.collect(verbose=False)
        assert st["added"] == ["EXP_0006"] and len(st["rejected"]) == 1 and (DB.INBOX / "bad.json").exists()
        assert DB.collect(verbose=False)["added"] == [], "collect is not idempotent"
        db = DB.load()
        assert db["experiments"][-1]["status"] == "partially_verified"
        # --- review promotes it and records the change
        assert DB.main(["review", "EXP_0006", "--by", "tester"]) == 0
        r = DB.load()["experiments"][-1]
        assert r["status"] == "verified" and r["status_history"][0]["from"] == "partially_verified" and r["metrics"] == {"accuracy": 0.7}
        # --- validation of the whole file
        db = DB.load()
        assert DB.validate(db) == [], DB.validate(db)
        db2 = json.loads(json.dumps(db))
        db2["experiments"].append(dict(db2["experiments"][0]))
        assert any("duplicate experiment_id" in p for p in DB.validate(db2))
        db3 = json.loads(json.dumps(db))
        db3["publication_claims"].append({"claim_id": "CLAIM_001", "claim": "x", "supporting_experiments": ["EXP_9999"], "status": "verified"})
        assert any("does not exist" in p for p in DB.validate(db3))
        db4 = json.loads(json.dumps(db))
        db4["publication_claims"].append({"claim_id": "CLAIM_001", "claim": "x", "supporting_experiments": ["EXP_0004"], "status": "verified"})
        assert any("all verified" in p for p in DB.validate(db4)), "verified claim on a failed experiment accepted"
        # --- claims through the command line
        cf = d / "claim.json"
        cf.write_text(json.dumps({"claim": "Model A scored higher than model B on set T.", "supporting_experiments": ["EXP_0001", "EXP_0002"],
                                  "status": "verified"}), encoding="utf-8")
        assert DB.main(["add-claim", str(cf)]) == 0 and DB.load()["publication_claims"][0]["claim_id"] == "CLAIM_001"
        # --- export
        counts = DB.export()
        rows = list(csv.DictReader(open(DB.EXPORTS / "experiments.csv", encoding="utf-8-sig")))
        long = list(csv.DictReader(open(DB.EXPORTS / "metrics.csv", encoding="utf-8-sig")))
        comp = list(csv.DictReader(open(DB.EXPORTS / "comparisons.csv", encoding="utf-8-sig")))
        assert len(rows) == 6 and counts["experiments"] == 6
        assert {r["metric"] for r in long} == {"accuracy"} and len(long) == 4
        va = next(r for r in comp if r["variant"] == "A")
        assert va["n_runs"] == "2" and math.isclose(float(va["mean"]), 0.925) and va["sd"] != ""
        vb = next(r for r in comp if r["variant"] == "B")
        assert vb["n_runs"] == "1" and vb["sd"] == "", "sd reported for a single run"
    # --- the real master file: untouched by this test, valid, unique ids, nothing line-level in it
    assert (DB.sha256(real_path) if real_path.exists() else None) == real_hash, "the test modified the real master file"
    ids = [r["experiment_id"] for r in real["experiments"]]
    assert len(ids) == len(set(ids)), "duplicate ids in the real master file"
    biggest = max((len(json.dumps(r)) for r in real["experiments"]), default=0)
    assert biggest < 40_000, f"a record holds {biggest} bytes: that looks like raw per-sample data"
    print(f"research evidence database: all checks pass ({len(ids)} experiments in the real master file, largest record {biggest} bytes)")


if __name__ == "__main__":
    main()
