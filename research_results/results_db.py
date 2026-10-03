"""Research evidence database: experiment-level results in one JSON file (the source of truth).

Not a log. One record = one finished experiment (a model evaluated on a dataset under stated conditions),
with its metrics, the evidence file it came from, and a verification status.

  research_results/experiments.json   master file: experiments, comparison_groups, publication_claims
  research_results/schema.json        JSON Schema of the master file
  research_results/inbox/*.json       records dropped by evaluation scripts, not yet merged
  research_results/exports/*.csv      generated from the master file (never edited by hand)

Rules enforced here:
  - append-only: a record is never changed or removed by this code; ids are EXP_0001, EXP_0002, ...
  - a re-run, or a changed evidence file, becomes a NEW record with run.rerun_of pointing at the earlier one
  - metrics must be finite numbers; status must be one of STATUSES; every evidence file must exist
  - 'verified' is never a default: a record says how it was verified (verification.method)

Command line (run from anywhere):
  python research_results/results_db.py validate            check the master file; exit 1 on any problem
  python research_results/results_db.py collect             merge inbox records, validate, export CSV
  python research_results/results_db.py export              regenerate exports/*.csv
  python research_results/results_db.py summary             counts by group / task / status
  python research_results/results_db.py add record.json     add one hand-written record (see README)
  python research_results/results_db.py add-claim claim.json   add a publication claim linked to experiments
  python research_results/results_db.py review EXP_0195 --by NAME   mark an automatically collected record verified
  python research_results/results_db.py check-evidence      list records whose evidence file changed or is missing
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MASTER = HERE / "experiments.json"
INBOX = HERE / "inbox"
EXPORTS = HERE / "exports"
SCHEMA_VERSION = "1.0"
STATUSES = ("verified", "partially_verified", "unverified", "failed", "planned")
REQUIRED = ("experiment_id", "experiment_name", "timestamp", "task", "status")
ID_RE = re.compile(r"^EXP_\d{4,}$")
NUMERIC_BLOCKS = ("metrics", "performance")


# ---------------------------------------------------------------- basic helpers
def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path) -> str:
    """Repo-relative path with forward slashes (what is stored in evidence[].file)."""
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def git_commit(path=None) -> str:
    """Current commit, or the last commit that touched `path`. Empty string when git is not available."""
    try:
        cmd = ["git", "-C", str(REPO), "log", "-1", "--format=%H"] + (["--", str(path)] if path else [])
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def environment() -> dict:
    env = {"os": platform.platform(), "python": platform.python_version(), "framework_versions": {}}
    for mod in ("torch", "torchvision", "ultralytics", "onnxruntime", "easyocr", "cv2", "numpy"):
        m = sys.modules.get(mod)
        if m is not None and getattr(m, "__version__", None):
            env["framework_versions"][mod] = str(m.__version__)
    return env


def hardware() -> dict:
    hw = {"device": platform.node(), "cpu": platform.processor() or platform.machine()}
    try:
        with open("/proc/device-tree/model", encoding="utf-8", errors="ignore") as f:
            hw["device_model"] = f.read().strip("\x00 \n")
    except OSError:
        pass
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            hw["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return hw


def empty_db() -> dict:
    return {"schema_version": SCHEMA_VERSION, "description": "Research evidence database. Source of truth; CSV files are generated from it.",
            "created": now_iso(), "updated": now_iso(), "experiments": [], "comparison_groups": [], "publication_claims": []}


def load(path: Path | None = None) -> dict:
    path = path or MASTER  # looked up at call time, so a test can point the module at a temporary copy
    if not path.exists():
        return empty_db()
    return json.loads(path.read_text(encoding="utf-8"))


def save(db: dict, path: Path | None = None) -> None:
    """Atomic write: a crash can never leave a half-written master file."""
    path = path or MASTER
    db["updated"] = now_iso()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(db, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- validation
def _num_problems(prefix, block):
    out = []
    for k, v in (block or {}).items():
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            if isinstance(v, float) and not math.isfinite(v):
                out.append(f"{prefix}.{k} is not a finite number ({v})")
        elif isinstance(v, dict):
            out += _num_problems(f"{prefix}.{k}", v)
        elif isinstance(v, list):
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)) for x in v):
                out.append(f"{prefix}.{k} must be a list of finite numbers")
        else:
            out.append(f"{prefix}.{k} must be a number, got {type(v).__name__} ({str(v)[:40]!r})")
    return out


def validate_record(r: dict, check_files: bool = True) -> list:
    eid = r.get("experiment_id", "<no id>")
    p = []
    for k in REQUIRED:
        if r.get(k) in (None, ""):
            p.append(f"{eid}: missing required field '{k}'")
    if r.get("experiment_id") and not ID_RE.match(str(r["experiment_id"])):
        p.append(f"{eid}: experiment_id must look like EXP_0001")
    if r.get("status") not in STATUSES:
        p.append(f"{eid}: status {r.get('status')!r} is not one of {STATUSES}")
    for k in ("timestamp", "recorded_at"):
        if r.get(k):
            try:
                dt.datetime.fromisoformat(str(r[k]))
            except ValueError:
                p.append(f"{eid}: {k} {r[k]!r} is not ISO-8601")
    for b in NUMERIC_BLOCKS:
        if b in r and not isinstance(r[b], dict):
            p.append(f"{eid}: '{b}' must be an object")
        else:
            p += [f"{eid}: {x}" for x in _num_problems(b, r.get(b))]
    st = r.get("status")
    has_numbers = bool(r.get("metrics") or r.get("performance") or r.get("human_feedback"))
    if st in ("verified", "partially_verified") and not has_numbers:
        p.append(f"{eid}: status '{st}' but no metrics / performance / human_feedback recorded")
    if st == "verified":
        if not r.get("evidence"):
            p.append(f"{eid}: status 'verified' needs at least one evidence file")
        if not (r.get("verification") or {}).get("method"):
            p.append(f"{eid}: status 'verified' needs verification.method (how it was verified)")
    if st == "planned" and (r.get("metrics") or r.get("performance")):
        p.append(f"{eid}: a planned experiment must not carry metrics")
    for e in r.get("evidence") or []:
        if not isinstance(e, dict) or not e.get("file"):
            p.append(f"{eid}: each evidence item needs a 'file'")
        elif check_files and not (REPO / e["file"]).exists():
            p.append(f"{eid}: evidence file not found: {e['file']}")
    return p


def validate(db: dict, check_files: bool = True) -> list:
    p = []
    for k in ("schema_version", "experiments", "comparison_groups", "publication_claims"):
        if k not in db:
            p.append(f"master file: missing top-level key '{k}'")
    exps = db.get("experiments", [])
    ids = [r.get("experiment_id") for r in exps]
    p += [f"duplicate experiment_id {i}" for i, n in Counter(ids).items() if n > 1]
    keys = Counter(r.get("record_key") for r in exps if r.get("record_key"))
    p += [f"duplicate record_key {k} (the same evidence was recorded twice)" for k, n in keys.items() if n > 1]
    for r in exps:
        p += validate_record(r, check_files)
        ro = (r.get("run") or {}).get("rerun_of")
        if ro and ro not in ids:
            p.append(f"{r.get('experiment_id')}: run.rerun_of points at unknown {ro}")
    gids = {g.get("id") for g in db.get("comparison_groups", [])}
    for r in exps:
        g = (r.get("comparison_group") or {}).get("id")
        if g and g not in gids:
            p.append(f"{r.get('experiment_id')}: comparison_group {g} is not defined in comparison_groups")
    idset = set(ids)
    cids = [c.get("claim_id") for c in db.get("publication_claims", [])]
    p += [f"duplicate claim_id {i}" for i, n in Counter(cids).items() if n > 1]
    for c in db.get("publication_claims", []):
        if c.get("status") not in STATUSES:
            p.append(f"{c.get('claim_id')}: claim status must be one of {STATUSES}")
        for e in c.get("supporting_experiments", []):
            if e not in idset:
                p.append(f"{c.get('claim_id')}: supporting experiment {e} does not exist")
        if c.get("status") == "verified":
            bad = [e for e in c.get("supporting_experiments", []) if e in idset
                   and next(r for r in exps if r["experiment_id"] == e)["status"] != "verified"]
            if bad or not c.get("supporting_experiments"):
                p.append(f"{c.get('claim_id')}: a verified claim needs supporting experiments that are all verified (not verified: {bad})")
    try:  # optional second check against schema.json when the jsonschema package is present
        import jsonschema
        schema = json.loads((HERE / "schema.json").read_text(encoding="utf-8"))
        for err in jsonschema.Draft7Validator(schema).iter_errors(db):
            p.append("schema: " + "/".join(str(x) for x in err.absolute_path) + ": " + err.message[:160])
    except ImportError:
        pass
    return p


def evidence_changes(db: dict) -> list:
    """Records whose evidence file is gone or no longer has the hash it had when recorded (e.g. a script overwrote it).
    The record itself is still the measured result; the notice says it can no longer be traced to that exact file."""
    out = []
    for r in db["experiments"]:
        for e in r.get("evidence") or []:
            f = REPO / e.get("file", "")
            if not f.exists():
                out.append((r["experiment_id"], e.get("file"), "file missing"))
            elif e.get("sha256") and sha256(f) != e["sha256"]:
                out.append((r["experiment_id"], e.get("file"), "file content changed"))
    return out


# ---------------------------------------------------------------- writing records
def _next_id(db) -> str:
    n = max([int(r["experiment_id"].split("_")[1]) for r in db["experiments"] if ID_RE.match(str(r.get("experiment_id", "")))] or [0])
    return f"EXP_{n + 1:04d}"


def record_key(rec: dict) -> str:
    """Identity of the evidence behind a record: file(s) + selector + file hash. Same key = same result."""
    ev = [(e.get("file", ""), e.get("selector", ""), e.get("sha256", "")) for e in rec.get("evidence") or []]
    if not ev:
        return ""
    return hashlib.sha256(json.dumps([rec.get("experiment_name", ""), ev], sort_keys=True).encode()).hexdigest()[:20]


def append(db: dict, rec: dict) -> str | None:
    """Add one record to an in-memory db. Returns the new id, or None when the same evidence is already recorded.
    If the same experiment (same name + evidence file + selector) exists with a different file hash, the new
    record is added and linked with run.rerun_of; the old record stays untouched."""
    rec = json.loads(json.dumps(rec))  # deep copy, and proves it is JSON-serialisable
    rec.pop("experiment_id", None)
    rec.setdefault("recorded_at", now_iso())
    rec.setdefault("timestamp", rec["recorded_at"])
    key = record_key(rec)
    if key:
        rec["record_key"] = key
        if any(r.get("record_key") == key for r in db["experiments"]):
            return None
        same = [r for r in db["experiments"] if r.get("experiment_name") == rec.get("experiment_name")
                and [(e.get("file"), e.get("selector", "")) for e in r.get("evidence") or []]
                == [(e.get("file"), e.get("selector", "")) for e in rec.get("evidence") or []]]
        if same:
            rec.setdefault("run", {})["rerun_of"] = same[-1]["experiment_id"]
    ordered = {"experiment_id": _next_id(db)}
    ordered.update(rec)
    problems = validate_record(ordered)
    if problems:
        raise ValueError("record rejected:\n  " + "\n  ".join(problems))
    g = ordered.get("comparison_group")
    if g and g.get("id") and not any(x["id"] == g["id"] for x in db["comparison_groups"]):
        db["comparison_groups"].append({"id": g["id"], "name": g.get("name", g["id"]), "purpose": g.get("purpose", "")})
    db["experiments"].append(ordered)
    return ordered["experiment_id"]


def record(rec: dict) -> Path | None:
    """Called by evaluation scripts. Writes ONE record file to research_results/inbox/ (no master-file access, so the
    Pi and the laptop can both record without git conflicts). Fills timestamp, environment, hardware, code commit and
    evidence hashes. Never raises: an experiment must not fail because its record could not be saved."""
    try:
        rec = dict(rec)
        rec.setdefault("timestamp", now_iso())
        rec.setdefault("software_environment", environment())
        rec.setdefault("hardware", hardware())
        rec.setdefault("versioning", {})
        rec["versioning"].setdefault("code_commit", git_commit())
        for e in rec.get("evidence") or []:
            e["file"] = rel(e["file"])
            f = REPO / e["file"]
            if f.exists() and "sha256" not in e:
                e["sha256"] = sha256(f)
        INBOX.mkdir(parents=True, exist_ok=True)
        out = INBOX / f"{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}_{platform.node() or 'host'}_{uuid.uuid4().hex[:8]}.json"
        out.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[research_results] record not saved: {exc}", file=sys.stderr)
        return None


def collect(verbose=True) -> dict:
    """Merge inbox records into the master file. Merged files move to inbox/merged/; rejected ones stay, with the reason."""
    db = load()
    stats = {"added": [], "already_recorded": 0, "rejected": []}
    files = sorted(INBOX.glob("*.json")) if INBOX.exists() else []
    for f in files:
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
            eid = append(db, rec)
        except Exception as exc:  # noqa: BLE001
            stats["rejected"].append((f.name, str(exc)))
            continue
        if eid:
            stats["added"].append(eid)
        else:
            stats["already_recorded"] += 1
        (INBOX / "merged").mkdir(exist_ok=True)
        os.replace(f, INBOX / "merged" / f.name)
    if stats["added"]:
        save(db)
    if verbose:
        print(f"collect: {len(stats['added'])} added, {stats['already_recorded']} already recorded, {len(stats['rejected'])} rejected")
        for name, why in stats["rejected"]:
            print(f"  REJECTED {name}: {why}")
    return stats


# ---------------------------------------------------------------- CSV export (generated; JSON stays the master)
def _flat(prefix, obj, out):
    for k, v in (obj or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            _flat(key + ".", v, out)
        elif isinstance(v, list):
            out[key] = json.dumps(v, ensure_ascii=False)
        else:
            out[key] = v


BASE_COLS = ["experiment_id", "experiment_name", "status", "timestamp", "task", "mode", "algorithm", "experiment_group",
             "comparison_group_id", "variant", "model_name", "dataset_name", "dataset_split", "dataset_samples", "dataset_kind",
             "hardware_device", "seed", "ablation_component", "ablation_configuration", "decision", "evidence_file", "code_commit"]


def _base(r):
    return {"experiment_id": r["experiment_id"], "experiment_name": r.get("experiment_name", ""), "status": r.get("status", ""),
            "timestamp": r.get("timestamp", ""), "task": r.get("task", ""), "mode": r.get("mode", ""), "algorithm": r.get("algorithm", ""),
            "experiment_group": r.get("experiment_group", ""), "comparison_group_id": (r.get("comparison_group") or {}).get("id", ""),
            "variant": (r.get("comparison") or {}).get("variant", ""), "model_name": (r.get("model") or {}).get("name", ""),
            "dataset_name": (r.get("dataset") or {}).get("name", ""), "dataset_split": (r.get("dataset") or {}).get("split", ""),
            "dataset_samples": (r.get("dataset") or {}).get("samples", ""), "dataset_kind": (r.get("dataset") or {}).get("real_or_synthetic", ""),
            "hardware_device": (r.get("hardware") or {}).get("device", ""), "seed": (r.get("run") or {}).get("seed", ""),
            "ablation_component": (r.get("ablation") or {}).get("component", ""),
            "ablation_configuration": (r.get("ablation") or {}).get("configuration", ""),
            "decision": (r.get("comparison") or {}).get("decision", ""),
            "evidence_file": ((r.get("evidence") or [{}])[0]).get("file", ""), "code_commit": (r.get("versioning") or {}).get("code_commit", "")}


def _write_csv(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def export(db: dict | None = None) -> dict:
    """experiments.csv (wide: one row per experiment), metrics.csv (long: one row per number, for plotting),
    comparisons.csv (per comparison group and variant: n runs, mean, sd, min, max of each metric)."""
    db = db or load()
    EXPORTS.mkdir(exist_ok=True)
    wide, metric_cols, long_rows = [], [], []
    for r in db["experiments"]:
        row = _base(r)
        nums = {}
        _flat("metrics.", r.get("metrics"), nums)
        _flat("performance.", r.get("performance"), nums)
        _flat("human_feedback.", r.get("human_feedback"), nums)
        for k, v in nums.items():
            if k not in metric_cols:
                metric_cols.append(k)
            row[k] = v
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                kind, name = k.split(".", 1)
                long_rows.append({**_base(r), "kind": kind, "metric": name, "value": v})
        wide.append(row)
    _write_csv(EXPORTS / "experiments.csv", BASE_COLS + sorted(metric_cols), wide)
    _write_csv(EXPORTS / "metrics.csv", BASE_COLS + ["kind", "metric", "value"], long_rows)
    groups = defaultdict(list)
    names = {g["id"]: g.get("name", "") for g in db.get("comparison_groups", [])}
    for row in long_rows:
        if row["comparison_group_id"] and row["status"] in ("verified", "partially_verified"):
            groups[(row["comparison_group_id"], row["variant"] or row["model_name"], row["dataset_name"], row["dataset_split"],
                    row["hardware_device"], row["kind"], row["metric"])].append((row["experiment_id"], row["seed"], row["value"]))
    comp = []
    for (gid, variant, ds, split, hw, kind, metric), vals in sorted(groups.items(), key=lambda x: [str(v) for v in x[0]]):
        v = [x[2] for x in vals]
        mean = sum(v) / len(v)
        sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else ""  # sd only when >= 2 runs exist
        comp.append({"comparison_group_id": gid, "comparison_group": names.get(gid, ""), "variant": variant, "dataset_name": ds,
                     "dataset_split": split, "hardware_device": hw, "kind": kind, "metric": metric, "n_runs": len(v), "mean": mean,
                     "sd": sd, "min": min(v), "max": max(v), "seeds": " ".join(str(x[1]) for x in vals if x[1] != ""),
                     "experiment_ids": " ".join(x[0] for x in vals)})
    _write_csv(EXPORTS / "comparisons.csv", ["comparison_group_id", "comparison_group", "variant", "dataset_name", "dataset_split",
                                             "hardware_device", "kind", "metric", "n_runs", "mean", "sd", "min", "max", "seeds",
                                             "experiment_ids"], comp)
    claims = [{"claim_id": c.get("claim_id"), "status": c.get("status"), "claim": c.get("claim"),
               "supporting_experiments": " ".join(c.get("supporting_experiments", []))} for c in db.get("publication_claims", [])]
    _write_csv(EXPORTS / "claims.csv", ["claim_id", "status", "claim", "supporting_experiments"], claims)
    return {"experiments": len(wide), "metric_rows": len(long_rows), "comparison_rows": len(comp), "claims": len(claims)}


def summary(db=None) -> str:
    db = db or load()
    ex = db["experiments"]
    L = [f"{len(ex)} experiments, {len(db['comparison_groups'])} comparison groups, {len(db['publication_claims'])} publication claims",
         "status: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r["status"] for r in ex).items())),
         "task:   " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r["task"] for r in ex).items()))]
    for g in db["comparison_groups"]:
        n = sum((r.get("comparison_group") or {}).get("id") == g["id"] for r in ex)
        L.append(f"  {g['id']:28s} {n:4d}  {g.get('name', '')}")
    return "\n".join(L)


# ---------------------------------------------------------------- command line
def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    cmd = argv[0] if argv else "summary"
    if cmd == "validate":
        probs = validate(load())
        for x in probs:
            print("PROBLEM", x)
        print(f"validate: {len(load()['experiments'])} experiments, {len(probs)} problems")
        return 1 if probs else 0
    if cmd == "collect":
        collect()
        probs = validate(load())
        for x in probs:
            print("PROBLEM", x)
        print("export:", export())
        return 1 if probs else 0
    if cmd == "export":
        print("export:", export())
        return 0
    if cmd == "summary":
        print(summary())
        return 0
    if cmd == "check-evidence":
        changed = evidence_changes(load())
        for eid, f, why in changed:
            print(f"NOTICE {eid}: {f}: {why}")
        print(f"check-evidence: {len(changed)} records whose evidence file changed or is missing since it was recorded")
        return 0
    if cmd == "review" and len(argv) > 1:
        by = argv[argv.index("--by") + 1] if "--by" in argv else ""
        note = argv[argv.index("--note") + 1] if "--note" in argv else ""
        ids = [a for a in argv[1:] if ID_RE.match(a)]
        if not by or not ids:
            print("usage: review EXP_0001 [EXP_0002 ...] --by NAME [--note TEXT]")
            return 2
        db = load()
        changed = {e for e, _, _ in evidence_changes(db)}
        done = 0
        for r in db["experiments"]:
            if r["experiment_id"] not in ids:
                continue
            if r["status"] not in ("partially_verified", "unverified"):
                print(f"{r['experiment_id']}: status is '{r['status']}', not changed")
                continue
            if r["experiment_id"] in changed or not r.get("evidence"):
                print(f"{r['experiment_id']}: evidence missing or changed since recording; cannot be marked verified")
                continue
            r.setdefault("status_history", []).append({"from": r["status"], "to": "verified", "at": now_iso(), "by": by, "note": note})
            r["status"] = "verified"
            r["verification"] = {"method": f"reviewed by {by} against the evidence file" + (f": {note}" if note else ""), "checked_at": now_iso(), "by": by}
            done += 1
        if done:
            save(db)
            export(db)
        print(f"review: {done} records marked verified")
        return 0
    if cmd == "add" and len(argv) > 1:
        db = load()
        rec = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        for e in rec.get("evidence") or []:
            e["file"] = rel(e["file"])
            if (REPO / e["file"]).exists():
                e.setdefault("sha256", sha256(REPO / e["file"]))
        eid = append(db, rec)
        if eid:
            save(db)
            export(db)
        print("added" if eid else "already recorded (same evidence)", eid or "")
        return 0
    if cmd == "add-claim" and len(argv) > 1:
        db = load()
        c = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        n = max([int(x["claim_id"].split("_")[1]) for x in db["publication_claims"]] or [0]) + 1
        c = {"claim_id": f"CLAIM_{n:03d}", "created": now_iso(), **{k: v for k, v in c.items() if k != "claim_id"}}
        c.setdefault("status", "unverified")
        db["publication_claims"].append(c)
        probs = [x for x in validate(db) if c["claim_id"] in x]
        if probs:
            print("claim rejected:", *probs, sep="\n  ")
            return 1
        save(db)
        export(db)
        print("added", c["claim_id"])
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
