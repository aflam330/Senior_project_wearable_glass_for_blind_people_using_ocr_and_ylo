"""Summarise one field-test run (a folder written by field_log.py). Standard library only; fine to run on the Pi.

  python scripts/field_report.py logs/field/<run>          # summary.md + summary.json, and labels.csv if missing
  python scripts/field_report.py logs/field/<run> --all    # also every run under logs/field, side by side

labels.csv has one row per answer the glass gave (currency, OCR, watermark check, Claude) with empty columns
true_denomination, true_label (genuine / counterfeit), true_text and note. The tester fills them in (from
memory, the trial sheet, or the saved frames/<event_id>.jpg). Run the script again and the summary adds:
  - denomination accuracy (currency),
  - counterfeit safety: counterfeit notes the glass called likely genuine (must be 0), and how often
    genuine notes were confirmed versus sent to a hand check,
  - OCR character error rate against true_text.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

LABEL_COLS = ["event_id", "time", "mode", "event", "result", "denomination", "verdict", "score", "frame",
              "true_denomination", "true_label", "true_text", "note"]


def rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pct(values, q):
    v = sorted(values)
    if not v:
        return None
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm(s):
    return " ".join(unicodedata.normalize("NFC", s or "").split())


def ensure_labels(run: Path, ev):
    lab = run / "labels.csv"
    if lab.exists():
        return
    with open(lab, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(LABEL_COLS)
        for r in ev:
            if r["event"] not in ("result", "watermark_check"):
                continue
            d = json.loads(r["detail"]) if r.get("detail") else {}
            press = d.get("press_event")
            frame = f"frames/{int(press):06d}.jpg" if press and (run / "frames" / f"{int(press):06d}.jpg").exists() else ""
            text = d.get("ocr_text") or r["result"]
            w.writerow([r["event_id"], r["time"], r["mode"], r["event"], text, r["denomination"], r["verdict"], r["score"],
                        frame, "", "", "", ""])


def summarise(run: Path) -> dict:
    ev = rows(run / "events.csv")
    sysr = rows(run / "system.csv") if (run / "system.csv").exists() else []
    session = json.loads((run / "session.json").read_text(encoding="utf-8")) if (run / "session.json").exists() else {}
    ensure_labels(run, ev)
    out = {"run": run.name, "start": session.get("start"), "end": session.get("end"), "device": session.get("device_model"),
           "events": dict(Counter(r["event"] for r in ev)), "settings": session.get("settings", {})}
    t = [num(r["t_s"]) for r in ev if num(r["t_s"]) is not None]
    out["duration_min"] = round((max(t) - min(t)) / 60, 1) if t else 0
    lat = defaultdict(list)
    for r in ev:
        if r["event"] in ("result", "object_announce", "watermark_check") and num(r["latency_ms"]) is not None:
            lat[f"{r['mode']}:{r['event']}"].append(num(r["latency_ms"]))
    out["latency_ms"] = {k: {"n": len(v), "median": round(statistics.median(v), 1), "p95": round(pct(v, 0.95), 1),
                             "max": round(max(v), 1)} for k, v in lat.items()}
    cur = [r for r in ev if r["event"] == "result" and r["mode"] == "currency"]
    out["currency_answers"] = {"n": len(cur), "denominations": dict(Counter(r["denomination"] for r in cur)),
                               "verdicts": dict(Counter(r["verdict"] for r in cur if r["verdict"]))}
    wm = [r for r in ev if r["event"] == "watermark_check"]
    out["watermark_checks"] = {"n": len(wm), "outcomes": dict(Counter(r["verdict"] for r in wm))}
    out["errors"] = [r["result"] for r in ev if r["event"] == "error"][:20]
    out["busy_presses_ignored"] = sum(r["event"] == "action_ignored_busy" for r in ev)
    temps = [num(r["cpu_temp_c"]) for r in sysr if num(r["cpu_temp_c"]) is not None]
    out["system"] = {"samples": len(sysr), "max_temp_c": max(temps) if temps else None,
                     "throttled_values": sorted({r["throttled_hex"] for r in sysr if r["throttled_hex"]}),
                     "min_mem_available_mb": min((num(r["mem_available_mb"]) for r in sysr if num(r["mem_available_mb"]) is not None), default=None),
                     "max_app_rss_mb": max((num(r["app_rss_mb"]) for r in sysr if num(r["app_rss_mb"]) is not None), default=None),
                     "max_cpu_busy_pct": max((num(r["cpu_busy_pct"]) for r in sysr if num(r["cpu_busy_pct"]) is not None), default=None),
                     "events_dropped": sysr[-1]["events_dropped"] if sysr else None}
    out["accuracy"] = accuracy(run)
    return out


def accuracy(run: Path) -> dict:
    lab = [r for r in rows(run / "labels.csv") if any(r[c].strip() for c in ("true_denomination", "true_label", "true_text"))]
    res = {"labelled_rows": len(lab)}
    den = [r for r in lab if r["mode"] == "currency" and r["event"] == "result" and r["true_denomination"].strip()]
    if den:
        ok = sum(r["denomination"].strip() == r["true_denomination"].strip() for r in den)
        res["denomination"] = {"n": len(den), "correct": ok, "accuracy": round(ok / len(den), 4)}
    safe = [r for r in lab if r["mode"] == "currency" and r["true_label"].strip() in ("genuine", "counterfeit") and r["verdict"]]
    if safe:
        passed = lambda r: r["verdict"] in ("likely_genuine", "genuine")
        cf = [r for r in safe if r["true_label"].strip() == "counterfeit"]
        gn = [r for r in safe if r["true_label"].strip() == "genuine"]
        res["counterfeit_safety"] = {"counterfeit_n": len(cf), "counterfeit_passed_as_genuine": sum(passed(r) for r in cf),
                                     "genuine_n": len(gn), "genuine_confirmed": sum(passed(r) for r in gn),
                                     "genuine_sent_to_hand_check": sum(not passed(r) for r in gn)}
    ocr = [r for r in lab if r["mode"] == "ocr" and r["true_text"].strip()]
    if ocr:
        cers = [lev(norm(r["true_text"]), norm(r["result"])) / max(len(norm(r["true_text"])), 1) for r in ocr]
        res["ocr"] = {"n": len(ocr), "mean_cer": round(sum(cers) / len(cers), 4), "exact": sum(c == 0 for c in cers)}
    return res


def to_md(s: dict) -> str:
    L = [f"# Field run {s['run']}", "", f"Device: {s['device']}  ·  start {s['start']}  ·  end {s['end']}  ·  {s['duration_min']} min", "",
         "## Events", "", "| Event | Count |", "|---|---:|"]
    L += [f"| {k} | {v} |" for k, v in sorted(s["events"].items())]
    L += ["", "## Latency (ms)", "", "| Mode : event | n | Median | 95th pct | Max |", "|---|---:|---:|---:|---:|"]
    L += [f"| {k} | {v['n']} | {v['median']} | {v['p95']} | {v['max']} |" for k, v in sorted(s["latency_ms"].items())]
    sy = s["system"]
    L += ["", "## System", "", f"- Max CPU temperature: {sy['max_temp_c']} °C; throttled flags seen: {sy['throttled_values'] or 'none'}",
          f"- Lowest free memory: {sy['min_mem_available_mb']} MB; app memory peak: {sy['max_app_rss_mb']} MB; busiest CPU sample: {sy['max_cpu_busy_pct']} %",
          f"- Log events dropped: {sy['events_dropped']}; presses ignored while busy: {s['busy_presses_ignored']}", "",
          "## Answers", "", f"- Currency: {s['currency_answers']}", f"- Watermark checks: {s['watermark_checks']}"]
    a = s["accuracy"]
    L += ["", "## Accuracy (from labels.csv)", ""]
    if a["labelled_rows"] == 0:
        L.append("No labels yet. Fill in true_denomination / true_label / true_text in labels.csv and run this script again.")
    else:
        for k in ("denomination", "counterfeit_safety", "ocr"):
            if k in a:
                L.append(f"- {k}: {a[k]}")
    if s["errors"]:
        L += ["", "## Errors (first 20)", ""] + [f"- {e}" for e in s["errors"]]
    return "\n".join(L) + "\n"


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    run = Path(sys.argv[1])
    runs = sorted(p for p in run.parent.iterdir() if (p / "events.csv").exists()) if "--all" in sys.argv else [run]
    for r in runs:
        s = summarise(r)
        (r / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
        (r / "summary.md").write_text(to_md(s), encoding="utf-8")
        print(to_md(s))


if __name__ == "__main__":
    main()
