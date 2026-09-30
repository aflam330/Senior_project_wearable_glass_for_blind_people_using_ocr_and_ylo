"""Analysis of the glass study log (savior_glass/logs/study_events.jsonl) for the capture-guidance study.

Pre-registered in paper_evidence/STUDY_PREREGISTRATION.md. The glass does not know which note it was
shown, so the experimenter keeps a trial sheet (trial_sheet_template.csv):
  participant, condition, trial, note_id, print_id, true_denomination, true_label, spoken_denomination
Events are matched to trials in time order within participant x condition. A mismatch in counts stops
the analysis instead of guessing.

Primary outcomes, per participant and condition:
  median seconds per watermark check;
  hand-check rate on GENUINE notes (asked to check by hand, or timed out).
Safety outcome (reported, not tested): counterfeit notes answered "likely genuine" (must be 0).
Secondary: denomination accuracy (spoken vs true), frames rejected per check, timeouts.
Test: two-sided Wilcoxon signed-rank, guided vs unguided, over participants; Holm over the two primaries.
Usage: python analyze_study_events.py study_events.jsonl trial_sheet.csv [out.json]
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from statistics import median


def load(events_path, sheet_path):
    ev = defaultdict(list)
    for line in open(events_path, encoding="utf-8"):
        if line.strip():
            e = json.loads(line)
            ev[(e["participant"], e["condition"])].append(e)
    tr = defaultdict(list)
    with open(sheet_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tr[(r["participant"], r["condition"])].append(r)
    rows = []
    for key in sorted(tr):
        t = sorted(tr[key], key=lambda r: int(r["trial"]))
        e = sorted(ev.get(key, []), key=lambda x: x["at"])
        if len(t) != len(e):
            raise SystemExit(f"{key}: {len(t)} trials on the sheet but {len(e)} logged checks; fix the sheet first")
        rows += [{**a, **{"event": b}} for a, b in zip(t, e)]
    return rows


def summarise(rows):
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        p = per[(r["participant"], r["condition"])]
        e = r["event"]
        p["seconds"].append(e["seconds"])
        p["rejected"].append(sum(e["rejected"].values()))
        p["timeout"].append(e["outcome"] == "timeout")
        if r["spoken_denomination"]:
            p["denom_correct"].append(r["spoken_denomination"] == r["true_denomination"])
        if r["true_label"] == "genuine":
            p["hand_check_genuine"].append(e["hand_check_requested"])
        else:
            p["counterfeit_passed"].append(e["outcome"] == "likely_genuine")
    mean = lambda v: sum(v) / len(v) if v else None
    return {k: {"median_seconds": median(v["seconds"]), "hand_check_rate_genuine": mean(v["hand_check_genuine"]),
                "counterfeit_passed": sum(v["counterfeit_passed"]), "counterfeit_n": len(v["counterfeit_passed"]),
                "denomination_accuracy": mean(v["denom_correct"]), "mean_frames_rejected": mean(v["rejected"]),
                "timeouts": sum(v["timeout"]), "checks": len(v["seconds"])} for k, v in per.items()}


def main() -> None:
    rows = load(sys.argv[1], sys.argv[2])
    s = summarise(rows)
    out = {"per_participant_condition": {f"{p}|{c}": v for (p, c), v in s.items()},
           "safety_counterfeit_passed_total": sum(v["counterfeit_passed"] for v in s.values())}
    people = sorted({p for p, _ in s if (p, "guided") in s and (p, "unguided") in s})
    out["paired_participants"] = len(people)
    if len(people) >= 6:
        from scipy.stats import wilcoxon
        tests = {}
        for m in ("median_seconds", "hand_check_rate_genuine"):
            a = [s[(p, "guided")][m] for p in people]
            b = [s[(p, "unguided")][m] for p in people]
            ok = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
            d = [x - y for x, y in ok]
            tests[m] = {"n": len(ok), "median_diff_guided_minus_unguided": median(d) if d else None,
                        "p": float(wilcoxon([x for x, _ in ok], [y for _, y in ok]).pvalue) if any(d) else 1.0}
        ps = sorted(tests, key=lambda k: tests[k]["p"])
        for i, k in enumerate(ps):  # Holm
            tests[k]["p_holm"] = min(1.0, max(tests[q]["p"] * (len(ps) - j) for j, q in enumerate(ps[:i + 1])))
        out["tests"] = tests
    text = json.dumps(out, indent=1, ensure_ascii=False)
    if len(sys.argv) > 3:
        open(sys.argv[3], "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
