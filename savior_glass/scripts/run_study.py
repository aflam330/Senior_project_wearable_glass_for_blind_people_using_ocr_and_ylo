"""Run ONE user-study session and save everything for that participant in one place.

    python scripts/run_study.py --participant P01 --condition guided
    python scripts/run_study.py --participant P01 --condition unguided
    python scripts/run_study.py --participant P01 --condition guided --verdict      (also speak genuine / counterfeit; unreliable)
    python scripts/run_study.py --summary                                           (table of all sessions so far)

What it does:
  1. starts the glass app with the study settings (participant id, condition, watermark check on, a photo saved for
     every button press, preview window for the experimenter);
  2. when the app is closed (Q in the preview, or Ctrl+C), it summarises the session: time per answer, what the
     glass said, hand-check requests, temperature and errors;
  3. asks the experimenter for the SUS (10 items) and NASA-TLX (6 scales) answers the participant gave, and comments;
  4. saves everything under study_data/<participant>/<condition>_<time>/ and adds one line to study_data/sessions.csv
     and study_data/questionnaires.csv (the format scripts/eval/score_sus_tlx.py reads).

Saved per session: events.csv, system.csv, session.json, frames/, summary.md, labels.csv (fill in the true note /
text for accuracy), study_events.jsonl (guided-capture checks), questionnaire.json, notes.txt.
study_data/ is not committed to git: it holds photos and answers of participants.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "study_data"
sys.path.insert(0, str(ROOT / "scripts"))

SUS_ITEMS = [
    "I think that I would like to use this glass frequently.",
    "I found the glass unnecessarily complex.",
    "I thought the glass was easy to use.",
    "I think that I would need the support of a technical person to be able to use this glass.",
    "I found the various functions in this glass were well integrated.",
    "I thought there was too much inconsistency in this glass.",
    "I would imagine that most people would learn to use this glass very quickly.",
    "I found the glass very cumbersome to use.",
    "I felt very confident using the glass.",
    "I needed to learn a lot of things before I could get going with this glass.",
]
TLX_ITEMS = [("mental", "Mental demand: how mentally demanding was the task?"),
             ("physical", "Physical demand: how physically demanding was the task?"),
             ("temporal", "Temporal demand: how hurried or rushed was the pace?"),
             ("performance", "Performance: how successful were you? (0 = perfect, 100 = failure)"),
             ("effort", "Effort: how hard did you have to work?"),
             ("frustration", "Frustration: how insecure, discouraged, irritated or stressed were you?")]


def ask_number(prompt: str, lo: int, hi: int):
    while True:
        raw = input(f"{prompt} [{lo}-{hi}, Enter = skip]: ").strip()
        if raw == "":
            return None
        if raw.lstrip("-").isdigit() and lo <= int(raw) <= hi:
            return int(raw)
        print(f"  please type a whole number from {lo} to {hi}")


def questionnaire() -> dict:
    print("\n--- System Usability Scale: 1 = strongly disagree ... 5 = strongly agree ---")
    sus = [ask_number(f"SUS {i}. {q}", 1, 5) for i, q in enumerate(SUS_ITEMS, 1)]
    print("\n--- NASA-TLX: 0 = very low ... 100 = very high ---")
    tlx = {k: ask_number(f"TLX {q}", 0, 100) for k, q in TLX_ITEMS}
    comment = input("\nComments from the participant (Enter = none): ").strip()
    out = {"sus": sus, "tlx": tlx, "comment": comment}
    if all(v is not None for v in sus):
        out["sus_score"] = 2.5 * (sum(sus[i] - 1 for i in range(0, 10, 2)) + sum(5 - sus[i] for i in range(1, 10, 2)))
    if all(v is not None for v in tlx.values()):
        out["raw_tlx"] = sum(tlx.values()) / 6
    return out


def append_csv(path: Path, header: list, row: list) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def summary_table() -> None:
    p = STUDY / "sessions.csv"
    if not p.exists():
        sys.exit("no sessions yet")
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    print(f"{len(rows)} sessions, {len({r['participant'] for r in rows})} participants")
    for r in rows:
        print(f"  {r['participant']:8s} {r['condition']:9s} {r['start'][:16]}  answers {r['answers']:>3s}  watermark checks {r['watermark_checks']:>3s}  "
              f"hand-check {r['hand_check_requests']:>3s}  SUS {r['sus_score'] or '-':>5s}  TLX {r['raw_tlx'] or '-':>5s}  {r['folder']}")
    print("\nScore the questionnaires:  python ../realtime_bangla_taka_detection/scripts/eval/score_sus_tlx.py study_data/questionnaires.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant")
    ap.add_argument("--condition", choices=("guided", "unguided"), default="guided")
    ap.add_argument("--verdict", action="store_true", help="also speak genuine / counterfeit (measured unreliable on whole notes)")
    ap.add_argument("--no-questionnaire", action="store_true")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--algorithms", action="store_true", help="show the live algorithm panel in the preview")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary_table()
    if not a.participant or not a.participant.replace("_", "").replace("-", "").isalnum():
        sys.exit("give --participant, letters / digits only, for example --participant P01")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    session = STUDY / a.participant / f"{a.condition}_{stamp}"
    session.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, STUDY_PARTICIPANT=a.participant, STUDY_CONDITION=a.condition, FIELD_LOG_ENABLED="1",
               FIELD_LOG_SAVE_FRAMES="1", FIELD_LOG_DIR=str(session / "run"), WATERMARK_CHECK_ENABLED="1", PYTHONIOENCODING="utf-8",
               SHOW_PREVIEW="0" if a.no_preview else os.environ.get("SHOW_PREVIEW", "1"), SHOW_ALGORITHMS="1" if a.algorithms else "0")
    code = "import config; " + ("config.JAAL_VERDICT_ENABLED = True; " if a.verdict else "") + "import main; main.main()"
    events_log = ROOT / "logs" / "study_events.jsonl"
    before = events_log.stat().st_size if events_log.exists() else 0
    print(f"Study session: participant {a.participant}, condition {a.condition}. Close the app (Q in the preview or Ctrl+C) to finish.")
    start = dt.datetime.now()
    try:
        subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env)
    except KeyboardInterrupt:
        pass
    end = dt.datetime.now()

    # move the run's files up into the session folder
    runs = sorted((session / "run").glob("*")) if (session / "run").exists() else []
    stats = {"answers": 0, "watermark_checks": 0, "hand_check_requests": 0}
    if runs:
        run = runs[-1]
        import field_report
        s = field_report.summarise(run)
        (run / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
        (run / "summary.md").write_text(field_report.to_md(s), encoding="utf-8")
        for f in run.iterdir():
            shutil.move(str(f), str(session / f.name))
        shutil.rmtree(session / "run", ignore_errors=True)
        ev = s.get("events", {})
        stats["answers"] = ev.get("result", 0)
        stats["watermark_checks"] = ev.get("watermark_check", 0)
        print("\n" + field_report.to_md(s))
    if events_log.exists() and events_log.stat().st_size > before:   # this session's guided-capture checks
        with open(events_log, "rb") as f:
            f.seek(before)
            new = f.read().decode("utf-8", "replace")
        (session / "study_events.jsonl").write_text(new, encoding="utf-8")
        stats["hand_check_requests"] = sum(1 for line in new.splitlines() if line.strip() and json.loads(line).get("hand_check_requested"))

    q = {} if a.no_questionnaire else questionnaire()
    notes = "" if a.no_questionnaire else input("Experimenter notes for this session (Enter = none): ").strip()
    (session / "questionnaire.json").write_text(json.dumps(q, ensure_ascii=False, indent=1), encoding="utf-8")
    (session / "notes.txt").write_text(notes + "\n", encoding="utf-8")
    append_csv(STUDY / "sessions.csv",
               ["participant", "condition", "start", "end", "minutes", "answers", "watermark_checks", "hand_check_requests",
                "sus_score", "raw_tlx", "verdict_spoken", "folder"],
               [a.participant, a.condition, start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds"),
                round((end - start).total_seconds() / 60, 1), stats["answers"], stats["watermark_checks"], stats["hand_check_requests"],
                q.get("sus_score", ""), q.get("raw_tlx", ""), int(a.verdict), str(session.relative_to(STUDY))])
    if q.get("sus") and any(v is not None for v in q["sus"]):
        append_csv(STUDY / "questionnaires.csv",
                   ["participant", "condition"] + [f"sus{i}" for i in range(1, 11)] + [f"tlx_{k}" for k, _ in TLX_ITEMS] + ["comment"],
                   [a.participant, a.condition] + ["" if v is None else v for v in q["sus"]]
                   + ["" if q["tlx"][k] is None else q["tlx"][k] for k, _ in TLX_ITEMS] + [q.get("comment", "")])
    print(f"\nSaved: {session}")
    print("Next: open labels.csv in that folder and fill in the true note / text, then run")
    print(f"  python scripts/field_report.py \"{session}\"")


if __name__ == "__main__":
    main()
