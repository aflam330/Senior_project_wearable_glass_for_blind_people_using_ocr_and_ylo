"""Start the glass for a voice-driven user study (see voice_study.py).

    python scripts/run_voice_study.py              # the study as participants should get it
    python scripts/run_voice_study.py --verdict    # also speak genuine / counterfeit (unreliable on whole notes; demo only)
    python scripts/run_voice_study.py --results    # print the answers saved so far
    python scripts/run_voice_study.py --mics       # list the microphones; choose one with STUDY_MIC=<part of its name>

In the preview window: the glass asks the participant's name by itself. The participant uses the glass
(M mode, A action, R read). Press D when they are done: six yes / no questions are asked in Bangla and answered by
voice (or Y / N keys). Press N for the next participant, Q to quit. Needs a microphone, and internet for the
Bangla speech recognition. Everything is saved in study_data/voice/ automatically.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOICE = ROOT / "study_data" / "voice"


def results() -> None:
    p = VOICE / "responses.csv"
    if not p.exists():
        sys.exit("no answers saved yet")
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    keys = [k for k in rows[0] if k not in ("id", "name", "start", "end", "minutes", "field_run")]
    print(f"{len(rows)} participants")
    for r in rows:
        print(f"  {r['id']}  {r['name'] or '(name in name.wav)':20s} {r['minutes']:>5s} min  " + "  ".join(f"{k}={r[k]}" for k in keys))
    print()
    for k in keys:
        yes = sum(r[k] == "yes" for r in rows)
        no = sum(r[k] == "no" for r in rows)
        print(f"  {k:18s} yes {yes:3d}  no {no:3d}  unclear {len(rows) - yes - no:3d}   ({100 * yes / max(yes + no, 1):.0f} % yes of clear answers)")


def main() -> None:
    if "--results" in sys.argv:
        return results()
    if "--mics" in sys.argv:
        import sounddevice as sd
        apis = sd.query_hostapis()
        print("Microphones (start with  STUDY_MIC=<part of the name>  to choose one):")
        for d in sd.query_devices():
            if d["max_input_channels"] > 0 and "WASAPI" in apis[d["hostapi"]]["name"]:
                print("  ", d["name"])
        return
    os.environ.setdefault("CAMERA_WIDTH", "1280")
    os.environ.setdefault("CAMERA_HEIGHT", "720")
    env = dict(os.environ, STUDY_VOICE="1", FIELD_LOG_ENABLED="1", FIELD_LOG_SAVE_FRAMES="1", FIELD_LOG_DIR=str(VOICE / "_runs"),
               WATERMARK_CHECK_ENABLED="1", SHOW_PREVIEW="1", PYTHONIOENCODING="utf-8")
    code = "import config; " + ("config.JAAL_VERDICT_ENABLED = True; " if "--verdict" in sys.argv else "") + "import main; main.main()"
    subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env)
    if (VOICE / "responses.csv").exists():
        results()


if __name__ == "__main__":
    main()
