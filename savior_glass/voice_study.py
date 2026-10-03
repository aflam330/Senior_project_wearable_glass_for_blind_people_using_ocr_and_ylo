"""Voice-driven user study: the glass asks in Bangla, the participant answers by voice, everything is saved by ID.

Flow (STUDY_VOICE=1, or scripts/run_voice_study.py):
  1. At start the glass asks the participant's name, records the answer and gives the participant the next free ID
     (P001, P002, ...).
  2. The participant uses the glass. Every button press is already logged by field_log.py.
  3. The experimenter presses D (done). The glass asks six short yes / no questions in Bangla, one at a time, and
     records each spoken answer. "হ্যাঁ" / "না" (and common variants) are recognised; an unclear answer is asked once more.
  4. Answers are saved automatically. N starts the next participant.

Saved under study_data/voice/ (never committed; it holds voices and names):
  responses.csv               one row per participant: id, name, start, end, minutes, the six answers, field-run folder
  <ID>/answers.json           the same, plus what the recogniser heard for each answer
  <ID>/name.wav, q1.wav ...   the recordings themselves, so an unclear answer can be checked by ear

Speech recognition: Google Web Speech (bn-BD) through the SpeechRecognition package; needs internet. Without internet
the recordings are still saved and the answers are marked "unclear". The experimenter can also press Y or N while a
question is being asked.

The six questions cover what usability papers report for assistive devices: ease of use, clarity of the spoken
output, usefulness for the two main tasks, trust, and intention to use. They are yes / no to keep the session short;
this is not the 10-item SUS (use scripts/run_study.py when the full questionnaire is wanted).
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import os
import threading
import time
import wave

import numpy as np

import config

logger = logging.getLogger("smart_glass.voice_study")

RATE = 16000
ROOT = os.path.join(config.BASE_DIR, "study_data", "voice")
QUESTIONS = [
    ("easy_to_use", "চশমাটি ব্যবহার করা কি সহজ ছিল?"),                       # Was the glass easy to use?
    ("speech_clear", "চশমার কথা কি স্পষ্ট বোঝা গেছে?"),                        # Was the glass's speech clear?
    ("helped_currency", "টাকার নোট চিনতে চশমাটি কি আপনাকে সাহায্য করেছে?"),      # Did it help you recognise notes?
    ("helped_reading", "লেখা পড়তে চশমাটি কি আপনাকে সাহায্য করেছে?"),            # Did it help you read text?
    ("trust", "চশমার উত্তরের উপর কি আপনি ভরসা করতে পারেন?"),                   # Can you trust its answers?
    ("would_use_daily", "আপনি কি প্রতিদিন এই চশমা ব্যবহার করতে চাইবেন?"),        # Would you use it every day?
]
YES = ("হ্যাঁ", "হ্যা", "হাঁ", "হা", "জি", "জ্বি", "জী", "ঠিক", "অবশ্যই", "yes", "yeah", "ha", "ji")
NO = ("না", "নাহ", "নয়", "নাই", "no", "nope", "na")


def parse_yes_no(text: str):
    """'yes', 'no' or None from what the recogniser heard. 'না' is checked first: 'হ্যাঁ না' style answers are rare,
    and 'না' inside a longer word (e.g. 'জানা') must not count, so whole words only."""
    words = [w.strip("।,.!?").lower() for w in (text or "").split()]
    if any(w in NO for w in words):
        return "no"
    if any(w in YES for w in words):
        return "yes"
    return None


class VoiceStudy:
    def __init__(self, app) -> None:
        self.app = app
        self.pid = None
        self.name = ""
        self.status = "study: not started"
        self.last_heard = ""
        self.key_answer = None          # set by the preview when the experimenter presses Y or N
        self._busy = threading.Lock()
        self._start = None
        os.makedirs(ROOT, exist_ok=True)

    # ---- public (called from the preview keys) ----
    def new_participant(self) -> None:
        self._spawn(self._enrol)

    def finish(self) -> None:
        if self.pid is None:
            self._spawn(self._enrol)
        else:
            self._spawn(self._ask_questions)

    def _spawn(self, fn) -> None:
        if self._busy.locked():
            return
        threading.Thread(target=self._guard, args=(fn,), daemon=True, name="voice-study").start()

    def _guard(self, fn) -> None:
        with self._busy:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                logger.warning("voice study step failed: %s", exc)
                self.status = f"study error: {exc}"

    # ---- audio ----
    def _say(self, text: str) -> None:
        self.app._tts.speak(text)
        time.sleep(0.3)
        self.app._tts.drain(timeout=25)
        time.sleep(0.25)

    def _beep(self) -> None:
        try:
            import winsound
            winsound.Beep(880, 180)
        except Exception:  # noqa: BLE001  (not Windows: no beep)
            time.sleep(0.1)

    def _record(self, seconds: float, path: str) -> np.ndarray:
        import sounddevice as sd
        self._beep()
        audio = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype="int16")
        t_end = time.time() + seconds
        while time.time() < t_end and self.key_answer is None:
            time.sleep(0.05)
        if self.key_answer is not None:
            sd.stop()
        else:
            sd.wait()
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(audio.tobytes())
        return audio

    def _transcribe(self, audio: np.ndarray) -> str:
        try:
            import speech_recognition as sr
            return sr.Recognizer().recognize_google(sr.AudioData(audio.tobytes(), RATE, 2), language="bn-BD")
        except Exception as exc:  # noqa: BLE001  (nothing understood, or no internet)
            logger.info("speech not recognised: %s", type(exc).__name__)
            return ""

    # ---- steps ----
    def _next_id(self) -> str:
        path = os.path.join(ROOT, "responses.csv")
        nums = [0]
        if os.path.exists(path):
            with open(path, encoding="utf-8-sig", newline="") as f:
                nums += [int(r["id"][1:]) for r in csv.DictReader(f) if r.get("id", "")[1:].isdigit()]
        nums += [int(d[1:]) for d in os.listdir(ROOT) if d.startswith("P") and d[1:].isdigit()]
        return f"P{max(nums) + 1:03d}"

    def _enrol(self) -> None:
        self.pid = self._next_id()
        folder = os.path.join(ROOT, self.pid)
        os.makedirs(folder, exist_ok=True)
        self.status = f"{self.pid}: asking the name"
        self._say("আপনার নাম বলুন।")                                   # please say your name
        self.status = f"{self.pid}: LISTENING for the name"
        audio = self._record(4.0, os.path.join(folder, "name.wav"))
        self.name = self._transcribe(audio)
        self.last_heard = self.name
        self._start = dt.datetime.now()
        self.status = f"{self.pid} {self.name or '(name not recognised; saved as audio)'}: using the glass. Press D when done"
        digits = " ".join(str(int(c)) for c in self.pid[1:])
        self._say(f"ধন্যবাদ {self.name}। আপনার নম্বর {digits}। এখন চশমা ব্যবহার করুন।")   # thank you; your number; now use the glass
        self._save(folder, {})   # the participant exists on disk from this moment, even if the session is cut short

    def _ask_questions(self) -> None:
        folder = os.path.join(ROOT, self.pid)
        answers, heard = {}, {}
        self._say("কয়েকটি ছোট প্রশ্ন। হ্যাঁ অথবা না বলুন।")                    # a few short questions; say yes or no
        for i, (key, question) in enumerate(QUESTIONS, 1):
            ans = None
            for attempt in (1, 2):
                self.key_answer = None
                self.status = f"{self.pid}: question {i} of {len(QUESTIONS)} (asking)"
                self._say(question if attempt == 1 else "বুঝতে পারিনি। হ্যাঁ অথবা না বলুন। " + question)
                self.status = f"{self.pid}: question {i} of {len(QUESTIONS)} LISTENING (or press Y / N)"
                audio = self._record(3.5, os.path.join(folder, f"q{i}_{key}{'' if attempt == 1 else '_retry'}.wav"))
                if self.key_answer is not None:
                    ans, text = self.key_answer, f"(key {self.key_answer})"
                else:
                    text = self._transcribe(audio)
                    ans = parse_yes_no(text)
                heard[key] = text
                self.last_heard = f"{text or '(nothing recognised)'} -> {ans or 'unclear'}"
                if ans:
                    break
            answers[key] = ans or "unclear"
        self.key_answer = None
        self._save(folder, answers, heard, done=True)
        self.status = f"{self.pid} saved. Press N for the next participant"
        self._say("ধন্যবাদ। আপনার উত্তর সংরক্ষণ করা হয়েছে।")                    # thank you; your answers are saved
        self.pid, self.name = None, ""

    def _save(self, folder: str, answers: dict, heard: dict | None = None, done: bool = False) -> None:
        end = dt.datetime.now()
        run_dir = ""
        try:
            import field_log
            run_dir = os.path.basename(field_log.get().dir or "")
        except Exception:  # noqa: BLE001
            pass
        rec = {"id": self.pid, "name": self.name, "start": self._start.isoformat(timespec="seconds") if self._start else "",
               "end": end.isoformat(timespec="seconds") if done else "", "complete": done,
               "minutes": round((end - self._start).total_seconds() / 60, 1) if (done and self._start) else "",
               "answers": answers, "heard": heard or {}, "field_run": run_dir,
               "questions": {k: q for k, q in QUESTIONS}}
        with open(os.path.join(folder, "answers.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        if not done:
            return
        path = os.path.join(ROOT, "responses.csv")
        cols = ["id", "name", "start", "end", "minutes"] + [k for k, _ in QUESTIONS] + ["field_run"]
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if new:
                w.writerow(cols)
            w.writerow([rec["id"], rec["name"], rec["start"], rec["end"], rec["minutes"]] + [answers.get(k, "") for k, _ in QUESTIONS] + [run_dir])
        try:
            import main
            main._flog("study_answers", result=json.dumps(answers, ensure_ascii=False), participant=self.pid)
        except Exception:  # noqa: BLE001
            pass
