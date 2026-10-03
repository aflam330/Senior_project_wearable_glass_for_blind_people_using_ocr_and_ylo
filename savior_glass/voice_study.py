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
YES = ("হ্যাঁ", "হ্যা", "হাঁ", "হা", "হ্যাঁা", "হ্যান", "হুম", "হু", "হ্যাঁগো", "জি", "জ্বি", "জী", "জ্বী", "ঠিক", "অবশ্যই", "আচ্ছা",
       "ইয়েস", "ইয়াস", "ইয়েশ", "ইয়েছ", "ইয়া", "ওকে", "yes", "yeah", "yah", "yep", "ya", "ok", "okay", "ha", "han", "haan", "ji")
NO = ("না", "নাহ", "নয়", "নাই", "নো", "নোপ", "নাহি", "no", "nope", "na", "nah", "not")


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
        self.level = 0.0                # live microphone level while listening (0..1), drawn by the preview
        self.spoke = False              # whether the last recording contained speech above the room noise
        self.clipped = False            # the last recording hit full scale (input level too high)
        self.rate = RATE
        self.mic_name = ""
        self._mic_cfg = None
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

    def _mic(self):
        """(device index, sample rate, channels). STUDY_MIC=<part of the name> picks a microphone (e.g. a headset).
        On Windows the WASAPI entry of the microphone is used at its own sample rate: the default (MME) entry at
        16 kHz cut speech off on the test laptop."""
        import sounddevice as sd
        if self._mic_cfg is not None:
            return self._mic_cfg
        devs, apis = sd.query_devices(), sd.query_hostapis()
        want = os.environ.get("STUDY_MIC", "").strip().lower()
        default = sd.query_devices(kind="input")
        key = want or default["name"][:16].lower()
        pick = None
        for i, d in enumerate(devs):
            if d["max_input_channels"] > 0 and "WASAPI" in apis[d["hostapi"]]["name"] and key in d["name"].lower():
                pick = (i, int(d["default_samplerate"]), min(2, d["max_input_channels"]))
                break
        if pick is None and want:
            for i, d in enumerate(devs):
                if d["max_input_channels"] > 0 and want in d["name"].lower():
                    pick = (i, int(d["default_samplerate"]), min(2, d["max_input_channels"]))
                    break
        if pick is None:
            pick = (None, int(default["default_samplerate"]), min(2, default["max_input_channels"]))
        self._mic_cfg = pick
        self.mic_name = devs[pick[0]]["name"] if pick[0] is not None else default["name"]
        logger.info("study microphone: %s, %d Hz", self.mic_name, pick[1])
        return pick

    def _record(self, max_seconds: float, path: str) -> np.ndarray:
        """Listen until the person has spoken and stopped (or max_seconds). Returns mono float audio at self.rate,
        trimmed to the speech and level-normalised; the untrimmed recording is saved to `path`."""
        import sounddevice as sd
        dev, rate, ch = self._mic()
        self.rate = rate
        block = rate // 20                      # 50 ms
        chunks, levels = [], []
        started, quiet_blocks, loud_blocks = False, 0, 0
        self._beep()
        with sd.InputStream(device=dev, samplerate=rate, channels=ch, dtype="float32", blocksize=block) as stream:
            noise = None
            t_end = time.time() + max_seconds
            while time.time() < t_end and self.key_answer is None:
                data, _ = stream.read(block)
                x = data.mean(axis=1)
                chunks.append(x)
                lvl = float(np.abs(x).mean())
                levels.append(lvl)
                self.level = min(1.0, lvl * 12)   # shown as a bar in the preview
                if len(levels) == 6:              # first 0.3 s: the room's own noise
                    noise = max(float(np.median(levels)), 1e-4)
                if noise is None:
                    continue
                loud = lvl > max(noise * 3.5, 0.012)
                if loud:
                    loud_blocks += 1
                    quiet_blocks = 0
                    if loud_blocks >= 4:          # 0.2 s of sound: a voice, not a click or a key press
                        started = True
                else:
                    quiet_blocks += 1
                    if not started and quiet_blocks >= 6:
                        loud_blocks = 0           # a short noise: forget it and keep waiting
                    if started and quiet_blocks >= 20:   # 1 s of quiet after speech: the answer is finished
                        break
        self.level = 0.0
        audio = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        self.spoke = started
        self.clipped = bool(len(audio)) and float((np.abs(audio) > 0.985).mean()) > 0.002   # input level too high: distorted
        if self.clipped:
            logger.info("microphone is clipping: lower the input level in Windows sound settings or move back a little")
        if started and noise is not None:      # keep the speech with a little margin, then bring it to a normal level
            lv = np.array(levels)
            idx = np.where(lv > max(noise * 3.5, 0.012))[0]
            a, b = max(0, (idx[0] - 6) * block), min(len(audio), (idx[-1] + 8) * block)
            audio = audio[a:b]
        audio = audio - audio.mean()
        return audio / max(float(np.percentile(np.abs(audio), 99.9)), 1e-6) * 0.8

    def _transcribe(self, audio: np.ndarray, yes_no: bool = False) -> list:
        """Every guess Google returns (best first), so a yes / no hidden in a lower-ranked guess is still found.
        For yes / no answers a second pass in English is made when the Bangla pass has no yes or no in it."""
        import speech_recognition as sr
        data = sr.AudioData((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes(), self.rate, 2)
        out = []
        for lang in ("bn-BD", "en-US") if yes_no else ("bn-BD",):
            try:
                res = sr.Recognizer().recognize_google(data, language=lang, show_all=True)
                out += [a.get("transcript", "") for a in (res.get("alternative") if isinstance(res, dict) else []) or [] if a.get("transcript")]
            except Exception as exc:  # noqa: BLE001  (no internet, or the service refused)
                logger.info("speech not recognised (%s): %s", lang, type(exc).__name__)
            if any(parse_yes_no(t) for t in out):
                break
        return out

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
        self.name = ""
        for attempt in (1, 2):
            self.status = f"{self.pid}: asking the name"
            self._say("আপনার নাম বলুন।" if attempt == 1 else "শুনতে পাইনি। একটু জোরে আপনার নাম বলুন।")   # please say your name / louder please
            self.status = f"{self.pid}: LISTENING for the name - speak now"
            audio = self._record(7.0, os.path.join(folder, "name.wav" if attempt == 1 else "name_retry.wav"))
            guesses = self._transcribe(audio) if self.spoke else []
            self.name = guesses[0] if guesses else ""
            self.last_heard = self.name or ("(no speech heard)" if not self.spoke else "(speech not understood)")
            if self.name:
                break
        self._start = dt.datetime.now()
        self.status = f"{self.pid} {self.name or '(name not recognised; saved as audio)'}: using the glass. Press D when done"
        digits = " ".join(str(int(c)) for c in self.pid[1:])
        self._say(f"ধন্যবাদ {self.name}। আপনার নম্বর {digits}। এখন চশমা ব্যবহার করুন।")   # thank you; your number; now use the glass
        self._save(folder, {})   # the participant exists on disk from this moment, even if the session is cut short

    def _ask_questions(self) -> None:
        folder = os.path.join(ROOT, self.pid)
        answers, heard, sources = {}, {}, {}
        self._say("কয়েকটি ছোট প্রশ্ন। হ্যাঁ অথবা না বলুন।")                    # a few short questions; say yes or no
        for i, (key, question) in enumerate(QUESTIONS, 1):
            ans = None
            for attempt in (1, 2):
                self.key_answer = None
                self.status = f"{self.pid}: question {i} of {len(QUESTIONS)} (asking)"
                self._say(question if attempt == 1 else "বুঝতে পারিনি। হ্যাঁ অথবা না বলুন। " + question)
                self.status = f"{self.pid}: question {i} of {len(QUESTIONS)} LISTENING - say হ্যাঁ or না (or press Y / N)"
                audio = self._record(8.0, os.path.join(folder, f"q{i}_{key}{'' if attempt == 1 else '_retry'}.wav"))
                if self.key_answer is not None:
                    ans, text, source = self.key_answer, f"(key {self.key_answer})", "key"
                else:
                    guesses = self._transcribe(audio, yes_no=True) if self.spoke else []
                    ans = next((a for a in (parse_yes_no(g) for g in guesses) if a), None)
                    text = " | ".join(guesses[:3]) if guesses else ("(no speech heard)" if not self.spoke else "(speech not understood)")
                    source = "voice"
                heard[key] = text
                sources[key] = source if ans else "none"
                self.last_heard = f"{text} -> {ans or 'unclear'}" + ("  [mic too loud: clipping]" if self.clipped else "")
                if ans:
                    break
            answers[key] = ans or "unclear"
        self.key_answer = None
        self._save(folder, answers, heard, done=True, sources=sources)
        self.status = f"{self.pid} saved. Press N for the next participant"
        self._say("ধন্যবাদ। আপনার উত্তর সংরক্ষণ করা হয়েছে।")                    # thank you; your answers are saved
        self.pid, self.name = None, ""

    def _save(self, folder: str, answers: dict, heard: dict | None = None, done: bool = False, sources: dict | None = None) -> None:
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
               "answers": answers, "heard": heard or {}, "answered_by": sources or {}, "microphone": self.mic_name, "field_run": run_dir,
               "questions": {k: q for k, q in QUESTIONS}}
        with open(os.path.join(folder, "answers.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        if not done:
            return
        path = os.path.join(ROOT, "responses.csv")
        cols = ["id", "name", "start", "end", "minutes"] + [k for k, _ in QUESTIONS] + ["answered_by_voice", "field_run"]
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if new:
                w.writerow(cols)
            w.writerow([rec["id"], rec["name"], rec["start"], rec["end"], rec["minutes"]] + [answers.get(k, "") for k, _ in QUESTIONS]
                       + [sum(v == "voice" for v in (sources or {}).values()), run_dir])
        try:
            import main
            main._flog("study_answers", result=json.dumps(answers, ensure_ascii=False), participant=self.pid)
        except Exception:  # noqa: BLE001
            pass
