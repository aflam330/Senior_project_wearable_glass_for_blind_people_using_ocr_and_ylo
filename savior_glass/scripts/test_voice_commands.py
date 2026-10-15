"""Voice commands (voice_commands.py): checks, and a measurement with synthetic speech.

    python scripts/test_voice_commands.py            # checks only (fast)
    python scripts/test_voice_commands.py --measure  # also: recognition of every command phrase, and false alarms

What --measure does, and what it is not:
  Every command phrase is spoken by the synthetic voices available on this machine (the Piper English voice, and the
  Windows voices when run on Windows), clean and with white noise added at 20 dB and 10 dB signal-to-noise ratio, and
  given to the recogniser. Then sentences that are NOT commands (some contain command words) and Bangla speech are
  given to it, and every command it reports is a false alarm.
  This is synthetic speech on a laptop. It is not people speaking into the glass's microphone in a room; that has
  not been measured. Output: results/voice_commands_synthetic.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import voice_commands as V  # noqa: E402

# ---------------------------------------------------------------- 1. parsing
assert V.parse("read text") == "mode:ocr" and V.parse("money") == "mode:currency"
assert V.parse("read text [unk]") is None and V.parse("[unk]") is None and V.parse("") is None
assert V.parse("read money") is None                     # two commands run together: not acted on
assert V.parse("glass louder", wake="glass") == "vol_up" and V.parse("louder", wake="glass") is None
assert set(V.MODE_INDEX.values()) == {config.MODE_OCR, config.MODE_OBJECT, config.MODE_CURRENCY, config.MODE_CLAUDE, config.MODE_EMOTION}
assert len(V._BY_PHRASE) == sum(len(p) for p in V.PHRASES.values()), "one phrase is listed under two commands"


# ---------------------------------------------------------------- 2. dispatch calls the button handlers
class FakeTTS:
    speaking = False

    def __init__(self):
        self.stopped = 0

    def stop_current(self):
        self.stopped += 1


class FakeApp:
    def __init__(self):
        self.mode, self.calls, self._tts = config.MODE_OCR, [], FakeTTS()

    def set_mode(self, idx):
        changed = idx != self.mode
        self.mode = idx
        self.calls.append(("mode", idx, changed))
        return changed

    def _on_action_press(self): self.calls.append("action")
    def _on_mode_press(self): self.calls.append("next")
    def _on_read_press(self): self.calls.append("read")
    def _on_vol_up(self): self.calls.append("up")
    def _on_vol_down(self): self.calls.append("down")


config.VOICE_CMD_SETTLE_S = 0.0
a = FakeApp()
V.dispatch(a, "mode:currency")
assert a.calls == [("mode", config.MODE_CURRENCY, True), "action"], a.calls        # switch, then check one note
a.calls.clear()
V.dispatch(a, "mode:currency")
assert a.calls == [("mode", config.MODE_CURRENCY, False), "action"], a.calls       # already there: just check
a.calls.clear()
V.dispatch(a, "mode:object")
assert a.calls == [("mode", config.MODE_OBJECT, True)], a.calls                    # object mode announces by itself
a.calls.clear()
for intent in ("next_mode", "action", "read", "vol_up", "vol_down", "stop"):
    V.dispatch(a, intent)
assert a.calls == ["next", "action", "read", "up", "down"] and a._tts.stopped == 1, a.calls
print("voice commands: parsing and dispatch checks pass")

why = V.available()
if why:
    print("recogniser not installed here, so recognition was not checked:", why)
    sys.exit(0)

RATE = 16000


# ---------------------------------------------------------------- synthetic voices
def to16k(pcm: bytes, rate: int) -> np.ndarray:
    x = np.frombuffer(pcm, np.int16).astype(np.float32)
    if rate != RATE:
        n = int(len(x) * RATE / rate)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return x


def piper_voice(model):
    from piper import PiperVoice
    v = PiperVoice.load(model)
    return lambda text: to16k(b"".join(c.audio_int16_bytes for c in v.synthesize(text)), v.config.sample_rate)


def sapi_voices():
    """The Windows voices, as {name: function}. Empty off Windows."""
    if sys.platform != "win32":
        return {}
    ps = "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() | % { $_.VoiceInfo.Name }"
    names = [n.strip() for n in subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout.splitlines() if n.strip()]

    def make(name):
        def say_all(texts):
            d = Path(tempfile.mkdtemp())
            lines = ["Add-Type -AssemblyName System.Speech", "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer",
                     f"$s.SelectVoice('{name}')",
                     "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)"]
            for i, t in enumerate(texts):
                lines += [f"$s.SetOutputToWaveFile('{d / f'{i}.wav'}', $f)", f"$s.Speak('{t}')"]
            lines.append("$s.SetOutputToNull()")
            script = d / "say.ps1"
            script.write_text("\n".join(lines), encoding="utf-8")
            subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)], capture_output=True)
            out = []
            for i in range(len(texts)):
                with wave.open(str(d / f"{i}.wav"), "rb") as w:
                    out.append(np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32))
            return out
        return say_all
    return {n: make(n) for n in names}


def with_noise(x: np.ndarray, snr_db, rng) -> bytes:
    """Half a second of quiet, the speech, a second of quiet; white noise over all of it at the given ratio."""
    y = np.concatenate([np.zeros(RATE // 2, np.float32), x, np.zeros(RATE, np.float32)])
    if snr_db is not None:
        p = float(np.mean(x ** 2)) or 1.0
        y = y + rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), len(y)).astype(np.float32)
    return np.clip(y, -32768, 32767).astype(np.int16).tobytes()


def hear(rec, pcm: bytes) -> list:
    rec.reset()
    out = []
    for i in range(0, len(pcm), 6400):                    # 0.2 s pieces, as the microphone delivers them
        out += rec.feed(pcm[i:i + 6400])
    return out + rec.flush()


# ---------------------------------------------------------------- 3. one spoken command is recognised
# The --measure study below keeps the settings it was first run with (no wake word, confidence 0.7), so that its
# numbers stay comparable; the glass's default since 15 October is the wake word "glass" (scripts/voice_eval.py).
rec = V.CommandRecognizer(RATE, wake="", min_conf=0.7)
voices = {}
if Path(config.PIPER_EN_MODEL).is_file():
    say = piper_voice(config.PIPER_EN_MODEL)
    voices["Piper en_US-amy"] = lambda texts, say=say: [say(t) for t in texts]
    got = hear(rec, with_noise(say("check money"), None, None))
    assert got == ["mode:currency"], (got, rec.last_heard)
    got = hear(rec, with_noise(say("I think it will rain in the afternoon"), None, None))
    assert got == [], (got, rec.last_heard)
    glass = V.CommandRecognizer(RATE)                        # the glass's default: wake word
    if glass.wake:
        got = hear(glass, with_noise(say("glass, check money"), None, None))
        assert got == ["mode:currency"], (got, glass.last_heard)
        got = hear(glass, with_noise(say("check money"), None, None))
        assert got == [], (got, glass.last_heard)          # without the wake word nothing happens
    print("voice commands: a spoken command is recognised, an ordinary sentence is not; the wake word is required")
else:
    print("no English Piper voice here: recognition not checked (python scripts/get_speech_models.py)")

if "--measure" not in sys.argv:
    sys.exit(0)

# ---------------------------------------------------------------- 4. measurement
voices.update(sapi_voices())
NOT_COMMANDS = [
    "I will read the newspaper later today", "the money is in the bank", "please open the window", "what time is it now",
    "there are many objects on the table", "she has a happy face", "can you describe the problem to me", "we should stop at the next station",
    "the volume of the box is large", "he said it again and again", "the weather is nice this morning", "my brother is coming home tomorrow",
    "how much does this cost", "the bus leaves at nine", "I need to check my phone", "let us take a picture together",
    "turn left at the corner", "the text book is on the shelf", "this is a quiet street", "good morning how are you",
    "where is the nearest hospital", "I would like a cup of tea", "the emotion in his voice was clear", "one two three four five",
    "thank you very much", "call me when you arrive", "the note was written by hand", "it is too loud in here",
    "the shop is closed on friday", "rice and fish for lunch",
]
BANGLA = ["পাঁচশত টাকার নোট সম্ভবত আসল", "ধূমপান নিষেধ", "কোনো লেখা পাওয়া যায়নি", "নোটটি আলোর সামনে ধরুন", "সামনে আছে চেয়ার এবং টেবিল",
          "একটু অপেক্ষা করুন", "আজ আবহাওয়া খুব ভালো", "আমি বাজারে যাচ্ছি", "আপনার নাম কী", "বাস কখন ছাড়বে"]
phrases = sorted(V._BY_PHRASE)
rng = np.random.default_rng(0)
res = {"date": "2026-10-14", "recogniser": Path(config.VOICE_CMD_MODEL).name, "min_conf": config.VOICE_CMD_MIN_CONF,
       "what": "synthetic voices on a laptop; not people speaking into the glass's microphone",
       "voices": list(voices), "phrases": len(phrases), "commands": {}, "false_alarms": {}, "confusions": []}
for snr in (None, 20, 10):
    key = "clean" if snr is None else f"snr_{snr}dB"
    right = wrong = missed = 0
    per_voice = {}
    for vname, say_all in voices.items():
        r = w = m = 0
        for p, x in zip(phrases, say_all(phrases)):
            got = hear(rec, with_noise(x, snr, rng))
            if got == [V._BY_PHRASE[p]]:
                r += 1
            elif not got:
                m += 1
            else:
                w += 1
                res["confusions"].append({"voice": vname, "noise": key, "said": p, "acted_on": got, "heard": rec.last_heard})
        per_voice[vname] = {"right": r, "wrong_command": w, "not_heard": m}
        right, wrong, missed = right + r, wrong + w, missed + m
    n = right + wrong + missed
    res["commands"][key] = {"n": n, "right": right, "wrong_command": wrong, "not_heard": missed,
                            "right_rate": round(right / n, 4), "per_voice": per_voice}
    print(key, res["commands"][key], flush=True)
    fa, total, which = 0, 0, []
    for vname, say_all in voices.items():
        for t, x in zip(NOT_COMMANDS, say_all(NOT_COMMANDS)):
            got = hear(rec, with_noise(x, snr, rng))
            total += 1
            if got:
                fa += 1
                which.append({"voice": vname, "said": t, "acted_on": got})
    res["false_alarms"][key] = {"english_sentences": total, "false_alarms": fa, "which": which}
    if Path(config.PIPER_BN_MODEL).is_file():
        bn = piper_voice(config.PIPER_BN_MODEL)
        got = [(t, g) for t in BANGLA for g in [hear(rec, with_noise(bn(t), snr, rng))] if g]
        res["false_alarms"][key].update({"bangla_sentences": len(BANGLA), "bangla_false_alarms": len(got), "bangla_which": got})
    print(key, "false alarms", {k: v for k, v in res["false_alarms"][key].items() if "which" not in k}, flush=True)
# noise alone: 60 s of white noise at a moderate level must not produce a command
got = hear(rec, rng.normal(0, 1500, RATE * 60).astype(np.int16).tobytes())
res["false_alarms"]["noise_only_60s"] = len(got)
out = Path(config.BASE_DIR) / "results" / "voice_commands_synthetic.json"
out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
print("saved", out)
