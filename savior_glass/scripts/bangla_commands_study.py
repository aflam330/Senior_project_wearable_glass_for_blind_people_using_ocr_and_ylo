"""Can the glass take Bangla voice commands offline? A study with Whisper (faster-whisper, CTranslate2, 8-bit, CPU).

Vosk, used for the English commands, has no Bangla model. Whisper's multilingual models include Bengali. Here the
"tiny" and "base" models transcribe Bangla command phrases, and the transcript is matched to the closest command
phrase by letter similarity (1 - edit distance / length).

Rule, fixed before the run (2026-10-14): a command is accepted when its best similarity is at least 0.6; nothing
else is tuned. Voices: the Piper Bangla voice (offline) and Google's Bangla voice (gTTS, online; only these phrases
are sent). Non-command Bangla sentences measure false commands. Synthetic voices on a laptop, not people.

    python scripts/bangla_commands_study.py
Output: results/bangla_voice_commands.json
"""
from __future__ import annotations

import io
import json
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

BN_COMMANDS = {
    "mode:ocr": ["লেখা পড়ো", "পড়ো"],
    "mode:currency": ["টাকা দেখো", "টাকা"],
    "mode:object": ["সামনে কী আছে", "বস্তু দেখো"],
    "mode:claude": ["বর্ণনা করো"],
    "mode:emotion": ["মুখ দেখো", "অনুভূতি"],
    "next_mode": ["পরের মোড"],
    "action": ["ছবি তোলো"],
    "read": ["আবার বলো"],
    "vol_up": ["জোরে"],
    "vol_down": ["আস্তে"],
    "stop": ["থামো"],
}
NOT_COMMANDS = ["আজ আবহাওয়া খুব ভালো", "আমি বাজারে যাচ্ছি", "আপনার নাম কী", "বাস কখন ছাড়বে", "ভাত খেয়েছেন",
                "আমার ভাই কাল আসবে", "দোকান বন্ধ আছে", "এক কাপ চা দিন", "রাস্তা পার হতে সাহায্য করুন", "হাসপাতাল কোথায়"]
THRESH = 0.6


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    return "".join(c for c in s if c.isalnum())


def sim(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return 1 - d[len(b)] / max(len(a), len(b))


def match(text: str):
    best = max(((sim(text, p), intent) for intent, ps in BN_COMMANDS.items() for p in ps), default=(0.0, None))
    return (best[1], round(best[0], 3)) if best[0] >= THRESH else (None, round(best[0], 3))


def to16k(x: np.ndarray, rate: int) -> np.ndarray:
    if rate == 16000:
        return x
    n = int(len(x) * 16000 / rate)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def piper_voice():
    from piper import PiperVoice
    v = PiperVoice.load(config.PIPER_BN_MODEL)

    def say(t):
        pcm = b"".join(c.audio_int16_bytes for c in v.synthesize(t))
        return to16k(np.frombuffer(pcm, np.int16).astype(np.float32) / 32768, v.config.sample_rate)
    return say


def google_voice():
    from gtts import gTTS
    import av  # decodes the mp3 (installed with faster-whisper)

    def say(t):
        buf = io.BytesIO()
        gTTS(t, lang="bn").write_to_fp(buf)
        buf.seek(0)
        frames = []
        with av.open(buf) as c:
            rs = av.AudioResampler(format="s16", layout="mono", rate=16000)
            for f in c.decode(audio=0):
                for g in rs.resample(f):
                    frames.append(g.to_ndarray().reshape(-1))
        return np.concatenate(frames).astype(np.float32) / 32768
    return say


def main():
    from faster_whisper import WhisperModel
    phrases = [(p, i) for i, ps in BN_COMMANDS.items() for p in ps]
    voices = {"piper_bn": piper_voice()}
    try:
        voices["google_bn"] = google_voice()
    except Exception as exc:  # noqa: BLE001
        print("Google voice unavailable:", exc)
    audio = {v: {"cmd": [say(p) for p, _ in phrases], "not": [say(t) for t in NOT_COMMANDS]} for v, say in voices.items()}
    res = {"date": "2026-10-14", "threshold": THRESH, "what": "synthetic voices on a laptop CPU; not people, not the Pi",
           "phrases": len(phrases), "not_commands": len(NOT_COMMANDS), "models": {}}
    prompt = "কমান্ড: " + ", ".join(p for p, _ in phrases)
    for size in ("tiny", "base", "small"):
        for use_prompt in (False, True):
            key = f"{size}{' + command list as prompt' if use_prompt else ''}"
            model = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=4)
            row = {"voices": {}}
            secs = []
            for v, a in audio.items():
                right = wrong = none = 0
                heard = []
                for (p, intent), x in zip(phrases, a["cmd"]):
                    t0 = time.perf_counter()
                    segs, _ = model.transcribe(x, language="bn", beam_size=1, vad_filter=False,
                                               initial_prompt=prompt if use_prompt else None, without_timestamps=True)
                    text = " ".join(s.text for s in segs).strip()
                    secs.append(time.perf_counter() - t0)
                    got, sc = match(text)
                    right += got == intent
                    wrong += got is not None and got != intent
                    none += got is None
                    heard.append({"said": p, "heard": text, "acted_on": got, "similarity": sc})
                fa = []
                for t, x in zip(NOT_COMMANDS, a["not"]):
                    segs, _ = model.transcribe(x, language="bn", beam_size=1, initial_prompt=prompt if use_prompt else None,
                                               without_timestamps=True)
                    text = " ".join(s.text for s in segs).strip()
                    got, sc = match(text)
                    if got:
                        fa.append({"said": t, "heard": text, "acted_on": got})
                row["voices"][v] = {"right": right, "wrong_command": wrong, "not_heard": none, "of": len(phrases),
                                    "false_commands": len(fa), "false_which": fa, "heard": heard}
            row["median_s_per_command_laptop"] = round(float(np.median(secs)), 3)
            res["models"][key] = row
            print(key, {v: {k: r[k] for k in ("right", "wrong_command", "not_heard", "false_commands")} for v, r in row["voices"].items()},
                  row["median_s_per_command_laptop"], "s", flush=True)
            out = Path(config.BASE_DIR) / "results" / "bangla_voice_commands.json"
            out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
