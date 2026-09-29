"""Speech-output test on the Windows harness path (the Pi uses espeak-ng/Piper instead).

Uses the glass's own routing (test_windows._english_fallback, the Bangla detector, and
assistive.adapt_feedback) and the same SAPI settings as WindowsTTS._sapi_speak
(volume, rate = 1 + emotion rate delta), but writes WAV files instead of playing them:
  results/speech_samples/*.wav  and  results/speech_test.json

Measured: routing of Bangla vs English text, duration change from emotion-adaptive rate,
loudness change from the volume setting, and (if online) a speech-recognition round trip
for English intelligibility. Subjective quality needs human listeners: NOT_MEASURED.
"""
from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "speech_samples"


def sapi_to_wav(text: str, path: Path, volume: int, rate: int) -> None:
    safe = text.replace("'", "''")
    script = ("Add-Type -AssemblyName System.Speech; "
              "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
              f"$s.Volume = {volume}; $s.Rate = {rate}; $s.SetOutputToWaveFile('{path}'); "
              f"$s.Speak('{safe}'); $s.Dispose();")
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], check=True,
                   capture_output=True, timeout=60)


def wav_stats(path: Path) -> dict:
    with wave.open(str(path)) as w:
        frames = w.readframes(w.getnframes())
        rate, width = w.getframerate(), w.getsampwidth()
        dur = w.getnframes() / rate
    x = np.frombuffer(frames, dtype=np.int16 if width == 2 else np.uint8).astype(np.float64)
    rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    return {"seconds": round(dur, 3), "rms": round(rms, 1), "sample_rate": rate}


def main() -> None:
    from test_windows import _BANGLA_RE, _english_fallback
    from assistive import adapt_feedback

    OUT.mkdir(parents=True, exist_ok=True)
    report = {"engine": "Windows SAPI (System.Speech)", "cases": []}

    def route(text):
        s = text.replace(" ", "")
        return "bn" if len(_BANGLA_RE.findall(s)) / max(len(s), 1) > 0.25 else "en"

    # 1. routing: what each glass message becomes on this machine
    for text in ("একশত টাকার নোট। আসল", "সামনে আছে: চেয়ার, মানুষ", "Bus Stop", "পড়া হচ্ছে: হাসপাতাল"):
        lang = route(text)
        report["cases"].append({"input": text, "language": lang,
                                "sapi_text": text if lang == "en" else _english_fallback(text)})

    # 2. emotion-adaptive rate: same message, different emotions, same volume
    msg = "100 taka note. Genuine. Hold the note a little closer for the security thread."
    rates = {}
    for emo in ("neutral", "happiness", "sadness", "fear"):
        a = adapt_feedback(msg, emo, enabled=True, bangla=False)
        rate = max(-10, min(10, 1 + int(a.get("rate") or 0)))
        p = OUT / f"en_{emo}.wav"
        sapi_to_wav(a["text"], p, 85, rate)
        st = wav_stats(p)
        rates[emo] = {"spoken_text": a["text"], "sapi_rate": rate, **st}
    report["emotion_adaptive"] = rates

    # 3. volume control: identical text and rate, volume 40 vs 85
    vol = {}
    for v in (40, 85):
        p = OUT / f"en_volume{v}.wav"
        sapi_to_wav(msg, p, v, 1)
        vol[v] = wav_stats(p)
    report["volume"] = {**{str(k): v for k, v in vol.items()},
                        "rms_ratio_40_over_85": round(vol[40]["rms"] / max(vol[85]["rms"], 1e-9), 3)}

    # 4. Bangla: online gTTS path used by the Windows harness (offline espeak-ng/Piper is the Pi path)
    try:
        from gtts import gTTS
        gTTS("একশত টাকার নোট। আসল", lang="bn").save(str(OUT / "bn_gtts.mp3"))
        report["bangla_gtts"] = {"status": "ok", "bytes": (OUT / "bn_gtts.mp3").stat().st_size}
    except Exception as exc:  # noqa: BLE001
        report["bangla_gtts"] = {"status": "NOT_MEASURED", "reason": f"{type(exc).__name__}: {exc}"[:200]}

    # 5. English intelligibility: speech-recognition round trip (needs internet)
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.AudioFile(str(OUT / "en_neutral.wav")) as src:
            heard = r.recognize_google(r.record(src), language="en-US")
        ref = rates["neutral"]["spoken_text"].lower().replace(".", "").replace(",", "").split()
        hyp = heard.lower().split()
        report["asr_round_trip"] = {"reference": " ".join(ref), "heard": heard,
                                    "word_overlap": round(len(set(ref) & set(hyp)) / max(len(set(ref)), 1), 3)}
    except Exception as exc:  # noqa: BLE001
        report["asr_round_trip"] = {"status": "NOT_MEASURED", "reason": f"{type(exc).__name__}: {exc}"[:200]}

    report["subjective_quality"] = "NOT_MEASURED (needs human listeners)"
    report["offline_bangla_voice_on_pi"] = "NOT_MEASURED (espeak-ng/Piper run on the Raspberry Pi, not installed here)"
    (ROOT / "results" / "speech_test.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")  # Bangla text; the Windows console defaults to cp1252
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
