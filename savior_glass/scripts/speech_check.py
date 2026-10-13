"""Speak a few sentences with the glass's own speech engine and time it. Run this on the Raspberry Pi to hear the voice.

    python scripts/speech_check.py              # speaks, prints which voice was used and how long rendering took
    TTS_VOICE=espeak python scripts/speech_check.py     # the same with espeak-ng only, to compare by ear
    python scripts/speech_check.py --quiet      # renders and times, plays nothing

Output: results/speech_check_<host>.json. "render_s" is the wait before a sentence starts; "audio_s" is its length.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import utils  # noqa: E402

SENTENCES = [("স্মার্ট গ্লাস চালু হচ্ছে", "bn"), ("পাঁচশত টাকার নোট, সম্ভবত আসল।", "bn"), ("কোনো লেখা পাওয়া যায়নি", "bn"),
             ("ঢাকা থেকে চট্টগ্রাম যেতে ছয় ঘণ্টা লাগে, ভাড়া ৫০০ টাকা।", "bn"),
             ("Exit is on the left.", "en"), ("No smoking. Push the door to open.", "en")]


def main() -> None:
    quiet = "--quiet" in sys.argv
    e = utils.TTSEngine()
    e._cache.clear()
    rows = []
    for text, lang in SENTENCES:
        t0 = time.perf_counter()
        pcm, rate = e._render(text, lang)
        dt = time.perf_counter() - t0
        model = config.PIPER_BN_MODEL if lang == "bn" else config.PIPER_EN_MODEL
        voice = "none"
        if pcm:
            voice = ("piper (python)" if model in e._voices else
                     "piper (binary)" if ("piper-binary", model) not in e._piper_failed and Path(config.PIPER_BINARY).is_file()
                     and Path(model).is_file() and config.TTS_VOICE != "espeak" else "espeak-ng")
        audio = len(pcm) / 2 / rate if rate else 0.0
        rows.append({"text": text, "lang": lang, "voice": voice, "render_s": round(dt, 3), "audio_s": round(audio, 2), "rate": rate})
        print(f"{lang}  {voice:15s} render {dt:5.2f} s  audio {audio:5.2f} s  {rate} Hz")
        if pcm and not quiet:
            e._play(pcm, rate)
    # the first sentence of each voice includes loading the voice; a second pass shows the steady cost
    e._cache.clear()
    for r, (text, lang) in zip(rows, SENTENCES):
        t0 = time.perf_counter()
        e._render(text, lang)
        r["render_again_s"] = round(time.perf_counter() - t0, 3)
    out = Path(config.BASE_DIR) / "results" / f"speech_check_{platform.node() or 'host'}.json"
    out.write_text(json.dumps({"date": time.strftime("%Y-%m-%d %H:%M"), "machine": platform.machine(), "tts_voice": config.TTS_VOICE,
                               "player": None if sys.platform == "win32" else utils.TTSEngine._player_cmd,
                               "sentences": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print("render again (voices loaded):", [r["render_again_s"] for r in rows])
    print("saved", out)


if __name__ == "__main__":
    main()
