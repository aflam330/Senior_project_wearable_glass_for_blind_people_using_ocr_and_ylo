"""Checks of the speech queue and of how text is cut into pieces (utils.TTSEngine). No sound is played:
the synthesis step is replaced by a recorder.

    python scripts/test_tts_queue.py
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import utils  # noqa: E402

config.TTS_INTER_UTTERANCE_PAUSE = 0.0


def engine(delay=0.0):
    e = utils.TTSEngine()
    e.spoken = []
    gate = threading.Event()
    gate.set()

    def fake(text, lang):
        gate.wait()
        if delay:
            time.sleep(delay)
        e.spoken.append((text, lang))
    e._synthesize = fake
    e.gate = gate
    return e


# 1. a Bangla sentence with commas, digits and the Bangla full stop is one piece per sentence, all in the Bangla voice
seg = utils.split_language_segments("ঢাকা, চট্টগ্রাম ও খুলনা। ভাড়া 50 টাকা।")
assert seg == [("ঢাকা, চট্টগ্রাম ও খুলনা। ভাড়া 50 টাকা।", "bn")], seg
e = engine()
e.speak("ঢাকা, চট্টগ্রাম ও খুলনা। ভাড়া 50 টাকা।")
e.drain(5)
assert e.spoken == [("ঢাকা, চট্টগ্রাম ও খুলনা।", "bn"), ("ভাড়া 50 টাকা।", "bn")], e.spoken

# 2. mixed text still changes voice at real words of the other language
seg = utils.split_language_segments("দাম 20 টাকা Thank you")
assert seg == [("দাম 20 টাকা", "bn"), ("Thank you", "en")], seg

# 3. a long text (12 sentences) is spoken whole and in order: nothing is pushed out of the queue
long = " ".join(f"এটি বাক্য নম্বর {i}।" for i in range(1, 13))
e = engine()
e.speak(long)
e.drain(5)
assert len(e.spoken) == 12 and e.spoken[0][0].endswith("1।") and e.spoken[-1][0].endswith("12।"), e.spoken

# 4. the lines of a page (keep=True) are never dropped, however many wait
e = engine()
e.gate.clear()
for i in range(1, 21):
    e.speak(f"লাইন {i}", keep=True)
e.gate.set()
e.drain(5)
assert [t for t, _ in e.spoken] == [f"লাইন {i}" for i in range(1, 21)], e.spoken

# 5. ordinary announcements still go stale: only the newest TTS_QUEUE_MAXSIZE wait
e = engine()
e.gate.clear()
e.speak("first")                      # taken by the worker at once, waits at the gate
time.sleep(0.2)
for i in range(1, 11):
    e.speak(f"object {i}")
e.gate.set()
e.drain(5)
said = [t for t, _ in e.spoken]
assert said[0] == "first" and said[1:] == [f"object {i}" for i in range(11 - config.TTS_QUEUE_MAXSIZE, 11)], said

# 6. stop_current (MODE button) abandons the rest of a long message
e = engine(delay=0.05)
e.speak(long)
time.sleep(0.12)
e.stop_current()
e.drain(5)
assert 1 <= len(e.spoken) < 12, len(e.spoken)

# 7. a voice Piper cannot speak is tried once, then skipped
e = utils.TTSEngine()
calls = {"piper": 0, "espeak": 0}
real_isfile = utils.os.path.isfile
utils.os.path.isfile = lambda p: True if p in (config.PIPER_BN_MODEL, config.PIPER_BINARY) else real_isfile(p)


def bad_popen(*a, **k):
    calls["piper"] += 1
    raise OSError("piper crashed")


real_popen, utils.subprocess.Popen = utils.subprocess.Popen, bad_popen
e._espeak = lambda text, voice: calls.__setitem__("espeak", calls["espeak"] + 1)
try:
    for _ in range(4):
        e._synthesize("বাংলা", "bn")
finally:
    utils.subprocess.Popen, utils.os.path.isfile = real_popen, real_isfile
assert calls == {"piper": 1, "espeak": 4}, calls
print("speech queue: all checks pass")
