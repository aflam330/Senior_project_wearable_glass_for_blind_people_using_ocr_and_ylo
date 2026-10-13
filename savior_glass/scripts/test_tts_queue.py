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
    e._prefetch = lambda *a: None
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

# 7. a voice that cannot speak is tried once, then skipped; the next voice in the chain takes over
e = utils.TTSEngine()
calls = {"piper": 0, "binary": 0, "espeak": 0, "played": 0}


def bad(name):
    def f(text, model):
        calls[name] += 1
        raise OSError(name + " crashed")
    return f


e._render_piper, e._render_piper_binary = bad("piper"), bad("binary")
e._render_espeak = lambda text, voice: (calls.__setitem__("espeak", calls["espeak"] + 1), (b"\x01\x00" * 100, 22050))[1]
e._play = lambda pcm, rate: calls.__setitem__("played", calls["played"] + 1)
for i in range(4):
    e._synthesize(f"বাংলা {i}", "bn")
assert calls == {"piper": 1, "binary": 1, "espeak": 4, "played": 4}, calls

# 8. the audio of a piece is finished before the player starts, the next sentence is rendered while this one plays,
#    and a short fixed prompt is rendered only once
e = utils.TTSEngine()
events = []


def render(text, model):
    events.append(("render", text))
    return b"\x01\x00" * 2000, 22050


def play(pcm, rate):
    events.append(("play", len(pcm), rate))
    time.sleep(0.15)


e._render_piper, e._play = render, play
e.speak("এক। দুই। তিন।")
e.drain(5)
kinds = [(k, v) for k, v, *_ in events]
plays = [i for i, k in enumerate(kinds) if k[0] == "play"]
assert len(plays) == 3 and kinds.index(("render", "এক।")) < plays[0], kinds
assert kinds.index(("render", "দুই।")) < plays[1] and kinds.index(("render", "তিন।")) < plays[2], kinds   # ready before its turn
lead = int(22050 * config.TTS_LEAD_IN_MS / 1000) * 2 + int(22050 * config.TTS_TAIL_MS / 1000) * 2
n = len(events)
e.speak("এক।")
e.drain(5)
assert [k for k, *_ in events[n:]] == ["play"], events[n:]            # from the cache: no second rendering
e2 = utils.TTSEngine()
got = []
e2._render_piper = render
if utils.sys.platform != "win32":
    class P:                                                         # stands in for aplay
        def __init__(self, cmd, **k):
            got.append(cmd)
            self.stdin = self

        def write(self, b):
            got.append(len(b))

        def close(self):
            pass

        def wait(self):
            return 0

        def poll(self):
            return 0
    real, utils.subprocess.Popen = utils.subprocess.Popen, P
    utils.TTSEngine._player_cmd = ["aplay", "-r"]
    try:
        e2._synthesize("চার", "bn")
    finally:
        utils.subprocess.Popen, utils.TTSEngine._player_cmd = real, None
    assert got == [["aplay", "-r", "22050", "-"], 4000 + lead], got

# 9. espeak-ng's WAV (written to a pipe, so without a true length in the header) is unpacked by hand
hdr = (b"RIFF" + b"\xff" * 4 + b"WAVEfmt " + (16).to_bytes(4, "little") + (1).to_bytes(2, "little") + (1).to_bytes(2, "little")
       + (22050).to_bytes(4, "little") + (44100).to_bytes(4, "little") + (2).to_bytes(2, "little") + (16).to_bytes(2, "little")
       + b"data" + b"\xff" * 4)
assert utils.wav_pcm(hdr + b"\x05\x00" * 10) == (b"\x05\x00" * 10, 22050)
assert utils.wav_pcm(b"") == (b"", 0)

# 10. the real Bangla voice, when its model and the piper package are installed: speech comes out, at the model's rate
if utils.os.path.isfile(config.PIPER_BN_MODEL):
    try:
        import piper  # noqa: F401
        e = utils.TTSEngine()
        pcm, rate = e._render("পাঁচশত টাকার নোট", "bn")
        assert rate == 22050 and len(pcm) > rate, (rate, len(pcm))    # more than half a second of sound
        assert not e._piper_failed, e._piper_failed
        print(f"Bangla Piper voice: {len(pcm) / 2 / rate:.2f} s of speech rendered")
    except ImportError:
        print("piper package not installed: Bangla falls back to espeak-ng")
else:
    print("no Bangla Piper model here: Bangla falls back to espeak-ng (python scripts/get_speech_models.py fetches it)")
# 11. the "still working" tick: a short tone through the same queue, never while something is said or waits
e = utils.TTSEngine()
played = []
e._play = lambda pcm, rate: (played.append((len(pcm), rate)), time.sleep(0.2))
e._render_piper = lambda text, model: (b"\x01\x00" * 4000, 22050)
e.tick()
e.drain(5)
assert played == [(int(22050 * 0.07) * 2, 22050)], played
e.speak("এক")
time.sleep(0.05)
e.tick()                                   # speaking: this tick is dropped
e.drain(5)
assert len(played) == 2, played
print("speech queue: all checks pass")
