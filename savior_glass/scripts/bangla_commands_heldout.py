"""Bangla voice commands, second step: match Whisper's output also in Latin letters, and test on voices never used.

Found in scripts/bangla_commands_study.py (results/bangla_voice_commands.json): Whisper "base" with language="bn"
often writes Bangla speech in Latin letters ("Taka Dekku", "Lekha Bodha"), which the Bangla-script matcher
rejected. This step adds a fixed Latin spelling for every command phrase. Because that change was made after
looking at the outputs for the Piper and Google voices, those two voices are used only to CHOOSE the Whisper model
("design voices"). The result is then measured once on four voices that were never used:
Microsoft neural voices bn-BD Nabanita and Pradeep (Bangladesh) and bn-IN Tanishaa and Bashkar (India), clean and
with white noise at 10 dB signal-to-noise ratio, plus ten non-command sentences per voice.

Rule fixed before the held-out voices were generated: similarity >= 0.6 (unchanged); model chosen on the design
voices by (right - wrong - false commands), ties to the smaller model. Changed before any held-out voice was
generated: only "base" is scored, because "small" took 9.5 s per command on the laptop in the first study, far too
slow for the Pi; the design voices are then only reported.

    python scripts/bangla_commands_heldout.py
Output: results/bangla_voice_commands_heldout.json
"""
from __future__ import annotations

import asyncio
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from bangla_commands_study import BN_COMMANDS, NOT_COMMANDS, THRESH, google_voice, piper_voice, sim  # noqa: E402

LATIN = {
    "mode:ocr": ["lekha poro", "poro"],
    "mode:currency": ["taka dekho", "taka"],
    "mode:object": ["shamne ki ache", "bostu dekho"],
    "mode:claude": ["bornona koro"],
    "mode:emotion": ["mukh dekho", "onubhuti"],
    "next_mode": ["porer mod"],
    "action": ["chobi tolo"],
    "read": ["abar bolo"],
    "vol_up": ["jore"],
    "vol_down": ["aste"],
    "stop": ["thamo"],
}
HELD_OUT = ["bn-BD-NabanitaNeural", "bn-BD-PradeepNeural", "bn-IN-TanishaaNeural", "bn-IN-BashkarNeural"]


def latin(s: str) -> str:
    return "".join(c for c in s.lower() if "a" <= c <= "z")


def match(text: str):
    """Best command over the Bangla-script phrases and their Latin spellings."""
    best = (0.0, None)
    for intent, ps in BN_COMMANDS.items():
        for p in ps:
            best = max(best, (sim(text, p), intent), key=lambda x: x[0])
    lt = latin(text)
    if lt:
        for intent, ps in LATIN.items():
            for p in ps:
                best = max(best, (sim(lt, latin(p)), intent), key=lambda x: x[0])
    return (best[1], round(best[0], 3)) if best[0] >= THRESH else (None, round(best[0], 3))


def edge_voice(name):
    import av
    import edge_tts

    def say(t):
        async def get():
            buf = b""
            async for ch in edge_tts.Communicate(t, name).stream():
                if ch["type"] == "audio":
                    buf += ch["data"]
            return buf
        data = asyncio.run(get())
        frames = []
        with av.open(io.BytesIO(data)) as c:
            rs = av.AudioResampler(format="s16", layout="mono", rate=16000)
            for f in c.decode(audio=0):
                for g in rs.resample(f):
                    frames.append(g.to_ndarray().reshape(-1))
        return np.concatenate(frames).astype(np.float32) / 32768
    return say


def noisy(x, snr_db, rng):
    p = float(np.mean(x ** 2)) or 1e-6
    return (x + rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), len(x))).astype(np.float32)


def score(model, audio, phrases):
    out, secs = {}, []
    for v, a in audio.items():
        right = wrong = none = 0
        heard, fa = [], []
        for (p, intent), x in zip(phrases, a["cmd"]):
            t0 = time.perf_counter()
            segs, _ = model.transcribe(x, language="bn", beam_size=1, without_timestamps=True)
            text = " ".join(s.text for s in segs).strip()
            secs.append(time.perf_counter() - t0)
            got, sc = match(text)
            right += got == intent
            wrong += got is not None and got != intent
            none += got is None
            heard.append({"said": p, "heard": text, "acted_on": got, "similarity": sc})
        for t, x in zip(NOT_COMMANDS, a["not"]):
            segs, _ = model.transcribe(x, language="bn", beam_size=1, without_timestamps=True)
            text = " ".join(s.text for s in segs).strip()
            got, _ = match(text)
            if got:
                fa.append({"said": t, "heard": text, "acted_on": got})
        out[v] = {"right": right, "wrong_command": wrong, "not_heard": none, "of": len(phrases), "false_commands": len(fa),
                  "of_not_commands": len(NOT_COMMANDS), "false_which": fa, "heard": heard}
    return out, float(np.median(secs))


def total(r):
    return sum(v["right"] - v["wrong_command"] - v["false_commands"] for v in r.values())


def main():
    from faster_whisper import WhisperModel
    phrases = [(p, i) for i, ps in BN_COMMANDS.items() for p in ps]
    rng = np.random.default_rng(0)
    design = {}
    for name, say in (("piper_bn", piper_voice()), ("google_bn", google_voice())):
        design[name] = {"cmd": [say(p) for p, _ in phrases], "not": [say(t) for t in NOT_COMMANDS]}
    res = {"date": "2026-10-14", "threshold": THRESH, "latin_spellings": LATIN, "design_voices": list(design),
           "held_out_voices": HELD_OUT, "what": "synthetic voices on a laptop CPU, 8-bit Whisper; not people, not the Pi",
           "design": {}}
    out = Path(config.BASE_DIR) / "results" / "bangla_voice_commands_heldout.json"
    models = {}
    for size in ("base",):           # "small" needed 9.5 s per command on the laptop: too slow for the Pi
        models[size] = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=4)
        r, s = score(models[size], design, phrases)
        res["design"][size] = {"voices": r, "median_s_per_command_laptop": round(s, 3), "score": total(r)}
        print("design", size, {v: {k: x[k] for k in ("right", "wrong_command", "not_heard", "false_commands")} for v, x in r.items()}, round(s, 2), flush=True)
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    chosen = "base"
    res["chosen_model"] = chosen
    held = {}
    for name in HELD_OUT:
        say = edge_voice(name)
        cmd = [say(p) for p, _ in phrases]
        nots = [say(t) for t in NOT_COMMANDS]
        held[name] = {"cmd": cmd, "not": nots}
        held[name + " + noise 10 dB"] = {"cmd": [noisy(x, 10, rng) for x in cmd], "not": [noisy(x, 10, rng) for x in nots]}
    r, s = score(models[chosen], held, phrases)
    res["held_out"] = {"model": chosen, "voices": r, "median_s_per_command_laptop": round(s, 3)}
    for cond in ("clean", "noise"):
        vs = [v for k, v in r.items() if ("noise" in k) == (cond == "noise")]
        res["held_out"][f"summary_{cond}"] = {"right": sum(v["right"] for v in vs), "wrong_command": sum(v["wrong_command"] for v in vs),
                                              "not_heard": sum(v["not_heard"] for v in vs), "of": sum(v["of"] for v in vs),
                                              "false_commands": sum(v["false_commands"] for v in vs),
                                              "of_not_commands": sum(v["of_not_commands"] for v in vs)}
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("held-out", chosen, res["held_out"]["summary_clean"], res["held_out"]["summary_noise"], round(s, 2), flush=True)


if __name__ == "__main__":
    main()
