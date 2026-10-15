"""Voice commands under hard conditions: accuracy, false commands and response time; settings chosen on design voices.

Corpus: scripts/voice_corpus.py (20 neural voices; 8 design, 12 held out, 4 of them Bangla voices reading English).
Noise: babble (three other voices talking at once; separate design and test babble) and real environmental sound
from ESC-50 (non-animal classes; folds 1-4 for design, fold 5 for test). Conditions:

  clean; babble at 10, 5, 0, -5, -10 dB SNR; environmental sound at 10, 5, 0, -5, -10 dB SNR;
  soft voice (-20 dB); whisper (simulated: the voice's spectral envelope, excited by noise instead of the vocal
  folds, LPC order 16, at -25 dB); whisper with babble at 10 dB.

Each utterance gets 0.6 s of lead-in and 1.5 s of tail and is fed in 0.1 s blocks, as a microphone delivers it.
Response time = time from the end of the speech (last sample above 1 % of the peak, in the clean signal) to the end
of the block in which the command was decided, plus the computing time spent on that block.

Settings compared (only on design voices, design babble and ESC-50 folds 1-4):
  front end: none / denoise / level / denoise+level;  early decision: off / on;  word confidence: 0.5 / 0.6 / 0.7;
  end of speech: Vosk's own / Silero voice-activity detection;
  wake word: none / "glass" before every command (added when babble turned other people's speech into commands; the
  wake-word recordings were made then, for all voices, before any held-out measurement). The test reports the current
  setting, the best setting without a wake word, and the best overall.
Score: right - 3 x wrong - 3 x false commands, over all conditions; ties to the simpler setting. The chosen setting
and the current one (none, off, 0.7) are then measured ONCE on the held-out voices, test babble and ESC-50 fold 5.
Synthetic voices on a laptop; not people, not the Pi's microphone.

    python scripts/voice_eval.py design      # all settings on the design voices
    python scripts/voice_eval.py test        # the chosen and the current setting on the held-out voices
Output: results/voice_eval.json
"""
from __future__ import annotations

import csv
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import voice_commands as V  # noqa: E402

CORPUS = Path(config.BASE_DIR).parent / "data set" / "voice_cmd_corpus"
ESC = Path(config.BASE_DIR).parent / "data set" / "ESC-50-master"
OUT = Path(config.BASE_DIR) / "results" / "voice_eval.json"
RATE, BLOCK = 16000, 1600
ANIMALS = {"dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow"}
SNRS = (10, 5, 0, -5, -10)


def read(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def esc_noise(folds) -> np.ndarray:
    rows = [r for r in csv.DictReader(open(ESC / "meta" / "esc50.csv")) if int(r["fold"]) in folds and r["category"] not in ANIMALS]
    xs = []
    for r in rows:
        import av
        with av.open(str(ESC / "audio" / r["filename"])) as c:
            rs = av.AudioResampler(format="s16", layout="mono", rate=RATE)
            fr = [g.to_ndarray().reshape(-1) for f in c.decode(audio=0) for g in rs.resample(f)]
        x = np.concatenate(fr).astype(np.float32) / 32768
        if np.sqrt(np.mean(x ** 2)) > 1e-3:             # skip the silent clips
            xs.append(x / (np.sqrt(np.mean(x ** 2)) + 1e-9))
    return np.concatenate(xs)


def lpc(frame: np.ndarray, order: int = 16) -> np.ndarray:
    r = np.correlate(frame, frame, "full")[len(frame) - 1:len(frame) + order]
    a = np.zeros(order + 1)
    a[0], e = 1.0, r[0] + 1e-9
    for i in range(1, order + 1):
        k = -(r[i] + np.dot(a[1:i], r[i - 1:0:-1])) / e
        a[1:i] = a[1:i] + k * a[i - 1:0:-1]
        a[i] = k
        e *= 1 - k * k
    return a


def whisper(x: np.ndarray, rng) -> np.ndarray:
    """Whispered speech: each 25 ms frame's spectral envelope (LPC) excited by white noise at the frame's energy."""
    n, h = 400, 200
    w = np.hanning(n)
    out = np.zeros(len(x) + n)
    for i in range(0, len(x) - n, h):
        f = x[i:i + n] * w
        en = float(np.sqrt(np.mean(f ** 2)))
        if en < 1e-5:
            continue
        a = lpc(f)
        exc = rng.normal(0, 1, n)
        y = np.zeros(n)
        for t in range(n):                               # all-pole filter 1/A(z)
            acc = exc[t]
            for k in range(1, min(t, 16) + 1):
                acc -= a[k] * y[t - k]
            y[t] = acc
        y *= en / (np.sqrt(np.mean(y ** 2)) + 1e-9)
        out[i:i + n] += y * w
    return out[:len(x)].astype(np.float32)


def place(x: np.ndarray):
    """Lead-in and tail around the speech; returns the signal and the sample where the speech ends."""
    pk = np.max(np.abs(x)) + 1e-9
    idx = np.where(np.abs(x) > 0.01 * pk)[0]
    end = idx[-1] if len(idx) else len(x)
    lead = int(0.6 * RATE)
    return np.concatenate([np.zeros(lead, np.float32), x, np.zeros(int(1.5 * RATE), np.float32)]), lead + end


def add(sig, noise, snr_db, speech_power, rng):
    if noise is None:
        return sig
    s = rng.integers(0, len(noise) - len(sig))
    nz = noise[s:s + len(sig)]
    nz = nz / (np.sqrt(np.mean(nz ** 2)) + 1e-9) * np.sqrt(speech_power / 10 ** (snr_db / 10))
    return sig + nz


def conditions(part):
    babble = read(CORPUS / "_babble" / f"{part}.wav")
    env = esc_noise({1, 2, 3, 4} if part == "design" else {5})
    c = [("clean", None, None, "normal")]
    c += [(f"babble {s} dB", babble, s, "normal") for s in SNRS]
    c += [(f"street/home sounds {s} dB", env, s, "normal") for s in SNRS]
    c += [("soft voice (-20 dB)", None, None, "soft"), ("whisper (simulated)", None, None, "whisper"),
          ("whisper + babble 10 dB", babble, 10, "whisper")]
    return c


def decode(vrec, fe, sig, end_sample, vad=None, wake=""):
    """Feed one utterance block by block; return the timeline of results: (kind, text, min_conf, t, segment).
    kind "partial" = the first settled partial result of a segment that is a whole, non-prefix command phrase."""
    vrec.Reset()
    pcm = (np.clip(sig, -1, 1) * 32767).astype(np.int16).tobytes()
    ev, seg, last, fired = [], 0, "", False
    for i in range(0, len(sig), BLOCK):
        t0 = time.perf_counter()
        chunk = fe(pcm[2 * i:2 * (i + BLOCK)])
        done = vrec.AcceptWaveform(chunk)
        vad_end = (not done) and vad is not None and vad.ended(chunk)
        dt = time.perf_counter() - t0
        t = (min(i + BLOCK, len(sig)) - end_sample) / RATE + dt
        if done or vad_end:
            r = json.loads(vrec.Result() if done else vrec.FinalResult())
            if r.get("text"):
                ev.append(("final", r["text"], min([w.get("conf", 0) for w in r.get("result", [])] or [0]), t, seg))
            seg, last, fired = seg + 1, "", False
            continue
        if fired:
            continue
        pr = json.loads(vrec.PartialResult())
        text = pr.get("partial", "")
        core = " ".join(text.split()[1:]) if wake else text
        if text and text == last and V.parse(text, wake) and core not in V._PREFIXES:
            ev.append(("partial", text, min([w.get("conf", 0) for w in pr.get("partial_result", [])] or [0]), t, seg))
            fired = True
        last = text
    r = json.loads(vrec.FinalResult())
    if r.get("text"):
        ev.append(("final", r["text"], min([w.get("conf", 0) for w in r.get("result", [])] or [0]), (len(sig) - end_sample) / RATE, seg))
    return ev


def decide(ev, early, conf, wake=""):
    """Commands the glass would carry out, in order, with their times, for one setting."""
    out, done_segs = [], set()
    for kind, text, c, t, seg in ev:
        intent = V.parse(text, wake)
        if intent is None or c < conf or seg in done_segs:
            continue
        if kind == "partial" and not early:
            continue
        out.append((intent, t))
        done_segs.add(seg)
    return out


EARLY_CONF = [(e, c) for e in (False, True) for c in (0.7, 0.6, 0.5)]


def evaluate(voices, part, front, rng_seed=0, endpoint="vosk", wake=""):
    """All early/confidence variants for one front end: {(early, conf): {condition: counts}}."""
    import vosk
    from voice_frontend import FrontEnd
    vosk.SetLogLevel(-1)
    model = V._model()
    phrases = sorted(V._BY_PHRASE)
    grammar = [f"{wake} {p}" for p in phrases] if wake else phrases
    vrec = vosk.KaldiRecognizer(model, RATE, json.dumps(grammar + ["[unk]"]))
    vrec.SetWords(True)
    vrec.SetPartialWords(True)
    man = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    out = {k: {} for k in EARLY_CONF}
    whisper_cache = {}
    for name, noise, snr, voice_kind in conditions(part):
        rng = np.random.default_rng(rng_seed)
        tally = {k: {"right": 0, "wrong": 0, "not_heard": 0, "false_commands": 0, "lat": []} for k in EARLY_CONF}
        for v in voices:
            for kind, texts in (("cmd", phrases), ("not", man["not_commands"])):
                for i, text in enumerate(texts):
                    x = read(CORPUS / v / f"{'wake' if (wake and kind == 'cmd') else kind}_{i:02d}.wav")
                    if voice_kind == "soft":
                        x = x * 0.1
                    elif voice_kind == "whisper":
                        key = (v, kind, i)
                        if key not in whisper_cache:
                            whisper_cache[key] = whisper(x, np.random.default_rng(i)) * 0.056
                        x = whisper_cache[key]
                    sig, end = place(x)
                    sp = float(np.mean(x[np.abs(x) > 0.01 * (np.max(np.abs(x)) + 1e-9)] ** 2))
                    sig = add(sig, noise, snr, sp, rng)
                    vad = None
                    if endpoint == "vad":
                        from voice_frontend import VadEndpointer
                        vad = VadEndpointer()
                    ev = decode(vrec, FrontEnd(front), sig, end, vad, wake)
                    for k in EARLY_CONF:
                        d = decide(ev, *k, wake=wake)
                        T = tally[k]
                        if kind == "cmd":
                            if d and d[0][0] == V._BY_PHRASE[text] and all(x[0] == d[0][0] for x in d):
                                T["right"] += 1
                                T["lat"].append(d[0][1])
                            elif d:
                                T["wrong"] += 1
                            else:
                                T["not_heard"] += 1
                        elif d:
                            T["false_commands"] += 1
        for k in EARLY_CONF:
            T = tally[k]
            n = T["right"] + T["wrong"] + T["not_heard"]
            out[k][name] = {"right": T["right"], "wrong": T["wrong"], "not_heard": T["not_heard"], "n": n,
                            "right_rate": round(T["right"] / max(n, 1), 4), "false_commands": T["false_commands"],
                            "non_commands": len(voices) * len(man["not_commands"]),
                            "response_s_median": round(float(np.median(T["lat"])), 3) if T["lat"] else None,
                            "response_s_p90": round(float(np.percentile(T["lat"], 90)), 3) if T["lat"] else None}
        print(" ", front, name, {f"{'E' if k[0] else 'F'}{k[1]}": (out[k][name]["right"], out[k][name]["wrong"], out[k][name]["false_commands"],
                                                                    out[k][name]["response_s_median"]) for k in EARLY_CONF}, flush=True)
    return out


def score(r):
    return sum(v["right"] - 3 * v["wrong"] - 3 * v["false_commands"] for v in r.values())


def key_of(front, early, conf, endpoint="vosk", wake=""):
    return f"{front} | early {'on' if early else 'off'} | conf {conf} | end {endpoint} | wake {wake or '-'}"


def main():
    man = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"date": "2026-10-14", "design": {}}
    if sys.argv[1] == "design":
        jobs = [(f, e, "") for f in ("none", "denoise", "level", "denoise+level") for e in ("vosk", "vad")]
        if "--wake" in sys.argv:                                # the wake-word mode ("glass <command>")
            jobs = [(f, e, "glass") for f in ("none", "denoise+level") for e in ("vosk", "vad")]
        nums = [a for a in sys.argv[2:] if a.isdigit()]
        if len(nums) == 2:                                      # one process per job: design <k> <n>
            jobs = jobs[int(nums[0])::int(nums[1])]
        for front, endpoint, wake in jobs:
            part_file = OUT.with_name(f"voice_eval_design_{front}_{endpoint}{'_wake' if wake else ''}.json")
            if part_file.exists():
                continue
            r = evaluate(man["design"], "design", front, endpoint=endpoint, wake=wake)
            part_file.write_text(json.dumps({key_of(front, e, c, endpoint, wake): {"score": score(cond), "conditions": cond}
                                             for (e, c), cond in r.items()}, indent=1), encoding="utf-8")
        if len(nums) == 2:
            return
        for f in sorted(OUT.parent.glob("voice_eval_design_*.json")):
            for k, v in json.loads(f.read_text(encoding="utf-8")).items():
                res["design"][k if "| wake" in k else k + " | wake -"] = v     # files from before the wake word
        OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
        order = ["none", "level", "denoise", "denoise+level"]

        def simple(k):
            f, e, c, n, w = [x.strip() for x in k.split("|")]
            return (order.index(f), e == "early on", -float(c.split()[1]), n == "end vad", w != "wake -")
        best = max(res["design"], key=lambda k: (res["design"][k]["score"], tuple(-x for x in simple(k))))
        res["chosen"] = best
        OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
        for k in sorted(res["design"], key=lambda k: -res["design"][k]["score"])[:8]:
            print(res["design"][k]["score"], k)
        print("chosen:", best)
    else:
        res.setdefault("test", {})
        groups = {"all held-out voices": man["heldout"], "Bangla voices reading English": [v for v in man["heldout"] if v.startswith("bn-")]}
        cur = key_of("none", False, 0.7)
        best_without_wake = max((k for k in res["design"] if k.endswith("wake -")), key=lambda k: res["design"][k]["score"])
        for key in dict.fromkeys([cur, best_without_wake, res["chosen"]]):
            if key in res["test"]:
                continue
            f, e, c, n, w = [x.strip() for x in key.split("|")]
            early, conf, endpoint, wake = e == "early on", float(c.split()[1]), n.split()[1], w.split()[1].strip("-")
            res["test"][key] = {}
            for g, vs in groups.items():
                r = evaluate(vs, "test", f, endpoint=endpoint, wake=wake)
                res["test"][key][g] = r[(early, conf)]
            OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
            print(key, "done", flush=True)


if __name__ == "__main__":
    main()
