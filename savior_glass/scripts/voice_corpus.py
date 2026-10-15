"""Build the speech corpus for the voice-command studies (synthetic voices; not people).

Every English command phrase of voice_commands.PHRASES and 30 ordinary sentences, spoken by 20 Microsoft neural
voices (edge-tts, online; only these phrases are sent), saved as 16 kHz mono WAV under
"<workspace>/data set/voice_cmd_corpus/<voice>/". Two groups, fixed here before any measurement:

  design   8 voices, the only ones any setting may be tuned on
  heldout 12 voices never used for tuning, among them four BANGLA voices reading the English commands
           (Bangladesh: Nabanita, Pradeep; India: Tanishaa, Bashkar), the nearest available stand-in for
           Bangladeshi users speaking English command words

Babble noise is made from six other voices reading other sentences (two babble sets: design and test).

    python scripts/voice_corpus.py
"""
from __future__ import annotations

import asyncio
import io
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import voice_commands as V  # noqa: E402

ROOT = Path(config.BASE_DIR).parent / "data set" / "voice_cmd_corpus"
DESIGN = ["en-US-GuyNeural", "en-GB-SoniaNeural", "en-AU-NatashaNeural", "en-IN-PrabhatNeural", "en-NG-AbeoNeural",
          "en-SG-LunaNeural", "en-KE-AsiliaNeural", "en-PH-RosaNeural"]
HELDOUT = ["en-IN-NeerjaNeural", "en-US-AriaNeural", "en-GB-RyanNeural", "en-IE-EmilyNeural", "en-ZA-LukeNeural",
           "en-PH-JamesNeural", "en-NZ-MollyNeural", "en-CA-LiamNeural",
           "bn-BD-NabanitaNeural", "bn-BD-PradeepNeural", "bn-IN-TanishaaNeural", "bn-IN-BashkarNeural"]
BABBLE = {"design": ["en-US-ChristopherNeural", "en-US-EricNeural", "en-GB-ThomasNeural"],
          "test": ["en-US-RogerNeural", "en-US-MichelleNeural", "en-GB-LibbyNeural"]}
NOT_COMMANDS = [
    "I will read the newspaper later today", "the money is in the bank", "please open the window", "what time is it now",
    "there are many objects on the table", "she has a happy face", "can you describe the problem to me",
    "we should stop at the next station", "the volume of the box is large", "he said it again and again",
    "the weather is nice this morning", "my brother is coming home tomorrow", "how much does this cost",
    "the bus leaves at nine", "I need to check my phone", "let us take a picture together", "turn left at the corner",
    "the text book is on the shelf", "this is a quiet street", "good morning how are you",
    "where is the nearest hospital", "I would like a cup of tea", "the emotion in his voice was clear",
    "one two three four five", "thank you very much", "call me when you arrive", "the note was written by hand",
    "it is too loud in here", "the shop is closed on friday", "rice and fish for lunch"]
BABBLE_TEXT = ["Yesterday we went to the market and bought vegetables, fish and some fruit for the family.",
               "The meeting has been moved to the afternoon because the manager is still travelling.",
               "Children were playing in the field while their parents talked near the tea stall.",
               "If the train is late again tomorrow I will take the bus and call you from the station.",
               "She said the new shop on the corner sells good shoes at a fair price this week."]


def synth(text: str, voice: str) -> np.ndarray:
    import av
    import edge_tts

    async def get():
        buf = b""
        async for ch in edge_tts.Communicate(text, voice).stream():
            if ch["type"] == "audio":
                buf += ch["data"]
        return buf
    data = asyncio.run(get())
    out = []
    with av.open(io.BytesIO(data)) as c:
        rs = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for f in c.decode(audio=0):
            for g in rs.resample(f):
                out.append(g.to_ndarray().reshape(-1))
    return np.concatenate(out).astype(np.int16)


def save(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(x.tobytes())


def main():
    phrases = sorted(V._BY_PHRASE)
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    shard, shards = (int(nums[0]), int(nums[1])) if len(nums) > 1 else (0, 1)   # parallel processes
    for group, voices in (("design", DESIGN), ("heldout", HELDOUT)):
        for v in voices:
            if (DESIGN + HELDOUT).index(v) % shards != shard:
                continue
            kinds = (("cmd", phrases), ("not", NOT_COMMANDS))
            if "--wake" in sys.argv:          # "glass, <command>" for the wake-word mode (VOICE_WAKE_WORD=glass)
                kinds = (("wake", [f"glass, {p}" for p in phrases]),)
            for kind, texts in kinds:
                for i, t in enumerate(texts):
                    p = ROOT / v / f"{kind}_{i:02d}.wav"
                    if not p.exists():
                        for attempt in range(5):
                            try:
                                save(p, synth(t, v))
                                break
                            except Exception as exc:  # noqa: BLE001
                                print("retry", v, t, exc, flush=True)
            print(group, v, "done", flush=True)
    if shard != 0 or "--wake" in sys.argv:
        return
    for part, voices in BABBLE.items():
        tracks = []
        for j, v in enumerate(voices):
            x = np.concatenate([synth(t, v).astype(np.float32) for t in BABBLE_TEXT[j:] + BABBLE_TEXT[:j]])
            tracks.append(x)
        n = min(len(t) for t in tracks)
        save(ROOT / "_babble" / f"{part}.wav", np.clip(sum(t[:n] for t in tracks) / len(tracks), -32768, 32767).astype(np.int16))
    (ROOT / "manifest.json").write_text(json.dumps({"design": DESIGN, "heldout": HELDOUT, "babble": BABBLE, "phrases": phrases,
                                                     "not_commands": NOT_COMMANDS}, indent=1), encoding="utf-8")
    print("corpus ready", ROOT)


if __name__ == "__main__":
    main()
