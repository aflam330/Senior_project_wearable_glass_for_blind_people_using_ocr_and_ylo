"""Fetch the speech models the glass uses (they are too large for the repository). Needs internet once.

    python scripts/get_speech_models.py            # voices (Bangla, English) and the voice-command recogniser
    python scripts/get_speech_models.py --check    # only say what is installed

  models/piper/bn_BD-google-medium.onnx (+ .json)   Bangla voice, 77 MB   (Piper voices, rhasspy/piper-voices)
  models/piper/en_US-amy-low.onnx (+ .json)         English voice, 63 MB
  models/vosk/vosk-model-small-en-us-0.15/          offline recogniser for voice commands, 68 MB (Vosk, Apache 2.0)

Without them the glass still works: it speaks with espeak-ng and takes commands from the buttons only.
"""
from __future__ import annotations

import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

HF = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
VOICES = {config.PIPER_BN_MODEL: HF + "bn/bn_BD/google/medium/bn_BD-google-medium.onnx",
          config.PIPER_EN_MODEL: HF + "en/en_US/amy/low/en_US-amy-low.onnx"}
VOSK_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"


def fetch(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print("  downloading", url, flush=True)
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(tmp, dest)


def main() -> int:
    check = "--check" in sys.argv
    missing = 0
    for path, url in VOICES.items():
        for p, u in ((path, url), (path + ".json", url + ".json")):
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                print("ok      ", os.path.relpath(p, config.BASE_DIR))
            elif check:
                print("MISSING ", os.path.relpath(p, config.BASE_DIR))
                missing += 1
            else:
                fetch(u, p)
    vosk = config.VOICE_CMD_MODEL
    if os.path.isdir(vosk) and os.listdir(vosk):
        print("ok      ", os.path.relpath(vosk, config.BASE_DIR))
    elif check:
        print("MISSING ", os.path.relpath(vosk, config.BASE_DIR))
        missing += 1
    else:
        z = vosk + ".zip"
        fetch(VOSK_URL, z)
        with zipfile.ZipFile(z) as f:
            f.extractall(os.path.dirname(vosk))
        os.remove(z)
    for pkg, why in (("piper", "natural voices: pip install piper-tts"), ("vosk", "voice commands: pip install vosk"),
                     ("sounddevice", "microphone: pip install sounddevice (and sudo apt install libportaudio2)")):
        try:
            __import__(pkg)
            print("ok       python package", pkg)
        except Exception as exc:  # noqa: BLE001
            print(f"MISSING  python package {pkg} ({why}) [{type(exc).__name__}]")
            missing += 1
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
