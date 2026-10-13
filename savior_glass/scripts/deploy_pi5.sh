#!/bin/sh
# Prepare this checkout on a Raspberry Pi 5. Does not invent latency numbers.
#
# The Python environment is created on the Pi's own disk (default ~/.venvs/savior_glass), not inside the
# project: a project kept on a FAT / exFAT USB drive cannot hold the symbolic links a venv needs
# ("Operation not permitted: 'lib' -> '.../.venv/lib64'"). Set GLASS_VENV to use another place.
# --system-site-packages lets the venv see the apt packages for the camera and GPIO (picamera2, lgpio).
set -eu
ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
VENV="${GLASS_VENV:-$HOME/.venvs/savior_glass}"
cd "$ROOT"
mkdir -p "$(dirname "$VENV")"
python3 -m venv --system-site-packages "$VENV"
. "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Natural voices and voice commands are optional: a failure here must not stop the setup.
python -m pip install -r requirements-speech.txt || echo "Speech packages not installed: the glass will use espeak-ng and the buttons."
python scripts/get_speech_models.py || echo "Some speech models are missing (see above)."
python scripts/pi5_preflight.py
echo "On this Pi, currency mode uses best_int8.onnx and the watermark check is on."
echo "Activate in a new terminal:  . $VENV/bin/activate"
echo "Benchmark: python scripts/benchmark_pi5.py --iters 100 --sustained 30"
echo "Hear the voice: python scripts/speech_check.py"
echo "App: python main.py"
