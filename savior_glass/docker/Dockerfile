# Savior Glass for Raspberry Pi 5 (linux/arm64).
# NOT BUILT OR TESTED: no Docker host or Pi 5 was available when this was written (2026-09-29).
#
# Build on the Pi (from the repository root, so the Taka models can be copied in):
#   docker build -f savior_glass/docker/Dockerfile -t savior-glass .
# Run (GPIO, camera and sound devices passed through):
#   docker run --rm -it --device /dev/gpiochip4 --device /dev/video0 --device /dev/snd \
#       -e ANTHROPIC_API_KEY= savior-glass
# The Pi 5's header GPIO is /dev/gpiochip4 on older kernels and /dev/gpiochip0 on newer ones;
# check with `gpiodetect`. A USB camera (/dev/video0) is the simple case; the CSI camera
# needs libcamera and is easier to use from the host install (install.sh).

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng espeak-ng-data alsa-utils \
        libgl1 libglib2.0-0 libgomp1 \
        tesseract-ocr tesseract-ocr-ben tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch wheels for aarch64, then the glass requirements.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY savior_glass/requirements.txt /app/savior_glass/requirements.txt
RUN pip install -r /app/savior_glass/requirements.txt onnxruntime psutil

# Application code and the Taka detector weights it loads from the sibling folder.
COPY savior_glass /app/savior_glass
COPY realtime_bangla_taka_detection/models/best.pt \
     realtime_bangla_taka_detection/models/best.onnx \
     realtime_bangla_taka_detection/models/best_int8.onnx \
     /app/realtime_bangla_taka_detection/models/

# Object-mode weights (yolov8s.pt / yolov8n.pt) and Piper voices are git-ignored; mount them:
#   -v /home/pi/smart-glass/models:/app/savior_glass/models
WORKDIR /app/savior_glass
CMD ["python", "main.py"]
