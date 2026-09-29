#!/bin/bash
# ===========================================================================
# Smart Glass — Full Installation Script for Raspberry Pi 5
# ===========================================================================
# Run once as the 'pi' user (with sudo access):
#   chmod +x install.sh && ./install.sh
# ===========================================================================
set -euo pipefail

INSTALL_DIR="$HOME/smart-glass"
VENV_DIR="$HOME/smart-glass-env"
PIPER_VERSION="2023.11.14-2"

echo "============================================================"
echo " Smart Glass Installer for Raspberry Pi 5"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo "[1/7] Installing system packages..."
sudo apt update -qq
sudo apt install -y \
    python3-pip python3-venv python3-dev \
    libcamera-apps libcamera-tools \
    espeak-ng espeak-ng-data \
    alsa-utils alsa-tools \
    libatlas-base-dev libopenblas-dev \
    libhdf5-dev libhdf5-serial-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libgstreamer1.0-dev \
    build-essential cmake pkg-config \
    git wget curl

# ---------------------------------------------------------------------------
# 2. Python virtual environment
# ---------------------------------------------------------------------------
echo "[2/7] Creating Python virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR" --system-site-packages
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip wheel setuptools

# ---------------------------------------------------------------------------
# 3. Python packages
# ---------------------------------------------------------------------------
echo "[3/7] Installing Python packages (this takes ~10-15 min on first run)..."
pip install \
    "easyocr>=1.7.1" \
    "ultralytics>=8.3.0" \
    "opencv-python-headless>=4.8.0" \
    "rpi-lgpio>=0.5" \
    "torch>=2.1.0" \
    "torchvision>=0.16.0" \
    "numpy>=1.24.0" \
    "Pillow>=10.0.0"

# ---------------------------------------------------------------------------
# 4. Piper TTS — ARM64 binary + English voice model
# ---------------------------------------------------------------------------
echo "[4/7] Installing Piper TTS..."
PIPER_DIR="$INSTALL_DIR/models/piper"
mkdir -p "$PIPER_DIR"
cd /tmp

PIPER_ARCHIVE="piper_linux_aarch64.tar.gz"
if [ ! -f "$PIPER_ARCHIVE" ]; then
    wget -q --show-progress \
        "https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/${PIPER_ARCHIVE}"
fi
tar -xzf "$PIPER_ARCHIVE" -C /tmp/
sudo cp /tmp/piper/piper /usr/local/bin/piper
sudo chmod +x /usr/local/bin/piper
echo "   Piper binary installed at /usr/local/bin/piper"

# English Amy voice (low = ~30 MB, fastest on RPi 5)
cd "$PIPER_DIR"
EN_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low"
wget -q --show-progress "${EN_BASE}/en_US-amy-low.onnx"       -O en_US-amy-low.onnx
wget -q --show-progress "${EN_BASE}/en_US-amy-low.onnx.json"  -O en_US-amy-low.onnx.json
echo "   English voice downloaded"

echo "   Bangla TTS uses espeak-ng (installed above)"
echo "   Optional Piper Bangla: place bn_BD-medium.onnx + .json in $PIPER_DIR"

# ---------------------------------------------------------------------------
# 5. YOLOv8n model
# ---------------------------------------------------------------------------
echo "[5/7] Downloading YOLOv8n model..."
cd "$INSTALL_DIR"
source "$VENV_DIR/bin/activate"
python3 - <<'EOF'
from ultralytics import YOLO
import shutil, os
m     = YOLO('yolov8n.pt')
cache = os.path.expanduser('~/.cache/ultralytics/yolov8n.pt')
dest  = os.path.join(os.path.expanduser('~/smart-glass/models'), 'yolov8n.pt')
os.makedirs(os.path.dirname(dest), exist_ok=True)
if os.path.isfile(cache) and not os.path.isfile(dest):
    shutil.copy(cache, dest)
    print(f'  Copied YOLOv8n -> {dest}')
else:
    print('  YOLOv8n already in models/')
EOF

echo ""
echo "   Optional NCNN export for ~40% faster ARM inference:"
echo "     python3 -c \"from ultralytics import YOLO; YOLO('models/yolov8n.pt').export(format='ncnn')\""
echo "     Then set YOLO_MODEL_PATH = 'models/yolov8n_ncnn_model' in config.py"

# ---------------------------------------------------------------------------
# 6. Pi Camera 3 overlay
# ---------------------------------------------------------------------------
echo "[6/7] Configuring camera..."
BOOT_CFG="/boot/firmware/config.txt"
if [ -f "$BOOT_CFG" ]; then
    if ! grep -q "imx708" "$BOOT_CFG"; then
        echo "dtoverlay=imx708" | sudo tee -a "$BOOT_CFG" > /dev/null
        echo "   Pi Camera 3 overlay added. Reboot required."
    else
        echo "   Camera overlay already configured."
    fi
else
    echo "   $BOOT_CFG not found — skip (using USB camera?)"
fi

# ---------------------------------------------------------------------------
# 7. systemd service for auto-start on boot
# ---------------------------------------------------------------------------
echo "[7/7] Installing systemd service..."
SERVICE_FILE="$INSTALL_DIR/smart_glass.service"
if [ -f "$SERVICE_FILE" ]; then
    sed "s|/home/pi|$HOME|g" "$SERVICE_FILE" > /tmp/smart_glass.service
    sudo cp /tmp/smart_glass.service /etc/systemd/system/smart_glass.service
    sudo systemctl daemon-reload
    sudo systemctl enable smart_glass
    echo "   Service enabled. Starts automatically on boot."
    echo "   To start now: sudo systemctl start smart_glass"
else
    echo "   smart_glass.service not found — skipping auto-start setup"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " Installation complete!"
echo "============================================================"
echo ""
echo " Verify installation:"
echo "   1. Reboot:               sudo reboot"
echo "   2. Test camera:          libcamera-hello --nopreview -t 3000"
echo "   3. Test Bangla TTS:      espeak-ng -v bn 'স্মার্ট গ্লাস চালু'"
echo "   4. Test English TTS:     echo 'Hello' | piper --model $PIPER_DIR/en_US-amy-low.onnx --output-raw | aplay -r22050 -fS16_LE -c1 -"
echo "   5. Run manually:         source $VENV_DIR/bin/activate && cd $INSTALL_DIR && python3 main.py"
echo "   6. Check service logs:   sudo journalctl -u smart_glass -f"
echo ""
echo " Optional currency model training:"
echo "   python3 $INSTALL_DIR/scripts/train_currency.py --data-dir /path/to/taka_images"
echo ""
