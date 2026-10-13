#!/bin/bash
# ===========================================================================
# SARA smart glass: setup on a Raspberry Pi 5 (Raspberry Pi OS Bookworm, 64-bit)
# ===========================================================================
#   bash install.sh            from savior_glass/, as the user who will run the glass
#
# 1. system packages (skipped with a note when sudo is not available)
# 2. Python environment, packages, speech models, model check   (scripts/deploy_pi5.sh)
# 3. optional: start the glass at boot                           (scripts/install_service.sh)
#
# The camera is a Camera Module 2 (IMX219); Raspberry Pi OS detects it by itself, no overlay is written.
# Rewritten on 14 October 2026: the earlier version installed into ~/smart-glass and set a Camera Module 3 overlay.
# ===========================================================================
set -eu
HERE="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"

echo "[1/3] System packages"
PKGS="python3-venv python3-dev python3-picamera2 python3-lgpio espeak-ng espeak-ng-data alsa-utils libportaudio2 libopenblas-dev"
if sudo -n true 2>/dev/null || [ -t 0 ]; then
    sudo apt update -qq && sudo apt install -y $PKGS || echo "   apt failed; continuing with what is installed"
else
    echo "   no sudo here: skipped. Needed packages: $PKGS"
fi

echo "[2/3] Python environment, packages and models"
sh "$HERE/scripts/deploy_pi5.sh"

echo "[3/3] Start at boot (optional)"
echo "   bash scripts/install_service.sh      installs a service for this user and this folder"
echo ""
echo "Check the voice:   python scripts/speech_check.py"
echo "Run the glass:     python main.py"
