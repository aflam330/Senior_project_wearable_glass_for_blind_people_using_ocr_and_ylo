#!/bin/sh
# Start the glass at boot, as this user, from this folder (systemd user service: no root, and the sound system
# of the user session is available to it).
#
#   bash scripts/install_service.sh           install and enable
#   systemctl --user start sara-glass         start now
#   journalctl --user -u sara-glass -f        watch its log
#   systemctl --user disable --now sara-glass remove
#
# To start before anyone logs in, lingering must be on for the user (asked for at the end).
set -eu
ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
VENV="${GLASS_VENV:-$HOME/.venvs/savior_glass}"
[ -x "$VENV/bin/python" ] || { echo "No Python environment at $VENV. Run scripts/deploy_pi5.sh first."; exit 1; }
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"
sed -e "s|@ROOT@|$ROOT|g" -e "s|@PYTHON@|$VENV/bin/python|g" "$ROOT/smart_glass.service" > "$UNIT_DIR/sara-glass.service"
systemctl --user daemon-reload
systemctl --user enable sara-glass.service
echo "Installed $UNIT_DIR/sara-glass.service"
loginctl enable-linger "$USER" 2>/dev/null && echo "Lingering on: the glass starts at boot without a login."     || echo "To start at boot without a login, run once:  sudo loginctl enable-linger $USER"
