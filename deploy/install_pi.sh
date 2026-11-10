#!/usr/bin/env bash
# Sets the bot up as a background service on a Raspberry Pi (or any Linux box with systemd).
# Run it from the repo folder ON THE PI:
#
#   ./deploy/install_pi.sh          # paper trading (default)
#   ./deploy/install_pi.sh live     # real money; asks you to type LIVE
#
# The service starts on boot and restarts itself if it crashes.
set -euo pipefail

MODE="${1:-paper}"
if [[ "$MODE" != "paper" && "$MODE" != "live" ]]; then
  echo "Usage: $0 [paper|live]"; exit 1
fi
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
  echo "Python 3.10 or newer is needed (Raspberry Pi OS 'Bookworm' or later has it)."; exit 1; }

for f in .env.paper ".env.$MODE"; do
  if ! grep -q '^APCA_API_KEY_ID=..*' "$f" 2>/dev/null; then
    echo "Missing keys: copy $f from your Mac into $REPO first (see README)."; exit 1
  fi
done
chmod 600 .env.*

EXTRA=""
if [[ "$MODE" == "live" ]]; then
  read -r -p "This installs the bot to trade REAL money on every boot. Type LIVE to continue: " answer
  [[ "$answer" == "LIVE" ]] || { echo "Cancelled."; exit 1; }
  EXTRA=" --live --confirm LIVE"
fi

echo "Installing Python packages (can take several minutes on a Pi)..."
python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

sudo tee /etc/systemd/system/swingbot.service >/dev/null <<UNIT
[Unit]
Description=Alpaca swing bot ($MODE)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$REPO
ExecStart=$REPO/.venv/bin/python bot.py$EXTRA
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now swingbot.service
sleep 5
systemctl --no-pager --lines=5 status swingbot.service || true
cat <<DONE

Done. The bot is running in $MODE mode and will start by itself after a reboot.
  See what it holds:   .venv/bin/python bot.py --status$( [[ "$MODE" == "live" ]] && echo " --live" )
  Watch the log:       tail -f logs/$MODE.log
  Stop it:             sudo systemctl stop swingbot
  Start it again:      sudo systemctl start swingbot
  Remove it:           sudo systemctl disable --now swingbot
DONE
