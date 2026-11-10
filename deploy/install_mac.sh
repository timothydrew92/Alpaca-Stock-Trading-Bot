#!/usr/bin/env bash
# Runs the PAPER bot as a macOS background service: it starts when you log in and
# restarts by itself if it crashes. It cannot run while the Mac is asleep.
#
#   ./deploy/install_mac.sh          # install and start
#   ./deploy/install_mac.sh remove   # stop and uninstall
#
# Real-money trading is deliberately not offered here; use an always-on machine for that.
set -euo pipefail

LABEL="com.swingbot.paper"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DOMAIN="gui/$(id -u)"

launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
if [[ "${1:-}" == "remove" ]]; then
  rm -f "$PLIST"
  echo "Removed. The bot will no longer start by itself."
  exit 0
fi

[[ -x "$REPO/.venv/bin/python" ]] || { echo "Set up the virtual environment first (see README)."; exit 1; }
pkill -f "python bot.py" 2>/dev/null || true   # stop any copy started by hand
mkdir -p "$REPO/logs" "$HOME/Library/LaunchAgents"

cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/caffeinate</string><string>-i</string>
    <string>$REPO/.venv/bin/python</string><string>bot.py</string>
  </array>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$REPO/logs/paper.out</string>
  <key>StandardErrorPath</key><string>$REPO/logs/paper.out</string>
</dict>
</plist>
PLIST

launchctl bootstrap "$DOMAIN" "$PLIST"
echo "Installed. The paper bot now starts at login and restarts if it stops."
echo "  Status:  .venv/bin/python bot.py --status"
echo "  Log:     tail -f logs/paper.log"
echo "  Remove:  ./deploy/install_mac.sh remove"
