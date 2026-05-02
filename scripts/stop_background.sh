#!/bin/bash
# Stop the trading daemon

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

PID_LIST="$(pgrep -f "crypto_trader/start_simulation.py" || true)"

if [ -z "$PID_LIST" ]; then
    echo "⚠️  Daemon is not running."
    exit 0
fi

echo "🛑 Stopping trading daemon (PID(s): ${PID_LIST//$'\n'/ })..."
while IFS= read -r pid; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
done <<< "$PID_LIST"

# Wait for process to exit
sleep 2
REMAINING="$(pgrep -f "crypto_trader/start_simulation.py" || true)"
if [ -n "$REMAINING" ]; then
    echo "⚠️  Process didn't exit gracefully, force killing..."
    while IFS= read -r pid; do
        [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
    done <<< "$REMAINING"
fi

echo "✅ Daemon stopped."
