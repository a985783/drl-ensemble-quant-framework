#!/bin/bash
# Start the trading daemon in the background

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
ensure_project_root
require_python
LOG_FILE="${LOG_DIR}/daemon.log"

# Check if already running
PID="$(pgrep -f "crypto_trader/start_simulation.py" || true)"
if [ -n "$PID" ]; then
    echo "⚠️  Daemon is already running (PID: $PID)"
    echo "Logs are being written to: $LOG_FILE"
    exit 1
fi

echo "🚀 Starting REAL TRADING daemon (with caffeinate)..."
export CONFIRM_REAL_MONEY=True
export DOTENV_PATH=.env.live
nohup /usr/bin/caffeinate -d -i -m -s "${PYTHON_BIN}" -u crypto_trader/start_simulation.py > "$LOG_FILE" 2>&1 &

# Initial verification
sleep 2
PID="$(pgrep -f "crypto_trader/start_simulation.py" || true)"

if [ -n "$PID" ]; then
    echo "✅ Daemon started successfully! (PID: $PID)"
    echo "Logs are being written to: $LOG_FILE"
    echo "To check status: ./scripts/check_status.sh"
    echo "To stop daemon:  ./scripts/stop_background.sh"
else
    echo "❌ Failed to start daemon. Check logs:"
    cat "$LOG_FILE"
fi
