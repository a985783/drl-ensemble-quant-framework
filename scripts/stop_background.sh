#!/bin/bash
# Stop the trading daemon

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PID=$(pgrep -f "crypto_trader/start_simulation.py")

if [ -z "$PID" ]; then
    echo "⚠️  Daemon is not running."
    exit 0
fi

echo "🛑 Stopping trading daemon (PID: $PID)..."
kill $PID

# Wait for process to exit
sleep 2
if pgrep -f "crypto_trader/start_simulation.py" > /dev/null; then
    echo "⚠️  Process didn't exit gracefully, force killing..."
    kill -9 $PID
fi

echo "✅ Daemon stopped."
