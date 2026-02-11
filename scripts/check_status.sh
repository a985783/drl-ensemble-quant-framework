#!/bin/bash
# Check status of trading daemon

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_ROOT/runs/daemon.log"

PID_LIST=$(pgrep -f "crypto_trader/start_simulation.py")

echo "=========================================="
echo "  Trading Daemon Status"
echo "=========================================="

if [ -n "$PID_LIST" ]; then
    # Format PIDs for display (replace newlines with spaces)
    PID_DISPLAY=$(echo "$PID_LIST" | tr '\n' ' ')
    echo "✅ Status: RUNNING"
    echo "🆔 PIDs:   $PID_DISPLAY"
    
    # Get runtime of the first PID (likely the parent caffeinate process or python)
    FIRST_PID=$(echo "$PID_LIST" | head -n 1)
    PS_TIME=$(ps -p $FIRST_PID -o etime= | xargs)
    echo "⏱️  Uptime: $PS_TIME"
else
    echo "❌ Status: STOPPED"
fi

echo ""
echo "📜 Recent Logs ($LOG_FILE):"
echo "------------------------------------------"
if [ -f "$LOG_FILE" ]; then
    tail -n 10 "$LOG_FILE"
else
    echo "(Log file not found)"
fi
echo "=========================================="
