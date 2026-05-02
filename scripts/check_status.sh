#!/bin/bash
# Check status of trading daemon

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
LOG_FILE="${LOG_DIR}/daemon.log"

PID_LIST="$(pgrep -f "crypto_trader/start_simulation.py" || true)"
UID_NUM="$(id -u)"
TRADE_LABEL="com.rlmoe.trade.daily"
CHECK_LABEL="com.rlmoe.trade.checks"
MON_DAILY_LABEL="com.rlmoe.monitor.daily"
MON_WEEKLY_LABEL="com.rlmoe.monitor.weekly"

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
echo "🗓️  Launchd Jobs:"
echo "------------------------------------------"
for label in "${TRADE_LABEL}" "${CHECK_LABEL}" "${MON_DAILY_LABEL}" "${MON_WEEKLY_LABEL}"; do
    if launchctl print "gui/${UID_NUM}/${label}" >/dev/null 2>&1; then
        echo "✅ ${label}: LOADED"
    else
        echo "❌ ${label}: NOT_LOADED"
    fi
done

echo ""
echo "📜 Recent Logs ($LOG_FILE):"
echo "------------------------------------------"
if [ -f "$LOG_FILE" ]; then
    tail -n 10 "$LOG_FILE"
else
    echo "(Log file not found)"
fi
echo "=========================================="
