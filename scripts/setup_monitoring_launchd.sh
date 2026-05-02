#!/bin/bash
# Setup launchd schedules for strategy monitoring (cron fallback).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
ensure_project_root
require_python

DAILY_HOUR="${DAILY_MONITOR_HOUR:-9}"
DAILY_MINUTE="${DAILY_MONITOR_MINUTE:-20}"
WEEKLY_WEEKDAY="${WEEKLY_MONITOR_WEEKDAY:-1}"   # 1=Monday in launchd
WEEKLY_HOUR="${WEEKLY_MONITOR_HOUR:-9}"
WEEKLY_MINUTE="${WEEKLY_MONITOR_MINUTE:-30}"

AGENT_DIR="${HOME}/Library/LaunchAgents"
DAILY_LABEL="com.rlmoe.monitor.daily"
WEEKLY_LABEL="com.rlmoe.monitor.weekly"
DAILY_PLIST="${AGENT_DIR}/${DAILY_LABEL}.plist"
WEEKLY_PLIST="${AGENT_DIR}/${WEEKLY_LABEL}.plist"
UID_NUM="$(id -u)"

mkdir -p "${AGENT_DIR}" "${LOG_DIR}"

cat > "${DAILY_PLIST}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${DAILY_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-lc</string>
    <string>cd "${PROJECT_ROOT}" &amp;&amp; "${SCRIPT_DIR}/monitor_effectiveness_daily.sh" &gt;&gt; "${LOG_DIR}/effectiveness_daily.log" 2&gt;&amp;1</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>${DAILY_HOUR}</integer>
    <key>Minute</key>
    <integer>${DAILY_MINUTE}</integer>
  </dict>
  <key>RunAtLoad</key>
  <false/>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/effectiveness_daily.launchd.out.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/effectiveness_daily.launchd.err.log</string>
</dict>
</plist>
EOF

cat > "${WEEKLY_PLIST}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${WEEKLY_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-lc</string>
    <string>cd "${PROJECT_ROOT}" &amp;&amp; "${SCRIPT_DIR}/monitor_effectiveness_weekly.sh" &gt;&gt; "${LOG_DIR}/effectiveness_weekly.log" 2&gt;&amp;1</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key>
    <integer>${WEEKLY_WEEKDAY}</integer>
    <key>Hour</key>
    <integer>${WEEKLY_HOUR}</integer>
    <key>Minute</key>
    <integer>${WEEKLY_MINUTE}</integer>
  </dict>
  <key>RunAtLoad</key>
  <false/>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/effectiveness_weekly.launchd.out.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/effectiveness_weekly.launchd.err.log</string>
</dict>
</plist>
EOF

load_agent() {
  local label="$1"
  local plist="$2"
  launchctl bootout "gui/${UID_NUM}/${label}" >/dev/null 2>&1 || true
  if launchctl bootstrap "gui/${UID_NUM}" "${plist}" >/dev/null 2>&1; then
    return 0
  fi
  launchctl unload "${plist}" >/dev/null 2>&1 || true
  launchctl load -w "${plist}" >/dev/null 2>&1
}

load_agent "${DAILY_LABEL}" "${DAILY_PLIST}"
load_agent "${WEEKLY_LABEL}" "${WEEKLY_PLIST}"

echo "✅ launchd 监控任务已配置"
echo "Daily: ${DAILY_LABEL} @ ${DAILY_HOUR}:${DAILY_MINUTE}"
echo "Weekly: ${WEEKLY_LABEL} weekday=${WEEKLY_WEEKDAY} @ ${WEEKLY_HOUR}:${WEEKLY_MINUTE}"
echo ""
echo "LaunchAgents:"
echo "  ${DAILY_PLIST}"
echo "  ${WEEKLY_PLIST}"
echo ""
echo "当前任务状态:"
launchctl list | rg -n "${DAILY_LABEL}|${WEEKLY_LABEL}" -S || true
