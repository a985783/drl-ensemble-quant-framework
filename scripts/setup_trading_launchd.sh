#!/bin/bash
# Setup launchd schedules for daily trade execution + post-run checks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
ensure_project_root
require_python

CN_TRADE_HOUR="${CN_TRADE_HOUR:-8}"
CN_TRADE_MINUTE="${CN_TRADE_MINUTE:-0}"
CN_CHECK1_HOUR="${CN_CHECK1_HOUR:-8}"
CN_CHECK1_MINUTE="${CN_CHECK1_MINUTE:-35}"
CN_CHECK2_HOUR="${CN_CHECK2_HOUR:-9}"
CN_CHECK2_MINUTE="${CN_CHECK2_MINUTE:-0}"

convert_cn_to_local() {
  local cn_hour="$1"
  local cn_minute="$2"
  "${PYTHON_BIN}" - "${cn_hour}" "${cn_minute}" <<'PY'
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

cn_hour = int(sys.argv[1])
cn_minute = int(sys.argv[2])
cn_now = datetime.now(ZoneInfo("Asia/Shanghai"))
local_tz = datetime.now().astimezone().tzinfo
target_cn = cn_now.replace(hour=cn_hour, minute=cn_minute, second=0, microsecond=0)
target_local = target_cn.astimezone(local_tz)
tz_name = getattr(local_tz, "key", str(local_tz))
print(f"{target_local.hour} {target_local.minute} {tz_name}")
PY
}

read -r TRADE_HOUR TRADE_MINUTE LOCAL_TZ_NAME <<< "$(convert_cn_to_local "${CN_TRADE_HOUR}" "${CN_TRADE_MINUTE}")"
read -r CHECK1_HOUR CHECK1_MINUTE _ <<< "$(convert_cn_to_local "${CN_CHECK1_HOUR}" "${CN_CHECK1_MINUTE}")"
read -r CHECK2_HOUR CHECK2_MINUTE _ <<< "$(convert_cn_to_local "${CN_CHECK2_HOUR}" "${CN_CHECK2_MINUTE}")"

AGENT_DIR="${HOME}/Library/LaunchAgents"
TRADE_LABEL="com.rlmoe.trade.daily"
CHECK_LABEL="com.rlmoe.trade.checks"
TRADE_PLIST="${AGENT_DIR}/${TRADE_LABEL}.plist"
CHECK_PLIST="${AGENT_DIR}/${CHECK_LABEL}.plist"
UID_NUM="$(id -u)"

mkdir -p "${AGENT_DIR}" "${LOG_DIR}"

cat > "${TRADE_PLIST}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${TRADE_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-lc</string>
    <string>cd "${PROJECT_ROOT}" &amp;&amp; DOTENV_PATH=.env.live CONFIRM_REAL_MONEY=True "${SCRIPT_DIR}/daily_trade.sh" &gt;&gt; "${LOG_DIR}/launchd_trade.log" 2&gt;&amp;1</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>${TRADE_HOUR}</integer>
    <key>Minute</key>
    <integer>${TRADE_MINUTE}</integer>
  </dict>
  <key>RunAtLoad</key>
  <false/>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/launchd_trade.agent.out.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/launchd_trade.agent.err.log</string>
</dict>
</plist>
EOF

cat > "${CHECK_PLIST}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${CHECK_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-lc</string>
    <string>cd "${PROJECT_ROOT}" &amp;&amp; DOTENV_PATH=.env.live "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/check_daily_execution.py" &gt;&gt; "${LOG_DIR}/launchd_monitor.log" 2&gt;&amp;1</string>
  </array>
  <key>StartCalendarInterval</key>
  <array>
    <dict>
      <key>Hour</key>
      <integer>${CHECK1_HOUR}</integer>
      <key>Minute</key>
      <integer>${CHECK1_MINUTE}</integer>
    </dict>
    <dict>
      <key>Hour</key>
      <integer>${CHECK2_HOUR}</integer>
      <key>Minute</key>
      <integer>${CHECK2_MINUTE}</integer>
    </dict>
  </array>
  <key>RunAtLoad</key>
  <false/>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/launchd_check.agent.out.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/launchd_check.agent.err.log</string>
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

load_agent "${TRADE_LABEL}" "${TRADE_PLIST}"
load_agent "${CHECK_LABEL}" "${CHECK_PLIST}"

echo "✅ launchd 交易任务已配置"
echo "北京时间交易计划: ${CN_TRADE_HOUR}:${CN_TRADE_MINUTE} / 检查 ${CN_CHECK1_HOUR}:${CN_CHECK1_MINUTE}, ${CN_CHECK2_HOUR}:${CN_CHECK2_MINUTE}"
echo "本地时区(${LOCAL_TZ_NAME})换算: 交易 ${TRADE_HOUR}:${TRADE_MINUTE} / 检查 ${CHECK1_HOUR}:${CHECK1_MINUTE}, ${CHECK2_HOUR}:${CHECK2_MINUTE}"
echo ""
echo "LaunchAgents:"
echo "  ${TRADE_PLIST}"
echo "  ${CHECK_PLIST}"
echo ""
echo "当前任务状态:"
launchctl list | rg -n "${TRADE_LABEL}|${CHECK_LABEL}" -S || true
