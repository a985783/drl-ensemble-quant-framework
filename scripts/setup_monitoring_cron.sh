#!/bin/bash
# Setup cron jobs for strategy effectiveness monitoring.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
ensure_project_root
require_python

MARKER_BEGIN="# >>> RL_MOE_MONITOR_AUTOGEN_BEGIN >>>"
MARKER_END="# <<< RL_MOE_MONITOR_AUTOGEN_END <<<"

# Local-time schedule defaults (override by env vars if needed)
DAILY_HOUR="${DAILY_MONITOR_HOUR:-9}"
DAILY_MINUTE="${DAILY_MONITOR_MINUTE:-20}"
WEEKLY_DAY="${WEEKLY_MONITOR_DAY:-MON}"
WEEKLY_HOUR="${WEEKLY_MONITOR_HOUR:-9}"
WEEKLY_MINUTE="${WEEKLY_MONITOR_MINUTE:-30}"

echo "=========================================="
echo "  Strategy Monitoring 定时任务配置"
echo "=========================================="
echo "项目路径: ${PROJECT_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "每日监控: ${DAILY_HOUR}:${DAILY_MINUTE} (本地时区)"
echo "每周周报: ${WEEKLY_DAY} ${WEEKLY_HOUR}:${WEEKLY_MINUTE} (本地时区)"
echo ""

if ! command -v crontab >/dev/null 2>&1; then
    echo "❌ 错误: 未找到 crontab 命令"
    exit 1
fi

TEMP_CRON="$(mktemp)"
TEMP_CLEAN="${TEMP_CRON}.clean"

crontab -l > "${TEMP_CRON}" 2>/dev/null || true

awk -v begin="${MARKER_BEGIN}" -v end="${MARKER_END}" '
  $0 == begin {skip=1; next}
  $0 == end {skip=0; next}
  !skip {print}
' "${TEMP_CRON}" > "${TEMP_CLEAN}"

cat >> "${TEMP_CLEAN}" <<EOF
${MARKER_BEGIN}
# Stable MoE effectiveness monitoring
${DAILY_MINUTE} ${DAILY_HOUR} * * * cd "${PROJECT_ROOT}" && "${SCRIPT_DIR}/monitor_effectiveness_daily.sh" >> "${LOG_DIR}/effectiveness_daily.log" 2>&1
${WEEKLY_MINUTE} ${WEEKLY_HOUR} * * ${WEEKLY_DAY} cd "${PROJECT_ROOT}" && "${SCRIPT_DIR}/monitor_effectiveness_weekly.sh" >> "${LOG_DIR}/effectiveness_weekly.log" 2>&1
${MARKER_END}
EOF

crontab "${TEMP_CLEAN}"
rm -f "${TEMP_CRON}" "${TEMP_CLEAN}"

echo "✅ 监控定时任务已配置"
echo "当前自动生成配置:"
crontab -l | awk -v begin="${MARKER_BEGIN}" -v end="${MARKER_END}" '
  $0 == begin {print; show=1; next}
  show {print}
  $0 == end {exit}
'
