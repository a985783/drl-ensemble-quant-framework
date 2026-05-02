#!/bin/bash
# Setup cron jobs for stable MoE daily execution.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
ensure_project_root
require_python

MARKER_BEGIN="# >>> RL_MOE_AUTOGEN_BEGIN >>>"
MARKER_END="# <<< RL_MOE_AUTOGEN_END <<<"

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

# Backward compatible overrides:
# - explicit local schedule envs (TRADE_HOUR, CHECK*_HOUR) take priority.
# - otherwise derive local cron time from Beijing schedule.
if [ -n "${TRADE_HOUR:-}" ] || [ -n "${TRADE_MINUTE:-}" ] || \
   [ -n "${CHECK1_HOUR:-}" ] || [ -n "${CHECK1_MINUTE:-}" ] || \
   [ -n "${CHECK2_HOUR:-}" ] || [ -n "${CHECK2_MINUTE:-}" ]; then
    TRADE_HOUR="${TRADE_HOUR:-9}"
    TRADE_MINUTE="${TRADE_MINUTE:-0}"
    CHECK1_HOUR="${CHECK1_HOUR:-9}"
    CHECK1_MINUTE="${CHECK1_MINUTE:-35}"
    CHECK2_HOUR="${CHECK2_HOUR:-10}"
    CHECK2_MINUTE="${CHECK2_MINUTE:-0}"
    LOCAL_TZ_NAME="$(date +%Z)"
else
    read -r TRADE_HOUR TRADE_MINUTE LOCAL_TZ_NAME <<< "$(convert_cn_to_local "${CN_TRADE_HOUR}" "${CN_TRADE_MINUTE}")"
    read -r CHECK1_HOUR CHECK1_MINUTE _ <<< "$(convert_cn_to_local "${CN_CHECK1_HOUR}" "${CN_CHECK1_MINUTE}")"
    read -r CHECK2_HOUR CHECK2_MINUTE _ <<< "$(convert_cn_to_local "${CN_CHECK2_HOUR}" "${CN_CHECK2_MINUTE}")"
fi

echo "=========================================="
echo "  Stable MoE 定时任务配置"
echo "=========================================="
echo "项目路径: ${PROJECT_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "北京时间交易计划: ${CN_TRADE_HOUR}:${CN_TRADE_MINUTE} / 检查 ${CN_CHECK1_HOUR}:${CN_CHECK1_MINUTE}, ${CN_CHECK2_HOUR}:${CN_CHECK2_MINUTE}"
echo "本地时区(${LOCAL_TZ_NAME})转换后: 交易 ${TRADE_HOUR}:${TRADE_MINUTE} / 检查 ${CHECK1_HOUR}:${CHECK1_MINUTE}, ${CHECK2_HOUR}:${CHECK2_MINUTE}"
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
# Stable MoE daily jobs
${TRADE_MINUTE} ${TRADE_HOUR} * * * cd "${PROJECT_ROOT}" && DOTENV_PATH=.env.live CONFIRM_REAL_MONEY=True "${SCRIPT_DIR}/daily_trade.sh" >> "${LOG_DIR}/cron.log" 2>&1
${CHECK1_MINUTE} ${CHECK1_HOUR} * * * cd "${PROJECT_ROOT}" && DOTENV_PATH=.env.live "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/check_daily_execution.py" >> "${LOG_DIR}/monitor.log" 2>&1
${CHECK2_MINUTE} ${CHECK2_HOUR} * * * cd "${PROJECT_ROOT}" && DOTENV_PATH=.env.live "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/check_daily_execution.py" >> "${LOG_DIR}/monitor.log" 2>&1
${MARKER_END}
EOF

crontab "${TEMP_CLEAN}"
rm -f "${TEMP_CRON}" "${TEMP_CLEAN}"

echo "✅ 定时任务已配置:"
echo "  1. 每日交易: ${TRADE_HOUR}:${TRADE_MINUTE}"
echo "  2. 执行检查: ${CHECK1_HOUR}:${CHECK1_MINUTE} 和 ${CHECK2_HOUR}:${CHECK2_MINUTE}"
echo ""
echo "当前自动生成配置:"
crontab -l | awk -v begin="${MARKER_BEGIN}" -v end="${MARKER_END}" '
  $0 == begin {print; show=1; next}
  show {print}
  $0 == end {exit}
'
