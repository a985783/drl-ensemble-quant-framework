#!/bin/bash
# One-shot migration: install launchd jobs, then remove project cron blocks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
ensure_project_root
require_python

CRON_MARKER_TRADE_BEGIN="# >>> RL_MOE_AUTOGEN_BEGIN >>>"
CRON_MARKER_TRADE_END="# <<< RL_MOE_AUTOGEN_END <<<"
CRON_MARKER_MON_BEGIN="# >>> RL_MOE_MONITOR_AUTOGEN_BEGIN >>>"
CRON_MARKER_MON_END="# <<< RL_MOE_MONITOR_AUTOGEN_END <<<"

echo "=========================================="
echo "  迁移到 launchd"
echo "=========================================="
echo "项目路径: ${PROJECT_ROOT}"
echo ""

"${SCRIPT_DIR}/setup_trading_launchd.sh"
"${SCRIPT_DIR}/setup_monitoring_launchd.sh"

if command -v crontab >/dev/null 2>&1; then
  TMP_CRON="$(mktemp)"
  TMP_CLEAN="$(mktemp)"
  crontab -l > "${TMP_CRON}" 2>/dev/null || true

  awk \
    -v b1="${CRON_MARKER_TRADE_BEGIN}" -v e1="${CRON_MARKER_TRADE_END}" \
    -v b2="${CRON_MARKER_MON_BEGIN}" -v e2="${CRON_MARKER_MON_END}" \
    '
      $0 == b1 {skip=1; end=e1; next}
      $0 == b2 {skip=1; end=e2; next}
      skip && $0 == end {skip=0; end=""; next}
      !skip {print}
    ' "${TMP_CRON}" > "${TMP_CLEAN}"

  crontab "${TMP_CLEAN}"
  rm -f "${TMP_CRON}" "${TMP_CLEAN}"
  echo "✅ 已清理项目 cron 自动块，避免重复执行"
else
  echo "ℹ️ 未检测到 crontab，跳过 cron 清理"
fi

echo ""
echo "当前 launchd 任务:"
launchctl list | rg -n "com\\.rlmoe\\.trade|com\\.rlmoe\\.monitor" -S || true
