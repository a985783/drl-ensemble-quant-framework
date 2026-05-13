#!/bin/bash

# 强化学习交易系统 - 每日执行脚本 (安全增强版)
# 创建时间: $(date '+%Y-%m-%d %H:%M:%S')

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
ensure_project_root
require_python

# --- 配置 ---
PROJECT_DIR="${PROJECT_ROOT}"
PIDFILE="${PROJECT_DIR}/.daily_trade.lock"
LOG_DIR="${PROJECT_DIR}/runs"
LOG_FILE="$LOG_DIR/daily_$(date +%Y%m%d).log"
MAX_LOG_AGE=30  # 保留30天日志

# --- 防重复执行检查 ---
if [ -f "$PIDFILE" ]; then
    OLD_PID=$(cat "$PIDFILE" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[$(date)] ERROR: 另一个实例正在运行 (PID: $OLD_PID)，本次执行终止" | tee -a "$LOG_FILE"
        exit 1
    fi
fi

echo $$ > "$PIDFILE"
trap "rm -f '$PIDFILE'; exit" EXIT INT TERM

# --- 目录初始化 ---
mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

# --- 日志轮转清理 ---
find "$LOG_DIR" -name "daily_*.log" -type f -mtime +$MAX_LOG_AGE -delete 2>/dev/null || true
echo "[$(date)] 日志清理完成：删除超过 ${MAX_LOG_AGE} 天的旧日志" | tee -a "$LOG_FILE"

# --- 启动交易 ---
echo "==========================================" | tee -a "$LOG_FILE"
echo "[$(date)] 开始执行每日交易任务" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "[$(date)] Python: ${PYTHON_BIN}" | tee -a "$LOG_FILE"
echo "[$(date)] Project: ${PROJECT_DIR}" | tee -a "$LOG_FILE"
export DOTENV_PATH="${DOTENV_PATH:-.env.live}"
export CONFIRM_REAL_MONEY="${CONFIRM_REAL_MONEY:-True}"
echo "[$(date)] DOTENV_PATH: ${DOTENV_PATH}" | tee -a "$LOG_FILE"

# --- OKX 网络代理兜底 ---
# 某些网络环境下直连 OKX 会被重置；若本地 7897 代理可用，则自动启用。
if [ -z "${OKX_HTTPS_PROXY:-}" ] && nc -z 127.0.0.1 7897 2>/dev/null; then
    export OKX_HTTPS_PROXY="http://127.0.0.1:7897"
    echo "[$(date)] OKX_HTTPS_PROXY auto-enabled: ${OKX_HTTPS_PROXY}" | tee -a "$LOG_FILE"
fi

set +e
"${PYTHON_BIN}" crypto_trader/run_live.py --auto 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] ✓ 交易执行成功" | tee -a "$LOG_FILE"
else
    echo "[$(date)] ✗ 交易执行失败 (退出码: $EXIT_CODE)" | tee -a "$LOG_FILE"
    # 发送告警通知（如果配置了）
    "${PYTHON_BIN}" - <<PY 2>/dev/null || true
import os
import sys
sys.path.insert(0, os.path.abspath("."))
from crypto_trader.alerting import AlertManager
AlertManager().send("ERROR", f"【日频执行失败】退出码: ${EXIT_CODE}")
PY
fi

echo "[$(date)] 任务结束" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

exit $EXIT_CODE
