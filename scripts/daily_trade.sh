#!/bin/bash

# 强化学习交易系统 - 每日执行脚本 (安全增强版)
# 创建时间: $(date '+%Y-%m-%d %H:%M:%S')

set -euo pipefail

# --- 配置 ---
PROJECT_DIR="/Users/cuiqingsong/Documents/强化学习 i"
VENV_DIR="$PROJECT_DIR/venv"
PIDFILE="$PROJECT_DIR/.daily_trade.lock"
LOG_DIR="$PROJECT_DIR/runs"
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

# 激活虚拟环境并执行
source "$VENV_DIR/bin/activate"
python crypto_trader/live_trading_okx.py --auto 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] ✓ 交易执行成功" | tee -a "$LOG_FILE"
else
    echo "[$(date)] ✗ 交易执行失败 (退出码: $EXIT_CODE)" | tee -a "$LOG_FILE"
    # 发送告警通知（如果配置了）
    if command -v python3 &> /dev/null; then
        python3 -c "import sys; sys.path.insert(0, 'crypto_trader'); from alerting import send_alert; send_alert('交易执行失败', f'退出码: $EXIT_CODE')" 2>/dev/null || true
    fi
fi

echo "[$(date)] 任务结束" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

exit $EXIT_CODE
