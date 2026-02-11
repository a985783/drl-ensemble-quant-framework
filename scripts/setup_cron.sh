#!/bin/bash
# Setup cron jobs for daily trading at 8:05 China Time

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="/usr/bin/python3"

echo "=========================================="
echo "  日频交易定时任务配置"
echo "=========================================="
echo ""

# Check if cron is available
if ! command -v crontab &> /dev/null; then
    echo "❌ 错误: 未找到 crontab 命令"
    exit 1
fi

# Create temporary cron file
TEMP_CRON=$(mktemp)

# Export current crontab
crontab -l > "$TEMP_CRON" 2>/dev/null || echo "# New crontab" > "$TEMP_CRON"

# Remove existing entries for this project
grep -v "强化学习 i" "$TEMP_CRON" > "${TEMP_CRON}.new" || true
mv "${TEMP_CRON}.new" "$TEMP_CRON"

# Add new entries
cat >> "$TEMP_CRON" << EOF

# 日频交易定时任务 - 强化学习项目
# 每天8:05执行交易策略（服务器需设置为上海时区）
5 8 * * * cd "$PROJECT_DIR" && $PYTHON "$PROJECT_DIR/crypto_trader/run_live.py" --auto >> "$PROJECT_DIR/runs/daily_$(date +\%Y\%m\%d).log" 2>&1

# 每天8:35检查是否已执行，如未执行则告警
35 8 * * * cd "$PROJECT_DIR" && $PYTHON "$PROJECT_DIR/scripts/check_daily_execution.py" >> "$PROJECT_DIR/runs/monitor.log" 2>&1

# 每天9:00再次检查（备用）
0 9 * * * cd "$PROJECT_DIR" && $PYTHON "$PROJECT_DIR/scripts/check_daily_execution.py" >> "$PROJECT_DIR/runs/monitor.log" 2>&1
EOF

# Install new crontab
crontab "$TEMP_CRON"
rm -f "$TEMP_CRON"

echo "✅ 定时任务已配置:"
echo ""
echo "  1. 交易策略执行: 每天 8:05"
echo "     命令: python3 crypto_trader/run_live.py --auto"
echo ""
echo "  2. 执行状态检查: 每天 8:35 和 9:00"
echo "     命令: python3 scripts/check_daily_execution.py"
echo ""
echo "当前 crontab:"
crontab -l | grep -A 10 "强化学习"
echo ""
echo "⚠️  重要提示:"
echo "  1. 请确保服务器时区设置为 Asia/Shanghai (UTC+8)"
echo "     设置命令: sudo timedatectl set-timezone Asia/Shanghai"
echo ""
echo "  2. 日志文件位置:"
echo "     - 执行日志: runs/daily_YYYYMMDD.log"
echo "     - 监控日志: runs/monitor.log"
echo ""
echo "  3. 如需停止定时任务，运行: crontab -e 并删除相关行"
echo ""
