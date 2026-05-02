#!/bin/bash
# 日志清理脚本 - 可手动或定时执行

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

PROJECT_DIR="${PROJECT_ROOT}"
LOG_DIR="${LOG_DIR}"
MAX_AGE=30

echo "正在清理超过 ${MAX_AGE} 天的日志文件..."
find "${LOG_DIR}" -name "daily_*.log" -type f -mtime +$MAX_AGE -delete
find "$PROJECT_DIR" -name "trade.log" -type f -size +100M -exec truncate -s 0 {} \; 2>/dev/null || true
echo "清理完成"
