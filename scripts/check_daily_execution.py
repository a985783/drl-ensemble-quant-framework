#!/usr/bin/env python3
"""
Daily Execution Monitor
Check if trading strategy has been executed today by 8:30
Send alert if not executed
"""
import os
import sys
import json
from datetime import datetime
from zoneinfo import ZoneInfo

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)

from crypto_trader.alerting import AlertManager

# Configuration (system local schedule should align with setup_cron.sh defaults)
CHECK_HOUR = int(os.getenv("DAILY_CHECK_HOUR", "8"))
CHECK_MINUTE = int(os.getenv("DAILY_CHECK_MINUTE", "35"))

def get_cn_now():
    """Get current China time (UTC+8)"""
    return datetime.now(ZoneInfo("Asia/Shanghai"))


def check_daily_execution():
    """Check if strategy has been executed today"""
    cn_tz = ZoneInfo("Asia/Shanghai")
    today = datetime.now(cn_tz).strftime("%Y-%m-%d")
    status_file = os.path.join(BASE_DIR, "runs", f"daily_status_{today}.json")

    print(f"【监控】检查今日({today})执行状态...")

    if not os.path.exists(status_file):
        return False, "今日尚未执行"

    try:
        with open(status_file, 'r') as f:
            data = json.load(f)

        status = data.get("status")
        executed_at = data.get("executed_at", "未知")

        if status == "success":
            return True, f"今日已于 {executed_at} 成功执行"
        elif status == "blocked":
            details = data.get("details", {})
            reason = details.get("reason", "未知原因")
            return False, f"今日已执行，但被 SAFE_MODE 阻止: {reason}"
        elif status == "failed":
            retry_count = data.get("retry_count", 0)
            return False, f"今日执行失败，已重试 {retry_count} 次"
        else:
            return False, f"今日状态异常: {status}"

    except Exception as e:
        return False, f"读取状态文件失败: {e}"


def send_alert(message):
    """Send alert via configured channels"""
    try:
        alerts = AlertManager()
        alerts.send("ERROR", f"【日频监控告警】{message}")
        print(f"✅ 告警已发送: {message}")
    except Exception as e:
        print(f"❌ 发送告警失败: {e}")


def main():
    now = get_cn_now()

    # Check if it's time to run (after 8:30)
    if now.hour < CHECK_HOUR or (now.hour == CHECK_HOUR and now.minute < CHECK_MINUTE):
        print(f"未到检查时间，将在 {CHECK_HOUR}:{CHECK_MINUTE:02d} 后执行")
        return 0

    success, message = check_daily_execution()

    if success:
        print(f"✅ {message}")
        return 0
    else:
        alert_msg = f"⚠️ {message} - 请检查交易机器人状态！"
        print(alert_msg)
        send_alert(alert_msg)
        return 1


if __name__ == "__main__":
    sys.exit(main())
