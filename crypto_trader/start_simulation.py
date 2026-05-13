#!/usr/bin/env python3
"""
Daily Trading Scheduler
Runs crypto_trader/run_live.py --auto once per day at 08:00 China Time (UTC+8)
"""
import subprocess
import time
import sys
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Set working directory to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)

# Configuration
EXEC_HOUR = 8       # China Time 8:00
EXEC_MINUTE = 0     # 08:00
EXEC_WINDOW = 30    # Execution window: 08:00 - 08:30


def get_cn_now():
    """Get current China time (UTC+8)"""
    return datetime.now(ZoneInfo("Asia/Shanghai"))


def seconds_until_next_exec():
    """Calculate seconds until next execution time"""
    now = get_cn_now()
    exec_time = now.replace(hour=EXEC_HOUR, minute=EXEC_MINUTE,
                           second=0, microsecond=0)

    if now > exec_time:
        # Today's execution time has passed, calculate for tomorrow
        exec_time += timedelta(days=1)

    return (exec_time - now).total_seconds()


def is_in_exec_window():
    """Check if current time is within execution window"""
    now = get_cn_now()
    exec_start = now.replace(hour=EXEC_HOUR, minute=EXEC_MINUTE,
                            second=0, microsecond=0)
    exec_end = exec_start + timedelta(minutes=EXEC_WINDOW)
    return exec_start <= now <= exec_end


def has_executed_today():
    """Check if already executed today"""
    today = get_cn_now().strftime("%Y-%m-%d")
    status_file = os.path.join(PROJECT_ROOT, "runs", f"daily_status_{today}.json")
    return os.path.exists(status_file)


def mark_executed(status="success", details=None):
    """Mark today's execution status"""
    today = get_cn_now().strftime("%Y-%m-%d")
    status_file = os.path.join(PROJECT_ROOT, "runs", f"daily_status_{today}.json")

    import json
    data = {
        "date": today,
        "status": status,
        "executed_at": get_cn_now().isoformat(),
        "details": details or {}
    }

    os.makedirs(os.path.dirname(status_file), exist_ok=True)
    with open(status_file, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    print("="*60)
    print("   日频交易调度器 (Daily Trading Scheduler)")
    print(f"   执行时间: 中国时间 {EXEC_HOUR:02d}:{EXEC_MINUTE:02d} (UTC+8)")
    print(f"   执行窗口: {EXEC_WINDOW} 分钟")
    print("="*60)
    print("按 Ctrl+C 停止\n")

    try:
        while True:
            now = get_cn_now()
            today = now.strftime("%Y-%m-%d")

            # Check if in execution window and not executed today
            if is_in_exec_window() and not has_executed_today():
                print(f"\n{'='*40}")
                print(f"🚀 触发策略执行: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*40}")

                # Execute strategy
                result = subprocess.run(
                    [sys.executable, "crypto_trader/run_live.py", "--auto"],
                    cwd=PROJECT_ROOT
                )

                if result.returncode == 0:
                    mark_executed("success")
                    print(f"\n✅ 执行成功: {now.strftime('%H:%M:%S')}")
                else:
                    mark_executed("failed", {"returncode": result.returncode})
                    print(f"\n❌ 执行失败，返回码: {result.returncode}")

                # Calculate wait time until next execution
                sleep_sec = seconds_until_next_exec()
                print(f"下次执行: 约 {sleep_sec/3600:.1f} 小时后")

            else:
                # Show waiting info (once per hour)
                if now.minute == 0:
                    next_in = seconds_until_next_exec() / 3600
                    status = "已执行" if has_executed_today() else "待执行"
                    print(f"[{now.strftime('%H:%M')}] {status} | 距离下次: {next_in:.1f}小时")

                # Sleep briefly
                time.sleep(60)

    except KeyboardInterrupt:
        print("\n🛑 调度器已停止")
        sys.exit(0)


if __name__ == "__main__":
    main()
