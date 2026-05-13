#!/usr/bin/env python3
"""
OKX 模拟盘交易启动脚本
使用 .env.demo 配置
"""
import os
import sys
import shutil

# 切换到项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(BASE_DIR)

# 复制模拟盘配置
demo_env = os.path.join(BASE_DIR, ".env.demo")
target_env = os.path.join(BASE_DIR, ".env")

if os.path.exists(demo_env):
    shutil.copy(demo_env, target_env)
    print("✅ 已加载模拟盘配置 (.env.demo)")
else:
    print("❌ 找不到 .env.demo 配置文件")
    sys.exit(1)

# 添加路径
sys.path.insert(0, SCRIPT_DIR)

# 运行主程序
from run_live import main
main()
