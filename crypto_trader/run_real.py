#!/usr/bin/env python3
"""
OKX 实盘交易启动脚本
使用 .env.live 配置

⚠️ 警告：此脚本使用真实资金交易！
"""
import os
import sys
import shutil

# 切换到项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(BASE_DIR)

# 复制实盘配置
live_env = os.path.join(BASE_DIR, ".env.live")
target_env = os.path.join(BASE_DIR, ".env")

if os.path.exists(live_env):
    # 检查是否配置了真实 API Key
    with open(live_env, 'r') as f:
        content = f.read()
    if 'YOUR_LIVE_API_KEY' in content:
        print("❌ 错误：请先在 .env.live 中配置实盘 API Key")
        print("   位置: {}/".format(BASE_DIR))
        sys.exit(1)
    
    shutil.copy(live_env, target_env)
    print("⚠️  已加载实盘配置 (.env.live)")
    print("⚠️  警告：将使用真实资金交易！")
    print("")
else:
    print("❌ 找不到 .env.live 配置文件")
    sys.exit(1)

# 添加路径
sys.path.insert(0, SCRIPT_DIR)

# 运行主程序
from run_live import main
main()
