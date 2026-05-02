#!/usr/bin/env python3
"""
OKX 实盘交易启动脚本
包含安全确认机制和账户状态显示
"""
import os
import sys
import argparse

# 添加项目路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.getenv("DOTENV_PATH", ".env"))

def print_banner():
    print("\n" + "="*60)
    print("   🚀 OKX 智能交易机器人 - 实盘模式")
    print("="*60)

def print_warning():
    is_demo = os.getenv('OKX_DEMO_MODE', 'True') == 'True'
    
    if not is_demo:
        print("\n" + "⚠️ "*20)
        print("\n   ⚠️  警告：您正在使用 【实盘模式】 ⚠️")
        print("   所有交易将使用真实资金！")
        print("\n" + "⚠️ "*20 + "\n")
        return True
    else:
        print("\n   ℹ️  当前为模拟盘模式 (Demo)")
        return False

def show_account_status(trader):
    """显示账户状态"""
    print("\n" + "-"*50)
    print("📊 账户状态")
    print("-"*50)
    
    try:
        real_pos, equity = trader.get_real_position()
        print(f"   • 账户权益: ${equity:,.2f}")
        print(f"   • 当前仓位: {real_pos:.4f}")
        
        # 获取当前价格
        ticker = trader.exchange.fetch_ticker('ETH/USDT:USDT')
        price = ticker.get('last', 0)
        print(f"   • ETH 价格: ${price:,.2f}")
        
        # 计算持仓价值
        position_value = abs(real_pos) * equity
        position_direction = "多头" if real_pos > 0 else ("空头" if real_pos < 0 else "空仓")
        print(f"   • 持仓方向: {position_direction}")
        print(f"   • 持仓价值: ${position_value:,.2f}")
        
        return True, equity, real_pos
    except Exception as e:
        print(f"   ❌ 获取账户状态失败: {e}")
        return False, 0, 0

def confirm_execution():
    """用户确认"""
    print("\n" + "-"*50)
    response = input("确认执行策略？(输入 'YES' 确认): ")
    return response.strip().upper() == 'YES'

def main():
    parser = argparse.ArgumentParser(description='OKX 实盘交易机器人')
    parser.add_argument('--auto', action='store_true', 
                       help='自动确认交易 (危险：跳过人工确认)')
    parser.add_argument('--check-only', action='store_true',
                       help='仅检查账户状态，不执行交易')
    parser.add_argument('--force-close', action='store_true',
                       help='强制平仓所有持仓')
    args = parser.parse_args()
    
    print_banner()
    is_live = print_warning()
    
    # 导入交易模块
    from live_trading_okx import OKXTrader
    trader = OKXTrader()
    trader.auto_mode = args.auto
    
    # 显示账户状态
    status_ok, equity, pos = show_account_status(trader)
    
    if args.check_only:
        if status_ok:
            print("\n✅ 账户检查完成")
            return 0
        print("\n❌ 账户检查失败（请先修正 .env.live 的 API/权限配置）")
        return 1
    
    if args.force_close:
        if is_live:
            confirm = input("\n⚠️ 确认强制平仓所有持仓？(输入 'CLOSE' 确认): ")
            if confirm.strip().upper() != 'CLOSE':
                print("已取消")
                return 0
        trader.force_close_all()
        return 0
    
    # 实盘模式下需要确认
    if is_live and not args.auto:
        if not confirm_execution():
            print("\n❌ 已取消执行")
            return 0
    
    # 执行策略
    print("\n" + "="*50)
    print("🎯 开始执行策略...")
    print("="*50)
    
    try:
        trader.run_strategy()
        print("\n✅ 策略执行完成")
        return 0
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()
        # 失败必须落状态，避免被监控误判为“未执行”
        try:
            trader.mark_execution("failed", {"error": str(e)})
        except Exception as mark_err:
            print(f"[WARN] 写入失败状态文件失败: {mark_err}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
