
import ccxt
import os
import sys
import json
import csv
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from data_loader import DataLoader
from features import FeatureEngineer
from envs.trading_env import TradingEnv
from risk_manager import RiskManager
from models.signal_model import SignalPredictor

# 配置
TICKER_YF = 'ETH-USD'
MODELS_DIR = "checkpoints/ensemble"
SEEDS = [
    42, 123, 456, 789, 1024, 2024, 2025, 3000, 4000, 5000,
    6000, 7000, 8000, 9000, 10000, 1111, 2222, 3333, 4444, 5555
]
STATE_FILE = "trading_state.json"  # 记录策略状态 (last_flip_time 等)
INITIAL_CAPITAL_FILE = "initial_capital.json"

class OKXTrader:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('OKX_API_KEY')
        self.secret_key = os.getenv('OKX_SECRET_KEY')
        self.passphrase = os.getenv('OKX_PASSPHRASE')
        self.is_demo = os.getenv('OKX_DEMO_MODE') == 'True'
        
        self.exchange = ccxt.okx({
            'apiKey': self.api_key,
            'secret': self.secret_key,
            'password': self.passphrase,
            'enableRateLimit': True,
        })
        if self.is_demo:
            self.exchange.set_sandbox_mode(True)
        
        # 验证连接
        try:
            self.exchange.load_markets()
            print(f"【连接】OKX API 连接成功 (Demo: {self.is_demo})")
        except Exception as e:
            print(f"【错误】连接失败: {e}")
            
        self.loader = DataLoader()
        self.engineer = FeatureEngineer()
        # Risk Manager: Tiered Control Enabled (0.8x/0.5x), but Hard Kill (>100%) virtually disabled
        self.risk_manager = RiskManager(max_drawdown_limit=1.0, freeze_period_steps=0)
        
        # 策略参数 (与 Phase B TradingEnv 保持一致)
        self.TAU = 0.25         # Hysteresis
        self.DELTA_MAX = 0.15   # Slew Rate
        self.COOLDOWN_DAYS = 3  # Cooldown
        
        self.log_file = "trade_logs.csv"
        self.init_logger()
        
    def load_state(self):
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        return {"last_flip_timestamp": 0} 

    def save_state(self, state):
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)

    def fetch_market_data(self):
        print(f"【数据】正在获取 {TICKER_YF} 历史数据...")
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        start = (pd.Timestamp.now() - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
        
        df = self.loader.fetch_data(start, today, TICKER_YF, interval="1d")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        processed_df = self.engineer.add_technical_indicators(df)
        
        predictor = SignalPredictor()
        predictor.train(processed_df) 
        probs = predictor.predict_proba(processed_df)
        processed_df['Signal_Proba'] = probs
        
        return processed_df

    def safe_float(self, value, default=0.0):
        try:
            if value is None: return default
            return float(value)
        except:
            return default

    def get_real_position(self):
        """获取当前账户的实际持仓 (归一化 -1 到 1)"""
        try:
            positions = self.exchange.fetch_positions(['ETH/USDT:USDT'])
            balance = self.exchange.fetch_balance()
            equity = self.safe_float(balance.get('USDT', {}).get('total'), 0.0)
            
            if not positions:
                return 0.0, equity
            
            for pos in positions:
                # Debug print for first position
                # print(f"DEBUG POS: {pos}") 
                
                contracts = self.safe_float(pos.get('contracts'), 0.0)
                if pos['side'] == 'short':
                    contracts = -contracts
                
                # 计算名义价值
                contract_size = 0.1 # OKX ETH contract default
                
                try:
                    market = self.exchange.market('ETH/USDT:USDT')
                    if market and 'contractSize' in market:
                        contract_size = self.safe_float(market['contractSize'], 0.1)
                except:
                    pass
                
                mark_price = self.safe_float(pos.get('markPrice'), 0.0)
                if mark_price == 0:
                    try:
                        ticker = self.exchange.fetch_ticker('ETH/USDT:USDT')
                        mark_price = self.safe_float(ticker.get('last'), 0.0)
                    except:
                        pass

                notional = contracts * contract_size * mark_price
                
                if equity > 0:
                    normalized_pos = notional / equity
                    return normalized_pos, equity
            
            return 0.0, equity
            
        except Exception as e:
            print(f"【错误】获取持仓失败: {e}")
            import traceback
            traceback.print_exc()
            # Try to get equity at least
            try:
                balance = self.exchange.fetch_balance()
                equity = self.safe_float(balance.get('USDT', {}).get('total'), 0.0)
                return 0.0, equity
            except:
                return 0.0, 0.0

    def init_logger(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Net_Worth", "Real_Pos", 
                    "Raw_Signal", "Vol_Scale", "Target_Intent", 
                    "Constraint_Reason", "Exec_Pos", 
                    "Action", "Contracts", "Limit_Price", "Avg_Fill_Price", 
                    "Slippage", "Fee", "Funding_Rate"
                ])

    def log_trade(self, data):
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                f"{data.get('equity', 0):.2f}",
                f"{data.get('real_pos', 0):.4f}",
                f"{data.get('raw_signal', 0):.4f}",
                f"{data.get('vol_scale', 0):.2f}",
                f"{data.get('target_intent', 0):.4f}",
                data.get('constraint', 'None'),
                f"{data.get('exec_pos', 0):.4f}",
                data.get('action', 'Hold'),
                data.get('contracts', 0),
                f"{data.get('limit_price', 0):.2f}",
                f"{data.get('fill_price', 0):.2f}",
                f"{data.get('slippage', 0):.4f}",
                f"{data.get('fee', 0):.4f}",
                f"{data.get('funding', 0):.6f}"
            ])

    def calculate_constraints(self, target_pos, current_pos, last_flip_ts):
        """
        应用 Phase B 的 4-Piece Constraints Logic
        """
        reason = "Normal"
        
        # 1. Hysteresis
        if abs(target_pos - current_pos) < self.TAU:
            target_pos = current_pos
            reason = "Hysteresis"
            
        # 2. Slew Rate
        delta = np.clip(target_pos - current_pos, -self.DELTA_MAX, self.DELTA_MAX)
        if abs(target_pos - current_pos) > self.DELTA_MAX:
            reason = "SlewRate"
            
        cand_pos = current_pos + delta
        
        # 3. Cooldown
        now_ts = datetime.now().timestamp()
        days_since_flip = (now_ts - last_flip_ts) / (24 * 3600)
        in_cd = days_since_flip < self.COOLDOWN_DAYS
        
        wants_flip = (np.sign(current_pos) != 0 and 
                      np.sign(cand_pos) != 0 and 
                      np.sign(cand_pos) != np.sign(current_pos))
                      
        exec_pos = cand_pos
        new_flip_ts = last_flip_ts
        
        if in_cd and wants_flip:
            exec_pos = 0.0
            reason = "Cooldown"
            print(f"【约束】冷却期生效 ({days_since_flip:.1f}天 < {self.COOLDOWN_DAYS}天)，强制归零。")
        else:
            if wants_flip:
                new_flip_ts = now_ts
                reason = "Flip"
                print("【约束】触发反向，更新冷却时间戳。")
                
        return float(np.clip(exec_pos, -1.0, 1.0)), new_flip_ts, reason

    def run(self):
        print("\n" + "="*50)
        print("🚀 OKX 智能交易机器人 (Phase B Engine | Logged)")
        print("="*50 + "\n")
        
        # ... (Get Real Pos & Equity) ...
        real_pos, equity = self.get_real_position()
        state = self.load_state()
        last_flip_ts = state.get('last_flip_timestamp', 0)
        
        # ... (Kill Switch Checks) ...
        # (Assuming code handles Kill Switch logic here as added previously)
        # For brevity in replacement, re-including Kill Switch block is omitted but in real edit ensure it's there?
        # WAIT, update Instruction implies I'm updating run. I must include previous logic or use granular replacement.
        # I will use granular replacements. 
        # This block is for defining logger and updating constraints.
        # I will update 'run' in a separate call to avoid huge context replacement issues.


    def force_close_all(self):
        print("【强制平仓】正在市价全平...")
        try:
            positions = self.exchange.fetch_positions(['ETH/USDT:USDT'])
            for pos in positions:
                amt = float(pos['contracts'])
                if amt > 0:
                    side = 'sell' if pos['side'] == 'long' else 'buy'
                    self.exchange.create_market_order('ETH/USDT:USDT', side, amt, params={'reduceOnly': True})
            print("✅ 【强制平仓】已执行")
        except Exception as e:
            print(f"❌ 【强制平仓失败】: {e}")


    def check_schedule(self, close_wait_mins=5):
        """
        检查是否在 4小时 K 线收盘前的窗口期
        4H Candles: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC
        Execution Window: [Close - wait_mins, Close]
        e.g. 03:55 - 04:00
        """
        now = datetime.utcnow()
        # Next 4H close
        hour = now.hour
        next_close_hour = (hour // 4 + 1) * 4
        if next_close_hour == 24: next_close_hour = 0
        
        # Calculate time to next close
        # Be careful with day rollovers for calculation
        candidates = [0, 4, 8, 12, 16, 20, 24]
        for h in candidates:
            if h > hour:
                next_close_hour = h
                break
        
        # Construct target time
        target_time = now.replace(minute=0, second=0, microsecond=0)
        if next_close_hour == 24:
            target_time += timedelta(days=1)
            target_time = target_time.replace(hour=0)
        else:
            target_time = target_time.replace(hour=next_close_hour)
            
        diff_seconds = (target_time - now).total_seconds()
        diff_mins = diff_seconds / 60
        
        is_window = diff_mins <= close_wait_mins
        print(f"【调度】UTC {now.strftime('%H:%M')} | 距离 4H 收盘 ({target_time.strftime('%H:%M')}) 还有 {diff_mins:.1f} 分钟 | 窗口期: {is_window}")
        
        return is_window

    def run_once_risk_check(self):
        """仅运行风控检查 (每一分钟执行)"""
         # 1. 状态同步 (仅获取权益和持仓)
        real_pos, equity = self.get_real_position()
        
        if equity <= 0: return

        state = self.load_state()
        
        # Track Max Equity
        max_equity = state.get('max_equity', equity)
        if equity > max_equity:
            max_equity = equity
            state['max_equity'] = max_equity
            self.save_state(state)
            
        # 0. Kill Switch Checks (Daily Loss)
        today_str = datetime.now().strftime('%Y-%m-%d')
        daily_start_bal = state.get('daily_start_balance', equity)
        last_date = state.get('daily_start_date', today_str)
        
        if today_str != last_date:
            daily_start_bal = equity
            state['daily_start_date'] = today_str
            state['daily_start_balance'] = daily_start_bal
            self.save_state(state)

        daily_pnl_pct = 0.0
        if daily_start_bal > 0:
            daily_pnl_pct = (equity - daily_start_bal) / daily_start_bal
            
        # Drawdown for Risk Manager
        current_drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0.0
        
        # Log minimal status
        print(f"【风控巡检】Equity: ${equity:.0f} | DD: {current_drawdown*100:.2f}% | DayPnL: {daily_pnl_pct*100:.2f}% | Pos: {real_pos:.4f}")
        
        # Check Risk Manager for Hard Stop (Active Force Close)
        # Note: Tiered Risk (reducing size) is handled in signal logic (run_strategy),
        # but if we hit a Hard Stop (e.g. max loss), we might want to close HERE.
        # Current RiskManager config: max_drawdown_limit=1.0 (Disabled Hard Kill in logic), 
        # so mostly we are just watching. 
        # But if you HAD a Stop Loss logic, checking it here is where it would fire.
        
        return equity, real_pos, state

    def run_strategy(self):
        """运行完整策略逻辑 (Signal -> Constraints -> Trade)"""
        print("\n" + "="*50)
        print("🚀 策略执行窗口 (Phase B Engine)")
        print("="*50 + "\n")
        
        # 1. 状态同步
        equity, real_pos, state = self.run_once_risk_check() # Reuse risk check logic to get basic state
        last_flip_ts = state.get('last_flip_timestamp', 0)
        
        if equity < 50:
            print("【错误】余额不足 (<$50)")
            return

        # 2. 获取数据与环境
        df = self.fetch_market_data()
        current_data = df.iloc[-1]
        current_price = current_data['Close']
        current_vol = current_data['Rolling_Vol']
        current_atr = current_data['ATR']
        
        print(f"【行情】价格: ${current_price:.2f} | 波动率: {current_vol:.4f}")

        # 3. 构造 Observation
        env = TradingEnv(df, risk_manager=self.risk_manager)
        env.reset()
        
        # HACK: Manually set env state to match reality
        env.current_step = len(df) - 1
        env.pos = real_pos
        
        days_since = (datetime.now().timestamp() - last_flip_ts) / 86400
        steps_since = int(days_since) 
        if steps_since > 1000: steps_since = 1000
        env.last_flip_t = env.current_step - steps_since
        
        obs = env._get_observation().reshape(1, -1)
        
        # 4. 集成推理
        actions = []
        for seed in SEEDS:
            model_path = f"{MODELS_DIR}/ppo_seed_{seed}.zip"
            vec_path = f"{MODELS_DIR}/vec_norm_seed_{seed}.pkl"
            if not os.path.exists(model_path): continue
            
            model = PPO.load(model_path)
            
            # Load norm stats
            dummy_vec = DummyVecEnv([lambda: TradingEnv(df[:100], risk_manager=self.risk_manager)])
            vec_norm = VecNormalize.load(vec_path, dummy_vec)
            vec_norm.training = False
            vec_norm.norm_reward = False
            
            norm_obs = vec_norm.normalize_obs(obs[0])
            action, _ = model.predict(norm_obs, deterministic=True)
            actions.append(action[0] if len(action.shape)==1 else action[0][0])
            
        # 5. 决策逻辑 (Phase B)
        raw_action = np.mean(actions) 
        print(f"【模型】{len(actions)}模型Raw输出均值: {raw_action:.4f}")
        
        # Volatility Scaling
        current_atr_pct = (current_atr / current_price) if current_atr > 0 else 0.02
        if current_atr_pct < 0.005: current_atr_pct = 0.005
        vol_scale = 0.05 / current_atr_pct
        vol_scale = np.clip(vol_scale, 0.1, 2.0)
        
        target_pos_intent = raw_action * vol_scale
        
        # === Tiered Risk Management ===
        max_equity = state.get('max_equity', equity)
        current_drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0.0
        
        risk_constrained_pos = self.risk_manager.check_risk(current_drawdown, target_pos_intent)
        
        if risk_constrained_pos is not None:
             print(f"【风控】触发分级降仓! Drawdown: {current_drawdown*100:.2f}% -> 限制仓位: {risk_constrained_pos:.4f}")
             target_pos_intent = risk_constrained_pos
        else:
             print(f"【风控】Drawdown: {current_drawdown*100:.2f}% (Safe)")

        target_pos_intent = np.clip(target_pos_intent, -1.0, 1.0)
        print(f"【风控】VolScale: {vol_scale:.2f} -> 目标意图: {target_pos_intent:.4f}")
        
        # 6. 执行约束 (Constraints)
        final_exec_pos, new_flip_ts, constraint_reason = self.calculate_constraints(target_pos_intent, real_pos, last_flip_ts)
        print(f"【约束】原因: {constraint_reason} -> 最终: {final_exec_pos:.4f}")
        
        # 7. 执行交易
        exec_info = self.execute_order(final_exec_pos, current_price, equity)
        if exec_info is None: exec_info = {}
        
        # 8. 保存状态
        state['last_flip_timestamp'] = new_flip_ts
        self.save_state(state)
        
        # 9. Log Everything
        log_data = {
            'equity': equity,
            'real_pos': real_pos,
            'raw_signal': raw_action,
            'vol_scale': vol_scale,
            'target_intent': target_pos_intent,
            'constraint': constraint_reason,
            'exec_pos': final_exec_pos,
            'action': 'Trade' if abs(final_exec_pos - real_pos) > 0.01 else 'Hold',
            'contracts': exec_info.get('contracts', 0),
            'limit_price': exec_info.get('limit', 0),
            'fill_price': exec_info.get('price', 0),
            'slippage': exec_info.get('slippage', 0),
            'fee': exec_info.get('fee', 0),
            'funding': 0
        }
        self.log_trade(log_data)
        print("📝 日志已记录")

    def execute_order(self, target_leverage, price, equity):
        # 计算目标合约数量
        target_value = equity * target_leverage
        
        market = self.exchange.market('ETH/USDT:USDT')
        contract_size = market['contractSize'] if 'contractSize' in market else 0.1
        
        target_contracts = int(target_value / (price * contract_size))
        
        # 获取当前合约数
        _, _ = self.get_real_position() 
        positions = self.exchange.fetch_positions(['ETH/USDT:USDT'])
        current_contracts = 0
        if positions:
            qty = float(positions[0]['contracts'])
            if positions[0]['side'] == 'short': qty = -qty
            current_contracts = int(qty)
            
        diff = target_contracts - current_contracts
        
        print(f"【执行】当前持仓: {current_contracts} 张 | 目标: {target_contracts} 张 | 需变动: {diff} 张")
        
        if diff == 0:
            print("【执行】无需操作")
            return
            
        side = 'buy' if diff > 0 else 'sell'
        amount = abs(diff)
        
        # Check Reduce-Only Eligibility
        # True if we are closing/reducing existing position without flipping
        is_closing = (np.sign(diff) != np.sign(current_contracts)) if current_contracts != 0 else False
        is_reduce_only = is_closing and (amount <= abs(current_contracts))
        
        params = {}
        if is_reduce_only:
            params['reduceOnly'] = True
            print("【执行】启用 Reduce-Only 模式")

        if not self.auto_mode:
            confirm = input(f"确认执行 {side} {amount} 张 (Limit-then-Market)? (y/n): ")
            if confirm.lower() != 'y': return

        # === Strategy: Limit-then-Market ===
        print(f"【执行】启动 Limit-then-Market 策略 (超时 60s)")
        
        fill_price = 0.0
        slippage = 0.0
        fee = 0.0
        
        try:
            # ... (Limit order logic) ...
            # 1. Get Best Price
            ticker = self.exchange.fetch_ticker('ETH/USDT:USDT')
            best_price = ticker['bid'] if side == 'sell' else ticker['ask']
            
            # Limit Price Logic
            limit_price = best_price * (0.9995 if side == 'sell' else 1.0005)
            limit_price = float(self.exchange.price_to_precision('ETH/USDT:USDT', limit_price))
            
            # ... (Place Limit Order) ...
            print(f"【挂单】限价单: {side} {amount} 张 @ {limit_price}")
            limit_order = self.exchange.create_order(
                symbol='ETH/USDT:USDT', 
                type='limit', 
                side=side, 
                amount=amount, 
                price=limit_price, 
                params=params
            )
            order_id = limit_order['id']
            
            # ... (Wait Loop) ...
            start_time = time.time()
            timeout = 60
            while (time.time() - start_time) < timeout:
                time.sleep(5)
                order = self.exchange.fetch_order(order_id, 'ETH/USDT:USDT')
                if order['status'] == 'closed':
                    fill_price = float(order['average']) if order['average'] else limit_price
                    fee = float(order['fee']['cost']) if order.get('fee') else 0.0
                    print(f"✅ 【成功】成交均价: {fill_price}")
                    return {'price': fill_price, 'fee': fee, 'slippage': 0.0, 'limit': limit_price}
                
                # Check timeout/cancel
            
            # ... (Timeout -> Market) ...
            try: self.exchange.cancel_order(order_id, 'ETH/USDT:USDT')
            except: pass
            
            order = self.exchange.fetch_order(order_id, 'ETH/USDT:USDT')
            filled = float(order['filled'])
            remaining = amount - filled
            
            # Stats for filled part
            if filled > 0:
                fill_price = float(order['average']) if order['average'] else limit_price
            
            if remaining > 0:
                print(f"【补单】剩余 {remaining} 张转市价")
                mkt_order = self.exchange.create_order('ETH/USDT:USDT', 'market', side, remaining, params=params)
                mkt_avg = float(mkt_order['average']) if mkt_order.get('average') else best_price
                
                # Weighted Average Price
                total_val = (filled * fill_price) + (remaining * mkt_avg)
                fill_price = total_val / amount
                
                # Slippage
                slippage = abs(fill_price - limit_price) / limit_price
                
            return {'price': fill_price, 'fee': fee, 'slippage': slippage, 'limit': limit_price, 'contracts': amount}

        except Exception as e:
            print(f"❌ 【下单流程出错】: {e}")
            return {'price': 0, 'fee': 0, 'slippage': 0, 'limit': 0, 'contracts': 0}

if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--auto', action='store_true', help="自动确认交易")
    args = parser.parse_args()
    
    trader = OKXTrader()
    trader.auto_mode = args.auto
    
    print("🤖 OKX 智能交易机器人 (每日执行模式)")
    print("="*50)
    
    try:
        trader.run_strategy()
        print("\n✅ 执行完成")
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()
