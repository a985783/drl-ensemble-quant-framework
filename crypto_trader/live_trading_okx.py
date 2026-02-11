
import ccxt
import os
import sys
import json
import csv
import time
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

# Execution Safety (Phase 1)
try:
    from execution_safety import (
        generate_action_id, load_state as load_safety_state, save_state as save_safety_state,
        is_action_pending, is_action_completed, register_action, update_action_status,
        complete_action, enter_safe_mode, exit_safe_mode, is_safe_mode, can_execute_action,
        record_api_success, record_api_failure, check_clock_drift, reconcile,
        get_local_position, set_local_position,
        get_safety_status, TradingState
    )
    HAS_SAFETY = True
except ImportError:
    HAS_SAFETY = False
    print("[WARN] execution_safety module not available")

try:
    from alerting import alerts
except ImportError:
    try:
        from crypto_trader.alerting import alerts
    except ImportError:
        class _Alerts:
            def send(self, *args, **kwargs):
                return

            def send_trade(self, *args, **kwargs):
                return

        alerts = _Alerts()

# Rollout Controller (Phase 4)
try:
    from rollout_controller import (
        get_rollout_level, get_active_model_path, record_trade,
        maybe_promote, load_registry, save_registry, get_status
    )
    HAS_ROLLOUT = True
except ImportError:
    HAS_ROLLOUT = False
    print("[WARN] rollout_controller module not available")


# 配置
# 使用绝对路径，确保从任何位置运行都能正确加载
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # 项目根目录

from config import get_default_config
config = get_default_config()

DATA_SYMBOL = config.data.symbol
MODELS_DIR = os.path.join(BASE_DIR, "checkpoints/ensemble")
SEEDS = config.seed.ensemble_seeds
STATE_FILE = os.path.join(BASE_DIR, "trading_state.json")  # 记录策略状态 (last_flip_time 等)
INITIAL_CAPITAL_FILE = os.path.join(BASE_DIR, "initial_capital.json")


class OKXTrader:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('OKX_API_KEY')
        self.secret_key = os.getenv('OKX_SECRET_KEY')
        self.passphrase = os.getenv('OKX_PASSPHRASE')
        self.is_demo = os.getenv('OKX_DEMO_MODE') == 'True'
        self.margin_mode = os.getenv('OKX_MARGIN_MODE')
        self.position_mode = os.getenv('OKX_POSITION_MODE')
        
        self.exchange = ccxt.okx({
            'apiKey': self.api_key,
            'secret': self.secret_key,
            'password': self.passphrase,
            'enableRateLimit': True,
        })
        if self.is_demo:
            self.exchange.set_sandbox_mode(True)
        self.verify_environment()

        try:
            self.exchange.load_markets()
            msg = f"【连接】OKX API 连接成功 (Demo: {self.is_demo})"
            print(msg)
            alerts.send("INFO", msg)
        except Exception as e:
            msg = f"【错误】连接失败: {e}"
            print(msg)
            alerts.send("ERROR", msg)

        self.verify_account_mode()
            
        self.loader = DataLoader()
        self.engineer = FeatureEngineer()
        self.risk_manager = RiskManager(max_drawdown_limit=0.15, freeze_period_steps=1)
        
        # Phase 4: Shadow mode
        self.shadow_mode = False
        self.rollout_level = 1.0
        self.models_dir = MODELS_DIR
        if HAS_ROLLOUT:
            try:
                active_path = get_active_model_path()
                if not os.path.isabs(active_path):
                    active_path = os.path.join(BASE_DIR, active_path)
                if active_path.endswith(".zip"):
                    active_path = os.path.dirname(active_path)
                self.models_dir = active_path
            except Exception as e:
                print(f"[WARN] Failed to resolve active model path: {e}")
        
        # 策略参数 (与 Phase B TradingEnv 保持一致)
        self.TAU = 0.25         # Hysteresis
        self.DELTA_MAX = 0.15   # Slew Rate
        self.COOLDOWN_DAYS = 3  # Cooldown
        self.atr_floor = config.risk.atr_floor
        self.vol_scale_min = config.risk.vol_scale_min
        self.vol_scale_max = config.risk.vol_scale_max
        self.max_slippage_risk_on = config.risk.max_slippage_risk_on
        self.expected_contract_size = 0.1
        self.contract_size_tolerance = 1e-6
        self.log_file = "trade_logs.csv"
        self.init_logger()
        
    def verify_environment(self):
        if not self.is_demo:
            confirm_flag = os.getenv('CONFIRM_REAL_MONEY')
            if confirm_flag != 'True':
                msg = (
                    "⛔️【严重安全拦截】\n"
                    "检测到您正在运行实盘模式 (OKX_DEMO_MODE=False)。\n"
                    "为了防止误操作，请在 .env 文件中设置 CONFIRM_REAL_MONEY=True 以确认解锁。"
                )
                print(msg)
                alerts.send("ERROR", msg)
                sys.exit(1)
            alerts.send("WARN", "⚠️ 注意：正在运行实盘模式！资金已接管。")

    def verify_account_mode(self):
        expected_margin_mode = (self.margin_mode or "cross").strip().lower()
        expected_position_mode = (self.position_mode or "net").strip().lower()

        try:
            res = self.exchange.private_get_account_config()
            data = res['data'][0]
            current_pos_mode = data['posMode']  # 'net_mode' or 'long_short_mode'
            
            # Check Position Mode
            is_net = current_pos_mode == 'net_mode'
            if expected_position_mode == 'net' and not is_net:
                msg = f"⛔️【账户模式错误】当前为双向持仓模式，策略要求【单向净持仓】(net_mode)。请在 OKX 手动修改。"
                print(msg)
                alerts.send("ERROR", msg)
                sys.exit(1)
            
            print(f"【安全】持仓模式校验通过: {current_pos_mode}")
            
            # For margin mode, we pass 'tdMode' in every order, which is the OKX recommended way.
            # No need to call set_margin_mode if we specify it per order.
            
        except Exception as e:
            print(f"【警告】无法自动校验账户模式: {e}")
            # Proceed with warning, but the per-order tdMode will still provide protection.

    def write_status(self, stage, equity=None, real_pos=None, drawdown=None, last_action=None, last_error=None):
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "stage": stage,
            "demo": self.is_demo,
            "equity": equity,
            "position": real_pos,
            "drawdown": drawdown,
            "last_action": last_action,
            "last_error": last_error,
        }
        path = os.path.join(BASE_DIR, "runs", "live_status.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(status, f)

    def write_heartbeat(self):
        path = os.path.join(BASE_DIR, "runs", "live_heartbeat.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(datetime.utcnow().isoformat())

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault("last_flip_timestamp", 0)
                    return data
            except json.JSONDecodeError as e:
                print(f"[WARN] Failed to load state, using defaults: {e}")
        return {"last_flip_timestamp": 0}

    def save_state(self, state):
        merged = {}
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    existing = json.load(f)
                if isinstance(existing, dict):
                    merged.update(existing)
            except json.JSONDecodeError:
                pass
        merged.update(state)
        temp_file = f"{STATE_FILE}.tmp"
        with open(temp_file, 'w') as f:
            json.dump(merged, f, indent=4)
        os.replace(temp_file, STATE_FILE)

    def fetch_market_data(self):
        print(f"【数据】正在获取 {DATA_SYMBOL} 历史数据...")
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        start = (pd.Timestamp.now() - pd.Timedelta(days=500)).strftime("%Y-%m-%d")
        
        df = self.loader.fetch_data(start, today, DATA_SYMBOL, interval="1d")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        processed_df = self.engineer.add_technical_indicators(df)
        
        # 加载预训练信号模型（与回测一致，不每日训练）
        predictor = SignalPredictor()
        model_path = os.path.join(BASE_DIR, "checkpoints", "signal_model.pkl")

        if not os.path.exists(model_path):
            raise RuntimeError(
                f"信号模型不存在: {model_path}\n"
                f"请先运行训练脚本生成信号模型，或从备份恢复。\n"
                f"训练命令: python crypto_trader/train_ensemble.py"
            )

        try:
            predictor = SignalPredictor.load(model_path)
            print(f"【信号模型】已加载预训练模型: {model_path}")
        except Exception as e:
            raise RuntimeError(f"信号模型加载失败: {e}")

        probs = predictor.predict_proba(processed_df)
        processed_df['Signal_Proba'] = probs
        
        return processed_df

    def safe_float(self, value, default=0.0):
        try:
            if value is None: return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _get_contract_size(self):
        try:
            market = self.exchange.market('ETH/USDT:USDT')
            if not market:
                self.exchange.load_markets()
                market = self.exchange.market('ETH/USDT:USDT')
        except Exception as e:
            msg = f"【错误】无法获取市场信息: {e}"
            print(msg)
            alerts.send("ERROR", msg)
            raise RuntimeError(msg)

        contract_size = self.safe_float(market.get('contractSize') if market else None, 0.0)
        if contract_size <= 0:
            msg = "【严重】交易所未返回 contractSize，拒绝交易以避免数量级错误"
            print(msg)
            alerts.send("ERROR", msg)
            raise RuntimeError(msg)

        if self.expected_contract_size is not None:
            if abs(contract_size - self.expected_contract_size) > self.contract_size_tolerance:
                msg = (
                    "【严重】contractSize 与预期不一致，已拒绝交易。 "
                    f"exchange={contract_size}, expected={self.expected_contract_size}"
                )
                print(msg)
                alerts.send("ERROR", msg)
                raise RuntimeError(msg)

        return contract_size

    def _risk_on_slippage_ok(self, side, reference_price):
        if self.max_slippage_risk_on is None or self.max_slippage_risk_on <= 0:
            return True, 0.0
        try:
            ticker = self.exchange.fetch_ticker('ETH/USDT:USDT')
            best_price = ticker['ask'] if side == 'buy' else ticker['bid']
            slippage = abs(best_price - reference_price) / reference_price if reference_price > 0 else 0.0
            return slippage <= self.max_slippage_risk_on, slippage
        except Exception as e:
            msg = f"【警告】无法获取盘口价格用于滑点校验: {e}"
            print(msg)
            alerts.send("WARN", msg)
            return False, None

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
                contract_size = self._get_contract_size()
                
                mark_price = self.safe_float(pos.get('markPrice'), 0.0)
                if mark_price == 0:
                    try:
                        ticker = self.exchange.fetch_ticker('ETH/USDT:USDT')
                        mark_price = self.safe_float(ticker.get('last'), 0.0)
                    except Exception:
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
            raise e

    def init_logger(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Net_Worth", "Real_Pos", 
                    "Raw_Signal", "ATR_Pct", "Vol_Scale", "Target_Intent", "Target_Leverage",
                    "Constraint_Reason", "Exec_Pos", 
                    "Action", "Contracts", "Limit_Price", "Avg_Fill_Price", 
                    "Slippage", "Fee", "Funding_Rate",
                    # Phase 1 Safety fields
                    "Action_ID", "Safe_Mode", "Safe_Reason", "Reconcile_Diff",
                    # Phase 4 Rollout fields
                    "Shadow", "Rollout_Level"
                ])

    def log_trade(self, data):
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                f"{data.get('equity', 0):.2f}",
                f"{data.get('real_pos', 0):.4f}",
                f"{data.get('raw_signal', 0):.4f}",
                f"{data.get('atr_pct', 0):.6f}",
                f"{data.get('vol_scale', 0):.2f}",
                f"{data.get('target_intent', 0):.4f}",
                f"{data.get('target_leverage', data.get('target_intent', 0)):.4f}",
                data.get('constraint', 'None'),
                f"{data.get('exec_pos', 0):.4f}",
                data.get('action', 'Hold'),
                data.get('contracts', 0),
                f"{data.get('limit_price', 0):.2f}",
                f"{data.get('fill_price', 0):.2f}",
                f"{data.get('slippage', 0):.4f}",
                f"{data.get('fee', 0):.4f}",
                f"{data.get('funding', 0):.6f}",
                # Phase 1 Safety fields
                data.get('action_id', ''),
                data.get('safe_mode', 'normal'),
                data.get('safe_reason', ''),
                data.get('reconcile_diff', 'OK'),
                # Phase 4 Rollout fields
                str(data.get('shadow', False)),
                f"{data.get('rollout_level', 1.0):.2f}"
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
        self.write_heartbeat()
        
        if equity <= 0:
            print("【风控巡检】余额为0，跳过策略执行")
            alerts.send("WARN", "⚠️ 账户余额归零，策略已暂停。")
            return 0, 0, self.load_state()

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

        # === 资金异常监控 ===
        last_equity = state.get('last_equity', equity)
        if last_equity > 0:
            equity_change = (equity - last_equity) / last_equity
            if abs(equity_change) > 0.05:  # 权益波动超过5%
                msg = f"🚨 【资金异常告警】权益剧烈波动: {equity_change:+.2%} | 当前: ${equity:,.2f} | 上次: ${last_equity:,.2f}"
                print(msg)
                alerts.send("ERROR", msg)

        # 保存当前权益用于下次比较
        state['last_equity'] = equity

        # === 回撤分级告警 ===
        if current_drawdown > 0.15:
            msg = f"🚨 【严重回撤告警】当前回撤: {current_drawdown*100:.2f}% | 已超过15%阈值，进入保命模式"
            print(msg)
            alerts.send("ERROR", msg)
        elif current_drawdown > 0.10:
            msg = f"⚠️ 【中度回撤告警】当前回撤: {current_drawdown*100:.2f}% | 已超过10%阈值"
            print(msg)
            alerts.send("WARN", msg)
        elif current_drawdown > 0.05:
            msg = f"ℹ️ 【轻度回撤提示】当前回撤: {current_drawdown*100:.2f}% | 已超过5%阈值"
            print(msg)
            alerts.send("INFO", msg)

        self.write_status("risk_check", equity=equity, real_pos=real_pos, drawdown=current_drawdown)
        self.save_state(state)

        # Log minimal status
        print(f"【风控巡检】Equity: ${equity:.0f} | DD: {current_drawdown*100:.2f}% | DayPnL: {daily_pnl_pct*100:.2f}% | Pos: {real_pos:.4f}")

        return equity, real_pos, state

    def check_daily_data_ready(self, max_retries=3, retry_interval=30):
        """
        检查日K线数据是否已更新
        确认最新K线日期等于当前UTC日期
        """
        from datetime import datetime, timezone

        print("【数据】检查日K线数据就绪状态...")

        for attempt in range(max_retries):
            try:
                # 获取最近几条日K线
                ohlcv = self.exchange.fetch_ohlcv(
                    DATA_SYMBOL,
                    timeframe='1d',
                    limit=3
                )

                if not ohlcv or len(ohlcv) < 2:
                    print(f"  [尝试 {attempt+1}/{max_retries}] 未获取到足够数据")
                    if attempt < max_retries - 1:
                        time.sleep(retry_interval)
                    continue

                # 获取最新K线的时间戳
                latest_candle_ts = ohlcv[-1][0]  # 毫秒时间戳
                latest_candle_time = datetime.fromtimestamp(
                    latest_candle_ts / 1000,
                    tz=timezone.utc
                )

                now_utc = datetime.now(timezone.utc)

                # 检查最新K线是否是今天的
                if latest_candle_time.date() == now_utc.date():
                    print(f"  ✅ 数据已就绪，最新K线时间: {latest_candle_time.strftime('%Y-%m-%d %H:%M')}")
                    return True
                else:
                    print(f"  [尝试 {attempt+1}/{max_retries}] 数据未就绪，"
                          f"最新K线: {latest_candle_time.date()}, "
                          f"当前UTC: {now_utc.date()}")

                    if attempt < max_retries - 1:
                        print(f"  等待 {retry_interval} 秒后重试...")
                        time.sleep(retry_interval)

            except Exception as e:
                print(f"  [尝试 {attempt+1}/{max_retries}] 检查失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_interval)

        print("⚠️ 数据就绪检查失败，但将继续尝试执行")
        return False

    def has_executed_today(self):
        """检查今天是否已经执行过策略"""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        cn_tz = ZoneInfo("Asia/Shanghai")
        today = datetime.now(cn_tz).strftime("%Y-%m-%d")
        status_file = os.path.join(BASE_DIR, "runs", f"daily_status_{today}.json")

        if os.path.exists(status_file):
            try:
                with open(status_file, 'r') as f:
                    data = json.load(f)
                if data.get("status") == "success":
                    print(f"【防重】今天({today})已成功执行过，跳过")
                    return True
                elif data.get("status") == "failed":
                    retry_count = data.get("retry_count", 0)
                    if retry_count >= 3:
                        print(f"【防重】今天已失败 {retry_count} 次，不再重试")
                        return True
            except Exception as e:
                print(f"[WARN] 读取执行状态失败: {e}")

        return False

    def mark_execution(self, status="success", details=None):
        """标记今日执行状态"""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        cn_tz = ZoneInfo("Asia/Shanghai")
        today = datetime.now(cn_tz).strftime("%Y-%m-%d")
        status_file = os.path.join(BASE_DIR, "runs", f"daily_status_{today}.json")

        os.makedirs(os.path.dirname(status_file), exist_ok=True)

        # 加载现有状态
        data = {}
        if os.path.exists(status_file):
            try:
                with open(status_file, 'r') as f:
                    data = json.load(f)
            except:
                pass

        # 更新状态
        data.update({
            "date": today,
            "status": status,
            "executed_at": datetime.now(cn_tz).isoformat(),
            "details": details or {}
        })

        if status == "failed":
            data["retry_count"] = data.get("retry_count", 0) + 1

        with open(status_file, 'w') as f:
            json.dump(data, f, indent=2)

    def run_strategy(self):
        """运行完整策略逻辑 (Signal -> Constraints -> Trade)"""
        print("\n" + "="*50)
        print("🚀 策略执行窗口 (Phase B Engine)")
        print("="*50 + "\n")

        # 0. 防重复执行检查
        if self.has_executed_today():
            print("【跳过】今日已执行，不再重复运行")
            return

        # 0.1 数据就绪检查
        data_ready = self.check_daily_data_ready(max_retries=3, retry_interval=30)
        if not data_ready:
            msg = "⚠️ 日K线数据未就绪，可能获取到旧数据"
            print(msg)
            alerts.send("WARN", msg)
            # 继续执行，但发出警告

        # 1. 状态同步
        equity, real_pos, state = self.run_once_risk_check() # Reuse risk check logic to get basic state
        last_flip_ts = state.get('last_flip_timestamp', 0)
        self.write_status("strategy_start", equity=equity, real_pos=real_pos)
        
        if equity < 50:
            msg = "【错误】余额不足 (<$50)"
            print(msg)
            alerts.send("ERROR", msg)
            return

        # 2. 获取数据与环境
        df = self.fetch_market_data()
        if df is None or df.empty:
            msg = "【错误】特征数据为空，跳过本次执行"
            print(msg)
            alerts.send("ERROR", msg)
            self.write_status("data_empty", equity=equity, real_pos=real_pos, last_error=msg)
            return
        current_data = df.iloc[-1]
        current_price = current_data['Close']
        current_vol = current_data['Rolling_Vol']
        current_atr = current_data['ATR']
        
        print(f"【行情】价格: ${current_price:.2f} | 波动率: {current_vol:.4f}")

        # 3. 构造 Observation
        env = TradingEnv(
            df,
            risk_manager=self.risk_manager,
            atr_floor=self.atr_floor,
            vol_scale_min=self.vol_scale_min,
            vol_scale_max=self.vol_scale_max
        )
        env.reset()
        
        # HACK: Manually set env state to match reality
        env.current_step = len(df) - 1
        env.pos = real_pos
        
        days_since = (datetime.now().timestamp() - last_flip_ts) / 86400
        steps_since = int(days_since) 
        if steps_since > 1000: steps_since = 1000
        env.last_flip_t = env.current_step - steps_since
        
        obs = env._get_observation().reshape(1, -1)
        
        # 4. 集成推理 (与回测完全一致)
        if HAS_ROLLOUT:
            try:
                active_path = get_active_model_path()
                if not os.path.isabs(active_path):
                    active_path = os.path.join(BASE_DIR, active_path)
                if active_path.endswith(".zip"):
                    active_path = os.path.dirname(active_path)
                self.models_dir = active_path
            except Exception as e:
                print(f"[WARN] Failed to refresh active model path: {e}")

        actions = []
        for seed in SEEDS:
            model_path = f"{self.models_dir}/ppo_seed_{seed}.zip"
            vec_path = f"{self.models_dir}/vec_norm_seed_{seed}.pkl"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file missing: {model_path}")
            if not os.path.exists(vec_path):
                raise FileNotFoundError(f"Normalization file missing: {vec_path}")
            
            model = PPO.load(model_path)
            
            # Load norm stats
            dummy_vec = DummyVecEnv([
                lambda: TradingEnv(
                    df[:100],
                    risk_manager=self.risk_manager,
                    atr_floor=self.atr_floor,
                    vol_scale_min=self.vol_scale_min,
                    vol_scale_max=self.vol_scale_max
                )
            ])
            vec_norm = VecNormalize.load(vec_path, dummy_vec)
            vec_norm.training = False
            vec_norm.norm_reward = False
            
            norm_obs = vec_norm.normalize_obs(obs[0])
            action, _ = model.predict(norm_obs, deterministic=True)
            actions.append(action[0] if len(action.shape)==1 else action[0][0])
            
        if not actions:
            print("【错误】未加载到任何模型，跳过本次执行")
            return

        # 5. 决策逻辑 (与回测一致：波动率缩放 -> 风控 -> 约束)
        raw_action = float(np.mean(actions))
        raw_action = float(np.clip(raw_action, -1.0, 1.0))

        current_atr_pct = (current_atr / current_price) if current_price > 0 and current_atr > 0 else 0.02
        if current_atr_pct < self.atr_floor:
            current_atr_pct = self.atr_floor
        vol_scale = float(np.clip(0.05 / current_atr_pct, self.vol_scale_min, self.vol_scale_max))

        target_intent = raw_action * vol_scale

        max_equity = state.get('max_equity', equity)
        if max_equity <= 0:
            max_equity = equity
        drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0.0
        self.write_status("pre_trade", equity=equity, real_pos=real_pos, drawdown=drawdown)

        risk_override = self.risk_manager.check_risk(drawdown, target_intent)
        risk_reason = "Normal"
        if risk_override is not None:
            target_intent = risk_override
            risk_reason = "RiskCap"

        # Phase 4: Apply rollout level (gradual deployment)
        if self.rollout_level < 1.0:
            original_pos = target_intent
            target_intent = target_intent * self.rollout_level
            print(f"【Rollout】仓位调整: {original_pos:.4f} × {self.rollout_level} = {target_intent:.4f}")

        target_intent = float(np.clip(target_intent, -1.0, 1.0))
        exec_pos, new_flip_ts, constraint_reason = self.calculate_constraints(
            target_intent, real_pos, last_flip_ts
        )

        if new_flip_ts != last_flip_ts:
            state['last_flip_timestamp'] = new_flip_ts
            self.save_state(state)

        if risk_reason != "Normal":
            if constraint_reason == "Normal":
                constraint_reason = risk_reason
            else:
                constraint_reason = f"{risk_reason}|{constraint_reason}"
            alerts.send("WARN", f"⚠️ 触发风控限制: {constraint_reason}")

        print(
            f"【模型】{len(actions)}模型输出均值: {raw_action:.4f} | 波动缩放: {vol_scale:.2f} "
            f"-> 目标仓位: {target_intent:.4f} | 执行仓位: {exec_pos:.4f}"
        )
        
        # Phase 4: Shadow mode indicator
        is_shadow = self.shadow_mode
        
        # === Phase 1: Execution Safety ===
        action_id = None
        safe_mode = "normal"
        safe_reason = ""
        reconcile_diff = "OK"
        reconcile_ok = True
        
        needs_trade = abs(exec_pos - real_pos) > 0.001

        if HAS_SAFETY and needs_trade:
            # Generate idempotent action key
            run_id = os.environ.get('RUN_ID', 'live')
            action_id = generate_action_id("ETH-USDT", exec_pos, time.time(), run_id)
            
            # Load safety state
            safety_state = load_safety_state()

            # Initialize local_position once (baseline) to avoid false mismatch on first run
            if getattr(safety_state, "local_position", None) is None:
                safety_state = set_local_position(real_pos, safety_state)
            
            # Check if action already pending or completed (idempotency)
            if is_action_pending(action_id, safety_state):
                print(f"【安全】Action {action_id} 已在执行中，跳过重复下单")
                return
            if is_action_completed(action_id, safety_state):
                print(f"【安全】Action {action_id} 已完成，跳过重复下单")
                return
            
            # Pre-execution reconciliation + clock drift check
            try:
                exchange_ts = self.exchange.fetch_time()
                safety_state = check_clock_drift(exchange_ts, safety_state)
            except Exception as e:
                print(f"【安全】时钟校验失败: {e}")
                record_api_failure(safety_state)

            # Pre-execution reconciliation
            try:
                open_orders = self.exchange.fetch_open_orders('ETH/USDT:USDT')
                local_pos = get_local_position(safety_state)
                recon_result = reconcile(
                    exchange_position=real_pos,
                    local_position=local_pos,
                    open_orders=open_orders,
                    state=safety_state
                )
                reconcile_diff = recon_result.summary()
                reconcile_ok = recon_result.is_consistent
                if not recon_result.is_consistent:
                    print(f"【安全】对账不一致: {reconcile_diff}")
                    # === 持仓异常告警 ===
                    msg = f"🚨 【持仓异常告警】本地与交易所持仓不一致 | 差异: {reconcile_diff} | 已进入SAFE_MODE"
                    alerts.send("ERROR", msg)
            except Exception as e:
                print(f"【安全】对账失败: {e}")
                record_api_failure(safety_state)
                reconcile_ok = False
            
            # Check SAFE_MODE and action permissions
            can_exec, reason = can_execute_action(exec_pos, real_pos, safety_state)
            safe_mode = "safe_mode" if is_safe_mode(safety_state) else "normal"
            safe_reason = reason
            
            if not can_exec:
                print(f"【安全】SAFE_MODE 阻止操作: {reason}")
                print(f"         目标仓位 {exec_pos:.4f} 需要加仓/开仓，当前仅允许 reduce-only")
                # Log blocked action
                log_data = {
                    'equity': equity, 'real_pos': real_pos, 'raw_signal': raw_action,
                    'atr_pct': current_atr_pct, 'vol_scale': vol_scale, 'target_intent': target_intent,
                    'target_leverage': target_intent,
                    'constraint': f'BLOCKED:{reason}', 'exec_pos': real_pos,
                    'action': 'Blocked', 'contracts': 0, 'limit_price': 0,
                    'fill_price': 0, 'slippage': 0, 'fee': 0, 'funding': 0,
                    'action_id': action_id, 'safe_mode': safe_mode,
                    'safe_reason': safe_reason, 'reconcile_diff': reconcile_diff
                }
                self.log_trade(log_data)
                return
            
            # Register action before execution
            safety_state = register_action(action_id, exec_pos, safety_state)
            save_safety_state(safety_state)
        
        # Phase 4: Shadow Mode - skip order execution
        if is_shadow:
            if needs_trade:
                print("🔮 【Shadow】跳过下单，仅记录信号")
            exec_info = {'contracts': 0, 'limit': 0, 'price': 0, 'slippage': 0, 'fee': 0}
        else:
            # 6. 执行交易 (使用约束后的执行仓位)
            if needs_trade:
                exec_info = self.execute_order(exec_pos, current_price, equity, action_id=action_id)
            else:
                exec_info = {'contracts': 0, 'limit': 0, 'price': 0, 'slippage': 0, 'fee': 0}
            
            if exec_info is None: exec_info = {}
        
        # Complete action in safety state - 只有成功成交才标记完成
        if HAS_SAFETY and action_id and needs_trade:
            safety_state = load_safety_state()
            fill_price = exec_info.get('price', 0)
            if fill_price > 0:
                # 成交成功，标记为完成
                safety_state = complete_action(action_id, safety_state, fill_price)
                # Update local position to executed target
                safety_state = set_local_position(exec_pos, safety_state)
                record_api_success(safety_state)
                print(f"✅ 订单成功，已记录 action_id: {action_id}")
                
                # Send Trade Alert
                alerts.send_trade(
                    action='BUY' if exec_pos > real_pos else 'SELL',
                    symbol='ETH/USDT',
                    price=fill_price,
                    amount=exec_info.get('contracts', 0),
                    equity=equity
                )
            else:
                # 成交失败，从pending中删除，允许下次重新下单
                if action_id in safety_state.pending_actions:
                    del safety_state.pending_actions[action_id]
                    print(f"⚠️ 订单失败或未成交，已清除pending状态")
            save_safety_state(safety_state)

        # Phase 4: Rollout KPI tracking
        if HAS_ROLLOUT and not is_shadow:
            trade_attempted = abs(exec_pos - real_pos) > 0.01
            if trade_attempted:
                try:
                    filled = exec_info.get('price', 0) > 0
                    slippage = exec_info.get('slippage', 0)
                    record_trade(filled=filled, slippage=slippage, reconcile_ok=reconcile_ok)
                    maybe_promote()
                except Exception as e:
                    print(f"[WARN] Rollout KPI update failed: {e}")

        # 7. Log
        log_data = {
            'equity': equity,
            'real_pos': real_pos,
            'raw_signal': raw_action,
            'atr_pct': current_atr_pct,
            'vol_scale': vol_scale,
            'target_intent': target_intent,
            'target_leverage': target_intent,
            'constraint': constraint_reason,
            'exec_pos': exec_pos,
            'action': 'Trade' if abs(exec_pos - real_pos) > 0.01 else 'Hold',
            'contracts': exec_info.get('contracts', 0),
            'limit_price': exec_info.get('limit', 0),
            'fill_price': exec_info.get('price', 0),
            'slippage': exec_info.get('slippage', 0),
            'fee': exec_info.get('fee', 0),
            'funding': 0,
            # Phase 1 Safety fields
            'action_id': action_id or '',
            'safe_mode': safe_mode,
            'safe_reason': safe_reason,
            'reconcile_diff': reconcile_diff,
            # Phase 4 Rollout fields
            'shadow': is_shadow,
            'rollout_level': self.rollout_level
        }
        self.log_trade(log_data)
        print("📝 日志已记录")
        self.write_status("post_trade", equity=equity, real_pos=real_pos, last_action=exec_pos)

        # 标记今日执行成功
        self.mark_execution("success", {
            "equity": equity,
            "position": exec_pos,
            "action": "Trade" if abs(exec_pos - real_pos) > 0.01 else "Hold",
            "data_ready": data_ready
        })


    def execute_order(self, target_leverage, price, equity, action_id=None):
        # 计算目标合约数量
        target_value = equity * target_leverage
        
        contract_size = self._get_contract_size()
        target_contracts = int(target_value / (price * contract_size))
        
        # 获取当前合约数
        _, _ = self.get_real_position() 
        positions = self.exchange.fetch_positions(['ETH/USDT:USDT'])
        current_contracts = 0
        current_notional = 0.0
        if positions:
            qty = float(positions[0]['contracts'])
            if positions[0]['side'] == 'short': qty = -qty
            current_contracts = int(qty)
            current_notional = abs(current_contracts) * contract_size * price
            
        diff = target_contracts - current_contracts
        intent = "risk_off" if abs(target_contracts) <= abs(current_contracts) else "risk_on"
        
        # ===== 详细下单日志 =====
        print("\n" + "="*60)
        print("💰 【下单详情】资金与仓位分析")
        print("="*60)
        
        # 1. 账户资金信息
        print(f"\n📊 账户概况:")
        print(f"   • 当前权益 (Equity): ${equity:,.2f}")
        print(f"   • 当前价格 (ETH): ${price:,.2f}")
        print(f"   • 合约面值: {contract_size} ETH/张")
        
        # 2. 当前仓位分析
        current_pos_pct = (current_notional / equity * 100) if equity > 0 else 0
        current_leverage = current_notional / equity if equity > 0 else 0
        position_direction = "多头" if current_contracts > 0 else ("空头" if current_contracts < 0 else "空仓")
        
        print(f"\n📍 当前仓位:")
        print(f"   • 持仓张数: {current_contracts} 张 ({position_direction})")
        print(f"   • 名义价值: ${current_notional:,.2f}")
        print(f"   • 资金占比: {current_pos_pct:.2f}% (相当于 {current_leverage:.2f}x 杠杆)")
        
        # 3. 目标仓位分析
        target_notional = abs(target_contracts) * contract_size * price
        target_pos_pct = (target_notional / equity * 100) if equity > 0 else 0
        target_leverage_calc = target_notional / equity if equity > 0 else 0
        target_direction = "多头" if target_contracts > 0 else ("空头" if target_contracts < 0 else "空仓")
        
        print(f"\n🎯 目标仓位:")
        print(f"   • 目标张数: {target_contracts} 张 ({target_direction})")
        print(f"   • 名义价值: ${target_notional:,.2f}")
        print(f"   • 资金占比: {target_pos_pct:.2f}% (相当于 {target_leverage_calc:.2f}x 杠杆)")
        print(f"   • 目标杠杆倍数: {abs(target_leverage):.4f}x")
        
        # 4. 本次操作详情
        order_notional = abs(diff) * contract_size * price
        order_pct = (order_notional / equity * 100) if equity > 0 else 0
        action_type = "加仓" if abs(target_contracts) > abs(current_contracts) else "减仓"
        if np.sign(target_contracts) != np.sign(current_contracts) and current_contracts != 0:
            action_type = "反手"
        elif target_contracts == 0:
            action_type = "平仓"
        elif current_contracts == 0:
            action_type = "开仓"
            
        print(f"\n📝 本次操作:")
        print(f"   • 操作类型: {action_type}")
        print(f"   • 变动张数: {diff:+d} 张 ({'买入' if diff > 0 else '卖出'})")
        print(f"   • 订单金额: ${order_notional:,.2f}")
        print(f"   • 占总资金: {order_pct:.2f}%")
        
        # 5. 仓位变化汇总
        pos_change = target_pos_pct - current_pos_pct
        leverage_change = target_leverage_calc - current_leverage
        
        print(f"\n📈 仓位变化:")
        print(f"   • 资金占比变化: {current_pos_pct:.2f}% → {target_pos_pct:.2f}% ({pos_change:+.2f}%)")
        print(f"   • 杠杆倍数变化: {current_leverage:.2f}x → {target_leverage_calc:.2f}x ({leverage_change:+.2f}x)")
        print("="*60 + "\n")
        
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
        params['tdMode'] = self.margin_mode
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
            # 1. Get Best Price
            ticker = self.exchange.fetch_ticker('ETH/USDT:USDT')
            best_price = ticker['bid'] if side == 'sell' else ticker['ask']
            
            # Limit Price Logic - 直接使用盘口价格，不加价（避免超出OKX价格限制）
            limit_price = best_price
            limit_price = float(self.exchange.price_to_precision('ETH/USDT:USDT', limit_price))
            
            # 尝试限价单
            print(f"【挂单】限价单: {side} {amount} 张 @ {limit_price}")
            try:
                limit_order = self.exchange.create_order(
                    symbol='ETH/USDT:USDT', 
                    type='limit', 
                    side=side, 
                    amount=amount, 
                    price=limit_price, 
                    params=params
                )
                order_id = limit_order['id']
            except Exception as limit_err:
                # 限价单失败，直接用市价单
                print(f"⚠️ 限价单失败: {limit_err}")
                print(f"【回退】直接使用市价单执行...")
                if intent == "risk_on":
                    ok, est_slippage = self._risk_on_slippage_ok(side, best_price)
                    if not ok:
                        msg = (
                            f"【风险保护】滑点过高，已取消加仓/开仓。 "
                            f"est_slippage={est_slippage}, limit={self.max_slippage_risk_on}"
                        )
                        print(msg)
                        alerts.send("WARN", msg)
                        return {
                            'price': 0,
                            'fee': 0,
                            'slippage': 0.0 if est_slippage is None else est_slippage,
                            'limit': best_price,
                            'contracts': 0
                        }
                mkt_order = self.exchange.create_order('ETH/USDT:USDT', 'market', side, amount, params=params)
                mkt_order_id = mkt_order['id']
                
                # 验证市价单是否成交
                time.sleep(2)  # 等待2秒让订单状态更新
                try:
                    order_status = self.exchange.fetch_order(mkt_order_id, 'ETH/USDT:USDT')
                    if order_status['status'] == 'closed':
                        mkt_avg = float(order_status['average']) if order_status.get('average') else best_price
                        fee = float(order_status['fee']['cost']) if order_status.get('fee') else 0.0
                        slippage = abs(mkt_avg - best_price) / best_price if best_price > 0 else 0.0
                        print(f"✅ 【市价成交】均价: {mkt_avg}")
                        return {'price': mkt_avg, 'fee': fee, 'slippage': slippage, 'limit': best_price, 'contracts': amount}
                    else:
                        # 未成交，撤销订单
                        print(f"⚠️ 市价单未立即成交，状态: {order_status['status']}，正在撤销...")
                        try:
                            self.exchange.cancel_order(mkt_order_id, 'ETH/USDT:USDT')
                            print(f"✅ 已撤销遗留订单: {mkt_order_id}")
                        except Exception as cancel_err:
                            print(f"⚠️ 撤销失败: {cancel_err}")
                        return {'price': 0, 'fee': 0, 'slippage': 0, 'limit': 0, 'contracts': 0}
                except Exception as fetch_err:
                    print(f"⚠️ 查询订单状态失败: {fetch_err}")
                    # 尝试撤销可能存在的订单
                    try:
                        self.exchange.cancel_order(mkt_order_id, 'ETH/USDT:USDT')
                    except Exception:
                        pass
                    return {'price': 0, 'fee': 0, 'slippage': 0, 'limit': 0, 'contracts': 0}
            
            # 等待限价单成交
            start_time = time.time()
            timeout = 60
            while (time.time() - start_time) < timeout:
                time.sleep(5)
                order = self.exchange.fetch_order(order_id, 'ETH/USDT:USDT')
                if order['status'] == 'closed':
                    fill_price = float(order['average']) if order['average'] else limit_price
                    fee = float(order['fee']['cost']) if order.get('fee') else 0.0
                    print(f"✅ 【成功】成交均价: {fill_price}")
                    return {'price': fill_price, 'fee': fee, 'slippage': 0.0, 'limit': limit_price, 'contracts': amount}
            
            # 超时，取消限价单并转市价
            try: self.exchange.cancel_order(order_id, 'ETH/USDT:USDT')
            except Exception: pass
            
            order = self.exchange.fetch_order(order_id, 'ETH/USDT:USDT')
            filled = float(order['filled'])
            remaining = amount - filled
            fee = 0.0
            fill_price = limit_price
            slippage = 0.0
            
            if filled > 0:
                fill_price = float(order['average']) if order['average'] else limit_price
                fee = float(order['fee']['cost']) if order.get('fee') else 0.0
            
            if remaining > 0:
                if intent == "risk_on":
                    ok, est_slippage = self._risk_on_slippage_ok(side, limit_price)
                    if not ok:
                        msg = (
                            f"【风险保护】滑点过高，已取消加仓/开仓。 "
                            f"est_slippage={est_slippage}, limit={self.max_slippage_risk_on}"
                        )
                        print(msg)
                        alerts.send("WARN", msg)
                        return {
                            'price': 0,
                            'fee': fee,
                            'slippage': 0.0 if est_slippage is None else est_slippage,
                            'limit': limit_price,
                            'contracts': 0
                        }
                print(f"【补单】剩余 {remaining} 张转市价")
                mkt_order = self.exchange.create_order('ETH/USDT:USDT', 'market', side, remaining, params=params)
                mkt_avg = float(mkt_order['average']) if mkt_order.get('average') else best_price
                mkt_fee = float(mkt_order['fee']['cost']) if mkt_order.get('fee') else 0.0
                
                # Weighted Average Price
                total_val = (filled * fill_price) + (remaining * mkt_avg)
                fill_price = total_val / amount
                fee += mkt_fee
                
                # Slippage
                slippage = abs(fill_price - limit_price) / limit_price
                print(f"✅ 【市价补单成交】均价: {mkt_avg}")
                
            return {'price': fill_price, 'fee': fee, 'slippage': slippage, 'limit': limit_price, 'contracts': amount}

        except Exception as e:
            print(f"❌ 【下单流程出错】: {e}")
            import traceback
            traceback.print_exc()
            return {'price': 0, 'fee': 0, 'slippage': 0, 'limit': 0, 'contracts': 0}

if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--auto', action='store_true', help="自动确认交易")
    parser.add_argument('--shadow', action='store_true', help="Shadow模式：只计算不下单")
    args = parser.parse_args()
    
    trader = OKXTrader()
    trader.auto_mode = args.auto
    trader.shadow_mode = args.shadow
    
    # Get rollout level
    if HAS_ROLLOUT:
        trader.rollout_level = get_rollout_level()
        print(f"📊 Rollout Level: {trader.rollout_level}")
    
    mode_str = "🔮 Shadow Mode" if args.shadow else "🤖 Live Mode"
    print(f"{mode_str} - OKX 智能交易机器人")
    print("="*50)
    
    try:
        trader.run_strategy()
        print("\n✅ 执行完成")
    except Exception as e:
        msg = f"\n❌ 执行出错: {e}"
        print(msg)
        alerts.send("ERROR", msg)
        import traceback
        traceback.print_exc()
