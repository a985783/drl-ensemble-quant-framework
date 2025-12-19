"""
Trading Environment with Four-Piece Turnover Reduction Constraints
交易环境 - 含四重换手率约束

Phase B: Constraints integrated into env for PPO to learn within constraints
Phase B：约束集成到环境中，使 PPO 在约束内学习

Constraints | 约束机制:
1. Hysteresis (τ): Small changes don't trigger trades | 滞回：小幅变化不触发交易
2. Slew-rate limit (δ): Max position change per step | 限速：每步最大仓位变化
3. Cooldown (N): Min interval between sign flips | 冷却期：反向交易最小间隔
4. Real cost model: Fee + Slippage + Funding Rate | 真实成本模型
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent dir to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import environment_config


class TradingEnv(gym.Env):
    """
    Cryptocurrency Trading Environment with Execution Constraints.
    加密货币交易环境，内置执行约束。

    This environment implements a realistic trading simulation with:
    本环境实现了一个逼真的交易模拟，包含：
    
    State Space | 观察空间 (13 dimensions):
        - pos: Current normalized position [-1, 1] | 当前归一化仓位
        - cooldown_remaining: Remaining cooldown [0, 1] | 剩余冷却时间
        - unrealized_pnl: Unrealized profit/loss ratio | 未实现盈亏比率
        - nw_change: Step-wise net worth change | 单步净值变化
        - signal_proba: XGBoost prediction [0, 1] | XGBoost 预测值
        - rsi: Relative Strength Index (normalized) | 相对强弱指数
        - rolling_vol: Rolling volatility | 滚动波动率
        - macd: MACD value (normalized) | MACD 指标
        - bb_width: Bollinger Band width (normalized) | 布林带宽度
        - dist_sma: Distance to 200 SMA | 与200日均线距离
        - atr: Average True Range (normalized) | 平均真实波幅
        - vol_ratio: Volume ratio | 成交量比率
        - direction: Current position direction | 当前持仓方向

    Action Space | 动作空间:
        Continuous [-1, 1] representing target position intent
        连续值 [-1, 1] 代表目标仓位意图
        - -1: Full short position | 全仓做空
        -  0: No position (cash) | 空仓观望
        - +1: Full long position | 全仓做多

    Execution Constraints | 执行约束 (Configurable via config.yaml):
        - TAU: Hysteresis threshold | 滞回阈值
        - DELTA_MAX: Maximum position change per step | 每步最大仓位变化
        - COOLDOWN_N: Minimum steps between direction flips | 反向交易最小间隔
        - K_SINGLE: Single-side transaction cost | 单边交易成本
        - FUNDING_DAILY: Daily funding rate | 日资金费率
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, 
                 fee_rate: float = 0.001, risk_manager=None, enable_kill_switch: bool = True):
        """
        Initialize the trading environment.
        初始化交易环境。
        
        Args:
            df: DataFrame with OHLCV data and features
                包含 OHLCV 数据和特征的 DataFrame
            initial_balance: Starting capital in USD (default: 10000)
                            起始资金（美元，默认：10000）
            fee_rate: Base fee rate for transactions (default: 0.001)
                     交易基础费率（默认：0.001）
            risk_manager: Optional RiskManager instance for position limits
                         可选的 RiskManager 实例用于仓位限制
            enable_kill_switch: Enable intra-day stop loss simulation (default: True)
                               启用日内止损模拟（默认：True）
        """
        super(TradingEnv, self).__init__()
        
        # Load constraint parameters from config | 从配置加载约束参数
        env_cfg = environment_config()
        self.TAU = env_cfg.get('tau', 0.2)
        self.DELTA_MAX = env_cfg.get('delta_max', 0.1)
        self.COOLDOWN_N = env_cfg.get('cooldown_n', 5)
        self.K_SINGLE = env_cfg.get('k_single', 0.001)
        self.FUNDING_DAILY = env_cfg.get('funding_daily', 0.0003)
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.risk_manager = risk_manager
        self.enable_kill_switch = enable_kill_switch
        
        # Action Space: Continuous [-1, 1] (target position intent)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation Space: 
        # [Pos, Cooldown_Remaining, UnPnL, NW_Chg, Prob, RSI, Vol, MACD, BB_Width, Dist_SMA, ATR, Vol_Ratio]
        # Shape: (13,) - Added cooldown_remaining as new dimension
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        
        # Required columns check
        required_cols = ['Close', 'Signal_Proba', 'RSI', 'Rolling_Vol', 'MACD', 'BB_Width', 'Dist_SMA_200', 'ATR', 'Vol_Ratio']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"DataFrame missing required column: {col}")

        # State Variables
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares = 0.0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        
        # Position tracking
        self.pos = 0.0  # Current executed position [-1, 1]
        self.last_flip_t = -100  # Last time step with sign flip
        self.kill_switch_lockout = 0 # Steps to remain out of market
        
        # Legacy compatibility
        self.current_position = 0.0
        self.prev_action = 0.0
        
        self.returns_history = []
        self.negative_returns_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares = 0.0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        
        # Reset position state
        self.pos = 0.0
        self.last_flip_t = -100
        
        self.current_position = 0.0
        self.prev_action = 0.0
        self.returns_history = []
        self.negative_returns_history = []
        
        if self.risk_manager:
            self.risk_manager.reset()
            self.risk_manager.peak_net_worth = self.initial_balance
            
        return self._get_observation(), {}

    def _apply_execution_constraints(self, target_pos: float) -> float:
        """
        Apply 4-piece turnover reduction constraints
        Returns executed position after constraints
        """
        target_pos = float(np.clip(target_pos, -1.0, 1.0))
        pos = float(self.pos)

        # (1) Hysteresis: Small changes don't trigger trades
        if abs(target_pos - pos) < self.TAU:
            target_pos = pos

        # (2) Slew-rate limit: Max change per step
        delta = float(np.clip(target_pos - pos, -self.DELTA_MAX, self.DELTA_MAX))
        cand_pos = pos + delta

        # (3) Cooldown: Only restrict sign flips (allow reduction to 0)
        in_cd = (self.current_step - self.last_flip_t) < self.COOLDOWN_N
        wants_flip = (np.sign(pos) != 0 and 
                     np.sign(cand_pos) != 0 and 
                     np.sign(cand_pos) != np.sign(pos))

        if in_cd and wants_flip:
            # During cooldown: can't flip, go to 0 instead
            exec_pos = 0.0
        else:
            exec_pos = cand_pos
            if wants_flip:
                self.last_flip_t = self.current_step

        return float(np.clip(exec_pos, -1.0, 1.0))

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        
        unrealized_pnl_pct = 0.0 
        nw_change_pct = (self.net_worth - self.prev_net_worth) / self.prev_net_worth if self.prev_net_worth > 0 else 0
        
        # Cooldown remaining normalized to [0, 1]
        cooldown_remaining = max(0, self.COOLDOWN_N - (self.current_step - self.last_flip_t)) / self.COOLDOWN_N
        
        obs = np.array([
            self.pos,                    # Current position
            cooldown_remaining,          # Cooldown state (NEW)
            unrealized_pnl_pct, 
            nw_change_pct,
            row['Signal_Proba'],
            row['RSI'] / 100.0, 
            row['Rolling_Vol'],
            row['MACD'] / 100.0, 
            row['BB_Width'] / 1000.0,
            row['Dist_SMA_200'],
            row['ATR'] / row['Close'],
            row['Vol_Ratio'],
            float(np.sign(self.pos))     # Current direction (helps agent)
        ], dtype=np.float32)
        
        return obs

    def step(self, action):
        raw_action = float(action[0])
        
        row = self.df.iloc[self.current_step]
        current_price = row['Close']
        
        # === Volatility Targeting (kept for stability) ===
        current_atr_pct = (row['ATR'] / current_price) if row['ATR'] > 0 else 0.02
        if current_atr_pct < 0.005: current_atr_pct = 0.005 
        
        vol_scale = 0.05 / current_atr_pct
        vol_scale = np.clip(vol_scale, 0.1, 2.0)  # Reduced max from 3.0 to 2.0
        
        target_pos = raw_action * vol_scale
        
        # Risk Management Override
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        if self.risk_manager:
            override = self.risk_manager.check_risk(drawdown, target_pos)
            if override is not None:
                target_pos = override

        target_pos = np.clip(target_pos, -1, 1)
        
        # === Intra-day Kill Switch (Simulation) ===
        # Check if the day's High/Low would trigger the -5% daily loss
        # If so, force exit at that price (approx) and lockout for 1 day
        if self.enable_kill_switch:
            if self.kill_switch_lockout > 0:
                target_pos = 0.0
                self.kill_switch_lockout -= 1
            else:
                # Check for trigger during this step
                row = self.df.iloc[self.current_step]
                prev_close = self.df.iloc[self.current_step-1]['Close'] if self.current_step > 0 else row['Open']
                
                trigger = False
                penalty_price = current_price
            
                if self.pos > 0.1: # Long
                    if 'Low' in row:
                        drop_pct = (row['Low'] / prev_close) - 1
                        pnl_est = drop_pct * self.pos
                        if pnl_est < -0.05:
                            trigger = True
                            # Trigger at exactly -5% loss price
                            penalty_price = prev_close * (1 - 0.05/self.pos) 
                elif self.pos < -0.1: # Short
                    if 'High' in row:
                        rise_pct = (row['High'] / prev_close) - 1
                        pnl_est = rise_pct * abs(self.pos) * -1 # short loses on rise
                        if pnl_est < -0.05:
                            trigger = True
                            penalty_price = prev_close * (1 + 0.05/abs(self.pos))
                
                if trigger:
                    # Force exit at penalty price
                    # Override current_price for this step's valuation to reflect the stop-loss
                    current_price = penalty_price 
                    target_pos = 0.0 # Force close
                    self.kill_switch_lockout = 1 # Lockout for next step
                # Note: This step's reward will be calculated using `current_price` (the stop price)
                # Next step will see pos=0.

        # === Apply Execution Constraints (4-piece set) ===
        pos_prev = self.pos
        pos_exec = self._apply_execution_constraints(target_pos)
        self.pos = pos_exec
        
        # === Calculate Costs ===
        turnover = abs(pos_exec - pos_prev)
        
        # Trade cost: only on actual turnover
        trade_cost = 0.0
        if turnover > 0.001:
            trade_cost = turnover * self.net_worth * self.K_SINGLE
        
        # Funding cost: on position size
        funding_cost = abs(pos_exec) * self.net_worth * self.FUNDING_DAILY
        
        # === Update Portfolio ===
        self.prev_net_worth = self.net_worth
        
        current_equity = self.balance + (self.shares * current_price)
        self.max_net_worth = max(current_equity, self.max_net_worth)
        
        target_asset_value = current_equity * pos_exec
        current_asset_value = self.shares * current_price
        trade_value = target_asset_value - current_asset_value
        
        # Deduct costs from balance
        self.balance -= trade_cost
        self.balance -= funding_cost
        self.balance -= trade_value
        self.shares += trade_value / current_price if current_price > 0 else 0
        
        # Final Net Worth
        self.net_worth = self.balance + (self.shares * current_price)
        self.current_position = pos_exec
        
        # === Reward Calculation ===
        step_return = np.log(self.net_worth / self.prev_net_worth) if self.prev_net_worth > 0 else 0
        self.returns_history.append(step_return)
        
        if step_return < 0:
            self.negative_returns_history.append(step_return)
            
        reward = step_return
        
        # Sortino bonus for stable uptrend
        if len(self.returns_history) > 30:
            downside_std = np.std(self.negative_returns_history[-30:]) if len(self.negative_returns_history) > 0 else 0.01
            if downside_std < 1e-6: downside_std = 0.01
            
            rolling_mean = np.mean(self.returns_history[-30:])
            sortino = rolling_mean / downside_std
            
            if sortino > 0.5:
                reward += 0.05 * sortino
        
        # Drawdown penalty
        current_drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        if current_drawdown > 0.05:
            reward -= current_drawdown * 0.5 
            
        # Remove the "activity incentive" penalty - it was causing excessive trading
        # The constraints now handle position stability
        
        self.prev_action = pos_exec
            
        # Done Flag
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        info = {
            'net_worth': self.net_worth,
            'drawdown': current_drawdown,
            'step_return': step_return,
            'action': raw_action,
            'position': pos_exec,
            'turnover': turnover,
            'trade_cost': trade_cost,
            'funding_cost': funding_cost,
        }
        
        return self._get_observation(), reward, terminated, truncated, info
