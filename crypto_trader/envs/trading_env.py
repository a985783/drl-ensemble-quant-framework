from __future__ import annotations

"""
Trading Environment with Four-Piece Turnover Reduction Constraints
Phase B: Constraints integrated into env for PPO to learn within constraints

Constraints:
1. Hysteresis (τ=0.25): Small changes don't trigger trades
2. Slew-rate limit (δ=0.15): Max position change per step
3. Cooldown (N=3): Min interval between sign flips
4. Real cost model: Fee + Slippage + Funding Rate
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

try:
    from crypto_trader.asset_profile import get_asset_profile
except ImportError:
    from asset_profile import get_asset_profile


def apply_execution_constraints_core(
    target_pos: float,
    current_pos: float,
    last_flip_marker: float,
    current_marker: float,
    tau: float,
    delta_max: float,
    cooldown_window: float,
):
    """
    Shared execution-constraint core for both backtest and live paths.
    Markers can be step numbers or timestamps, as long as units are consistent
    with cooldown_window.
    """
    reason = "Normal"

    target_pos = float(np.clip(target_pos, -1.0, 1.0))
    current_pos = float(np.clip(current_pos, -1.0, 1.0))
    last_flip_marker = float(last_flip_marker)
    current_marker = float(current_marker)
    tau = float(max(tau, 0.0))
    delta_max = float(max(delta_max, 0.0))
    cooldown_window = float(max(cooldown_window, 0.0))

    if abs(target_pos - current_pos) < tau:
        target_pos = current_pos
        reason = "Hysteresis"

    delta = float(np.clip(target_pos - current_pos, -delta_max, delta_max))
    if abs(target_pos - current_pos) > delta_max:
        reason = "SlewRate"

    cand_pos = current_pos + delta

    in_cd = (current_marker - last_flip_marker) < cooldown_window
    wants_flip = (
        np.sign(current_pos) != 0
        and np.sign(cand_pos) != 0
        and np.sign(cand_pos) != np.sign(current_pos)
    )

    exec_pos = cand_pos
    new_flip_marker = last_flip_marker

    if in_cd and wants_flip:
        exec_pos = 0.0
        reason = "Cooldown"
    elif wants_flip:
        new_flip_marker = current_marker
        reason = "Flip"

    exec_pos = float(np.clip(exec_pos, -1.0, 1.0))
    return exec_pos, float(new_flip_marker), reason

class TradingEnv(gym.Env):
    """
    Trading environment with execution constraints.
    Agent outputs target position intent a_raw ∈ [-1, 1]
    Env applies constraints to get pos_exec, settles PnL/costs on pos_exec
    """
    metadata = {'render_modes': ['human']}

    # === Constraint Defaults ===
    TAU = 0.25
    DELTA_MAX = 0.15
    COOLDOWN_N = 3
    TARGET_ATR_PCT = 0.05

    # === Cost Defaults (OKX Perpetual) ===
    K_SINGLE = 0.0008
    FUNDING_DAILY = 0.0003
    MIN_NET_WORTH = 1e-8

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        fee_rate: float = 0.001,
        risk_manager=None,
        enable_kill_switch: bool = False,
        atr_floor: float = 0.005,
        vol_scale_min: float = 0.1,
        vol_scale_max: float = 2.0,
        feature_mask=None,
        reward_profile=None,
        symbol: str | None = None,
        target_atr_pct: float | None = None,
        tau: float | None = None,
        delta_max: float | None = None,
        cooldown_n: int | None = None,
        k_single: float | None = None,
        funding_daily: float | None = None,
        regime_mask: np.ndarray | None = None,
        regime_main_reward_weight: float = 1.0,
        regime_off_reward_weight: float = 1.0,
        sat_penalty_coef: float = 0.0,
        sat_threshold: float = 0.9,
        directional_bias_coef: float = 0.0,
        directional_bias_alpha: float = 0.01,
        action_cap: float = 1.0,
        funding_cost_multiplier: float = 1.0,
        short_squeeze_threshold: float | None = None,
        short_squeeze_max_short: float = 0.25,
    ):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.risk_manager = risk_manager
        self.enable_kill_switch = enable_kill_switch
        profile = get_asset_profile(symbol)
        env_cfg = profile.env
        # Keep ETH path on legacy observation scaling so locked ETH models
        # remain behaviorally consistent with historical training/inference.
        self.use_legacy_observation = (profile.key == "ETH")
        self.atr_floor = float(atr_floor if atr_floor is not None else env_cfg.atr_floor)
        self.vol_scale_min = float(vol_scale_min if vol_scale_min is not None else env_cfg.vol_scale_min)
        self.vol_scale_max = float(vol_scale_max if vol_scale_max is not None else env_cfg.vol_scale_max)
        self.target_atr_pct = float(target_atr_pct if target_atr_pct is not None else env_cfg.target_atr_pct)
        self.tau = float(tau if tau is not None else env_cfg.tau)
        self.delta_max = float(delta_max if delta_max is not None else env_cfg.delta_max)
        self.cooldown_n = int(cooldown_n if cooldown_n is not None else env_cfg.cooldown_n)
        self.k_single = float(k_single if k_single is not None else env_cfg.k_single)
        self.funding_daily = float(funding_daily if funding_daily is not None else env_cfg.funding_daily)
        self.regime_mask = None
        if regime_mask is not None:
            mask_arr = np.asarray(regime_mask).astype(np.float32).reshape(-1)
            if len(mask_arr) != len(self.df):
                raise ValueError(f"regime_mask length mismatch: {len(mask_arr)} != {len(self.df)}")
            self.regime_mask = mask_arr
        self.regime_main_reward_weight = float(max(regime_main_reward_weight, 0.0))
        self.regime_off_reward_weight = float(max(regime_off_reward_weight, 0.0))
        self.sat_penalty_coef = float(max(sat_penalty_coef, 0.0))
        self.sat_threshold = float(np.clip(sat_threshold, 0.0, 1.0))
        self.directional_bias_coef = float(max(directional_bias_coef, 0.0))
        self.directional_bias_alpha = float(np.clip(directional_bias_alpha, 0.0, 1.0))
        self.action_cap = float(np.clip(action_cap, 0.0, 1.0))
        self.funding_cost_multiplier = float(max(funding_cost_multiplier, 0.0))
        self.short_squeeze_threshold = None if short_squeeze_threshold is None else float(max(short_squeeze_threshold, 0.0))
        self.short_squeeze_max_short = float(np.clip(short_squeeze_max_short, 0.0, 1.0))
        self.feature_mask = None
        if feature_mask is not None:
            mask = sorted({int(i) for i in feature_mask})
            for i in mask:
                if i < 0 or i >= 13:
                    raise ValueError(f"feature_mask index out of range: {i}")
            self.feature_mask = mask
        default_reward_profile = {
            "return": 1.0,
            "sortino": 1.0,
            "drawdown": 1.0,
            "turnover": 0.0,
        }
        reward_profile = reward_profile or {}
        self.reward_profile = {
            "return": float(reward_profile.get("return", default_reward_profile["return"])),
            "sortino": float(reward_profile.get("sortino", default_reward_profile["sortino"])),
            "drawdown": float(reward_profile.get("drawdown", default_reward_profile["drawdown"])),
            "turnover": float(reward_profile.get("turnover", default_reward_profile["turnover"])),
        }
        
        # Action Space: Continuous [-1, 1] (target position intent)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation Space:
        # [Pos, Cooldown_Remaining, UnPnL, NW_Chg, Prob, RSI, Vol, MACD_*, BB_Width_*, Dist_SMA, ATR_Pct, Vol_Ratio]
        # Keep legacy MACD/BB scaling to preserve ETH model behavior.
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
        self.position_bias_ema = 0.0

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
        self.position_bias_ema = 0.0
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
        exec_pos, new_flip_t, _ = apply_execution_constraints_core(
            target_pos=target_pos,
            current_pos=self.pos,
            last_flip_marker=self.last_flip_t,
            current_marker=self.current_step,
            tau=self.tau,
            delta_max=self.delta_max,
            cooldown_window=self.cooldown_n,
        )
        self.last_flip_t = int(new_flip_t)
        return exec_pos

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        
        unrealized_pnl_pct = 0.0 
        nw_change_pct = (self.net_worth - self.prev_net_worth) / self.prev_net_worth if self.prev_net_worth > 0 else 0
        
        # Cooldown remaining normalized to [0, 1]
        cooldown_remaining = max(0, self.cooldown_n - (self.current_step - self.last_flip_t)) / max(self.cooldown_n, 1)
        close = float(max(row['Close'], 1e-8))
        if self.use_legacy_observation:
            macd_feature = float(row['MACD'] / 100.0)
            bb_width_feature = float(row['BB_Width'] / 1000.0)
        else:
            macd_feature = float(row['MACD'] / close)
            bb_width_feature = float(row['BB_Width'] / close)
        
        obs = np.array([
            self.pos,                    # Current position
            cooldown_remaining,          # Cooldown state (NEW)
            unrealized_pnl_pct, 
            nw_change_pct,
            row['Signal_Proba'],
            row['RSI'] / 100.0, 
            row['Rolling_Vol'],
            macd_feature,
            bb_width_feature,
            row['Dist_SMA_200'],
            row['ATR'] / close,
            row['Vol_Ratio'],
            float(np.sign(self.pos))     # Current direction (helps agent)
        ], dtype=np.float32)

        if self.feature_mask is not None:
            masked = np.zeros_like(obs)
            masked[self.feature_mask] = obs[self.feature_mask]
            obs = masked
        
        return obs

    @staticmethod
    def _safe_log_return(curr_net_worth: float, prev_net_worth: float) -> float:
        safe_curr = max(float(curr_net_worth), TradingEnv.MIN_NET_WORTH)
        safe_prev = max(float(prev_net_worth), TradingEnv.MIN_NET_WORTH)
        value = float(np.log(safe_curr / safe_prev))
        if not np.isfinite(value):
            return -1.0
        return value

    def step(self, action):
        raw_action = float(np.clip(action[0], -self.action_cap, self.action_cap))
        
        row = self.df.iloc[self.current_step]
        current_price = row['Close']
        
        # === Volatility Targeting (kept for stability) ===
        current_atr_pct = (row['ATR'] / current_price) if row['ATR'] > 0 else self.target_atr_pct
        if current_atr_pct < self.atr_floor:
            current_atr_pct = self.atr_floor
        
        vol_scale = self.target_atr_pct / current_atr_pct
        vol_scale = np.clip(vol_scale, self.vol_scale_min, self.vol_scale_max)
        
        target_pos = raw_action * vol_scale
        
        # Risk Management Override
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        if self.risk_manager:
            override = self.risk_manager.check_risk(drawdown, target_pos)
            if override is not None:
                target_pos = override

        # Short squeeze brake (training safety regularizer):
        # in strong intraday up-move, cap short intent to avoid tail-loss behavior.
        if self.short_squeeze_threshold is not None and target_pos < 0:
            prev_close = self.df.iloc[self.current_step-1]['Close'] if self.current_step > 0 else row['Open']
            prev_close = float(max(prev_close, 1e-8))
            intraday_up = float(row.get('High', current_price) / prev_close - 1.0)
            if intraday_up >= self.short_squeeze_threshold:
                target_pos = max(target_pos, -self.short_squeeze_max_short)

        target_pos = np.clip(target_pos, -1, 1)
        
        # === Intra-day Kill Switch (Simulation, optional) ===
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
            cost_base = max(float(self.net_worth), 0.0)
            trade_cost = turnover * cost_base * self.k_single
        
        # Funding cost: on position size
        # 使用动态 Funding Rate（若数据中包含 Funding_Rate 列），否则用固定值
        if 'Funding_Rate' in self.df.columns:
            daily_rate = float(row.get('Funding_Rate', self.funding_daily))
        else:
            daily_rate = self.funding_daily
        funding_base = max(float(self.net_worth), 0.0)
        funding_cost = abs(pos_exec) * funding_base * daily_rate
        funding_cost *= self.funding_cost_multiplier
        
        # === Update Portfolio ===
        self.prev_net_worth = self.net_worth
        
        current_equity = self.balance + (self.shares * current_price)
        if not np.isfinite(current_equity):
            current_equity = 0.0
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
        bankrupt = self.net_worth <= self.MIN_NET_WORTH or (not np.isfinite(self.net_worth))
        if bankrupt:
            self.net_worth = self.MIN_NET_WORTH
            self.balance = max(self.balance, 0.0)
            self.shares = 0.0
        self.current_position = pos_exec
        
        # === Reward Calculation ===
        step_return = self._safe_log_return(self.net_worth, self.prev_net_worth)
        self.returns_history.append(step_return)
        
        if step_return < 0:
            self.negative_returns_history.append(step_return)
            
        reward = self.reward_profile["return"] * step_return
        
        # Sortino bonus for stable uptrend
        sortino_bonus = 0.0
        if len(self.returns_history) > 30:
            downside_std = np.std(self.negative_returns_history[-30:]) if len(self.negative_returns_history) > 0 else 0.01
            if downside_std < 1e-6: downside_std = 0.01
            
            rolling_mean = np.mean(self.returns_history[-30:])
            sortino = rolling_mean / downside_std
            
            if sortino > 0.5:
                sortino_bonus = 0.05 * sortino
        reward += self.reward_profile["sortino"] * sortino_bonus
        
        # Drawdown penalty
        safe_peak = max(float(self.max_net_worth), self.MIN_NET_WORTH)
        current_drawdown = (safe_peak - self.net_worth) / safe_peak
        drawdown_penalty = 0.0
        if current_drawdown > 0.05:
            drawdown_penalty = current_drawdown * 0.5
        reward -= self.reward_profile["drawdown"] * drawdown_penalty
        reward -= self.reward_profile["turnover"] * turnover

        # Regime-weighted reward shaping (training mode):
        # reward contribution is emphasized in expert target regime.
        in_regime = 1.0
        if self.regime_mask is not None:
            in_regime = float(self.regime_mask[self.current_step] > 0.5)
            reward *= self.regime_main_reward_weight if in_regime > 0.5 else self.regime_off_reward_weight

        # Saturation penalty: discourage persistent full-size actions.
        if self.sat_penalty_coef > 0:
            sat_excess = max(0.0, abs(raw_action) - self.sat_threshold)
            reward -= self.sat_penalty_coef * (sat_excess ** 2)

        # Directional-bias penalty: discourage long-lived one-sided positioning.
        if self.directional_bias_coef > 0 and self.directional_bias_alpha > 0:
            self.position_bias_ema = (
                (1.0 - self.directional_bias_alpha) * self.position_bias_ema
                + self.directional_bias_alpha * float(pos_exec)
            )
            reward -= self.directional_bias_coef * abs(self.position_bias_ema)

        if not np.isfinite(reward):
            reward = -1.0
            
        # Remove the "activity incentive" penalty - it was causing excessive trading
        # The constraints now handle position stability
        
        self.prev_action = pos_exec
            
        # Done Flag
        self.current_step += 1
        terminated = (self.current_step >= len(self.df) - 1) or bankrupt
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
            'funding_rate': daily_rate,
            'in_regime': in_regime,
            'position_bias_ema': float(self.position_bias_ema),
        }
        
        return self._get_observation(), reward, terminated, truncated, info
