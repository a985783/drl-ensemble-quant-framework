from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from envs.trading_env import TradingEnv


def _build_df() -> pd.DataFrame:
    n = 80
    close = np.linspace(100.0, 108.0, n)
    return pd.DataFrame(
        {
            "Close": close,
            "Open": close,
            "High": close * 1.002,
            "Low": close * 0.998,
            "Signal_Proba": np.clip(np.linspace(0.3, 0.7, n), 0, 1),
            "RSI": np.full(n, 50.0),
            "Rolling_Vol": np.full(n, 0.02),
            "MACD": np.full(n, 0.1),
            "BB_Width": np.full(n, 20.0),
            "Dist_SMA_200": np.zeros(n),
            "ATR": np.full(n, 1.0),
            "Vol_Ratio": np.ones(n),
        }
    )


def test_feature_mask_zeros_excluded_observation_slots() -> None:
    df = _build_df()
    keep_idx = [0, 4, 10]
    env = TradingEnv(df, feature_mask=keep_idx)
    obs, _ = env.reset()

    for i, value in enumerate(obs.tolist()):
        if i not in keep_idx:
            assert value == 0.0


def test_reward_profile_turnover_penalty_reduces_reward() -> None:
    df = _build_df()

    base_env = TradingEnv(df)
    penalized_env = TradingEnv(df, reward_profile={"turnover": 1.0})

    base_env.reset()
    penalized_env.reset()

    _, base_reward, _, _, _ = base_env.step(np.array([1.0], dtype=np.float32))
    _, penalized_reward, _, _, _ = penalized_env.step(np.array([1.0], dtype=np.float32))

    assert penalized_reward < base_reward


def test_safe_log_return_is_finite_for_non_positive_net_worth() -> None:
    # Extreme training paths can drive temporary non-positive equity.
    # Reward computation should stay finite and not poison PPO with NaNs.
    val = TradingEnv._safe_log_return(curr_net_worth=-10.0, prev_net_worth=10000.0)
    assert np.isfinite(val)
    assert val <= 0.0


def test_eth_uses_legacy_obs_scaling() -> None:
    df = _build_df()

    env_eth = TradingEnv(df, symbol="ETH/USDT:USDT")

    obs_eth, _ = env_eth.reset()

    # Slot 7: MACD feature
    assert np.isclose(obs_eth[7], 0.001, atol=1e-6)  # 0.1 / 100.0

    # Slot 8: BB width feature
    assert np.isclose(obs_eth[8], 0.02, atol=1e-6)   # 20 / 1000.0
