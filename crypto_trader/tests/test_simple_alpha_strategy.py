from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from strategies.simple_alpha import (
    SimpleAlphaParams,
    backtest_simple_alpha,
    choose_params,
    compute_target_position,
)


def _row(**overrides):
    base = {
        "Signal_Proba": 0.55,
        "Dist_SMA_200": 0.0,
        "MACD": 0.0,
        "ATR": 5.0,
        "Close": 100.0,
        "Rolling_Vol": 0.02,
    }
    base.update(overrides)
    return pd.Series(base)


def test_compute_target_position_uses_probability_and_trend() -> None:
    params = SimpleAlphaParams(signal_weight=0.7, trend_weight=0.3, threshold=0.08)

    long_pos = compute_target_position(
        _row(Signal_Proba=0.68, Dist_SMA_200=0.15, MACD=10.0),
        params,
        vol_cap=0.04,
    )
    short_pos = compute_target_position(
        _row(Signal_Proba=0.32, Dist_SMA_200=-0.15, MACD=-10.0),
        params,
        vol_cap=0.04,
    )

    assert long_pos > 0
    assert short_pos < 0


def test_high_volatility_reduces_position_size() -> None:
    params = SimpleAlphaParams(signal_weight=1.0, trend_weight=0.0, threshold=0.01)
    normal = compute_target_position(_row(Signal_Proba=0.75, Rolling_Vol=0.02), params, vol_cap=0.04)
    high_vol = compute_target_position(_row(Signal_Proba=0.75, Rolling_Vol=0.08), params, vol_cap=0.04)

    assert abs(high_vol) < abs(normal)


def test_backtest_uses_next_bar_return_for_signal() -> None:
    df = pd.DataFrame(
        {
            "Close": [100.0, 110.0, 99.0],
            "Signal_Proba": [0.8, 0.2, 0.5],
            "Dist_SMA_200": [0.1, -0.1, 0.0],
            "MACD": [1.0, -1.0, 0.0],
            "ATR": [5.0, 5.0, 5.0],
            "Rolling_Vol": [0.02, 0.02, 0.02],
        }
    )
    params = SimpleAlphaParams(signal_weight=1.0, trend_weight=0.0, threshold=0.01, max_position=1.0)

    result = backtest_simple_alpha(df, params)

    assert result["positions"][0] > 0
    assert result["net_worth"][1] > result["net_worth"][0]


def test_choose_params_uses_candidate_grid() -> None:
    df = pd.DataFrame(
        {
            "Close": [100, 102, 104, 103, 105, 107, 109, 108, 110, 112],
            "Signal_Proba": [0.7] * 10,
            "Dist_SMA_200": [0.1] * 10,
            "MACD": [1.0] * 10,
            "ATR": [5.0] * 10,
            "Rolling_Vol": [0.02] * 10,
        }
    )

    params, result = choose_params(df)

    assert isinstance(params, SimpleAlphaParams)
    assert result["total_return"] > 0
