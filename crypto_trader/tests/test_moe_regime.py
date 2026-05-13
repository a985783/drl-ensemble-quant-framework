from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from moe.regime import select_market_slice


def _build_df() -> pd.DataFrame:
    n = 240
    idx = np.arange(n)

    close = np.concatenate([
        np.linspace(100, 130, 60),
        np.linspace(130, 90, 60),
        110 + 3 * np.sin(np.linspace(0, 16, 60)),
        110 + 0.5 * np.sin(np.linspace(0, 8, 60)),
    ])
    atr = np.concatenate([
        np.full(60, 1.0),
        np.full(60, 1.2),
        np.full(60, 5.0),
        np.full(60, 0.8),
    ])

    df = pd.DataFrame(
        {
            "Close": close,
            "ATR": atr,
            "Signal_Proba": np.clip(np.linspace(0.2, 0.8, n), 0, 1),
            "RSI": np.full(n, 50.0),
            "Rolling_Vol": np.full(n, 0.02),
            "MACD": np.zeros(n),
            "BB_Width": np.full(n, 20.0),
            "Dist_SMA_200": np.zeros(n),
            "Vol_Ratio": np.ones(n),
        },
        index=idx,
    )
    return df


def test_regime_slices_are_non_empty() -> None:
    df = _build_df()

    for name in ["bull", "bear", "range", "high_vol", "low_vol", "full"]:
        sliced = select_market_slice(df, name)
        assert len(sliced) > 0
        assert set(sliced.columns) == set(df.columns)
