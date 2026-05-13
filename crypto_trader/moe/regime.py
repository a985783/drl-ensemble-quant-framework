from __future__ import annotations

import numpy as np
import pandas as pd


def _trend_signal(close: pd.Series, window: int = 20) -> pd.Series:
    shifted = close.shift(window)
    trend = (close - shifted) / shifted.replace(0, np.nan)
    return trend.fillna(0.0)


def _atr_pct(df: pd.DataFrame) -> pd.Series:
    if "ATR" in df.columns and "Close" in df.columns:
        close = df["Close"].replace(0, np.nan)
        return (df["ATR"] / close).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)


def select_market_slice(df: pd.DataFrame, slice_name: str) -> pd.DataFrame:
    """Return a market-regime subset for expert specialization.

    Supported slices: full, bull, bear, range, high_vol, low_vol.
    Falls back to full dataset when a slice is empty.
    """
    name = (slice_name or "full").strip().lower()
    if name == "full":
        return df.copy()

    if "Close" not in df.columns:
        raise ValueError("select_market_slice requires 'Close' column")

    trend = _trend_signal(df["Close"])
    atr_pct = _atr_pct(df)

    bull_th = float(trend.quantile(0.65))
    bear_th = float(trend.quantile(0.35))
    range_th = float(trend.abs().quantile(0.40))
    high_vol_th = float(atr_pct.quantile(0.70))
    low_vol_th = float(atr_pct.quantile(0.30))

    if name == "bull":
        mask = trend >= bull_th
    elif name == "bear":
        mask = trend <= bear_th
    elif name == "range":
        mask = trend.abs() <= range_th
    elif name == "high_vol":
        mask = atr_pct >= high_vol_th
    elif name == "low_vol":
        mask = atr_pct <= low_vol_th
    else:
        raise ValueError(f"Unsupported market slice: {slice_name}")

    sliced = df.loc[mask].copy()
    if len(sliced) == 0:
        return df.copy()
    return sliced
