from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from strategies.simple_alpha_walk_forward import (
    FoldSpec,
    build_fold_dataframe,
    summarize_fold_rows,
)


def _raw_df() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=8, freq="D")
    return pd.DataFrame(
        {
            "Close": [100, 101, 102, 103, 104, 105, 106, 107],
            "Signal_Proba": [0.99] * 8,
            "RSI": [50] * 8,
            "MACD": [1] * 8,
            "Dist_SMA_200": [0.1] * 8,
            "ATR": [5] * 8,
            "Rolling_Vol": [0.02] * 8,
        },
        index=idx,
    )


def test_build_fold_dataframe_drops_preexisting_signal_column() -> None:
    fold = FoldSpec("toy", "2020-01-01", "2020-01-04", "2020-01-05", "2020-01-08")
    train_df, test_df = build_fold_dataframe(_raw_df(), fold)

    assert "Signal_Proba" not in train_df.columns
    assert "Signal_Proba" not in test_df.columns
    assert train_df.index.max() <= pd.Timestamp("2020-01-04")
    assert test_df.index.min() >= pd.Timestamp("2020-01-05")


def test_summarize_fold_rows_compounds_returns() -> None:
    rows = [
        {"fold": "a", "total_return": 0.10, "benchmark_return": 0.0, "alpha": 0.10, "max_drawdown": 0.05},
        {"fold": "b", "total_return": -0.05, "benchmark_return": -0.1, "alpha": 0.05, "max_drawdown": 0.08},
    ]

    summary = summarize_fold_rows(rows)

    assert round(summary["cumulative_return"], 4) == 0.045
    assert round(summary["avg_alpha"], 6) == 0.075
    assert summary["max_drawdown"] == 0.08
