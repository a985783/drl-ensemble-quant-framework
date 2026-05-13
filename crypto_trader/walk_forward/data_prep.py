"""
DataPreparer – per-fold XGBoost training + next-bar shift for walk-forward validation.

For each walk-forward fold:
1. Fetch train and test OHLCV data independently via DataLoader.
2. Apply feature engineering (FeatureEngineer) to both datasets.
3. Train XGBoost (SignalPredictor) on training data ONLY – no leakage.
4. Predict Signal_Proba for both train and test using the trained model.
5. Apply next-bar shift (resolve_execution_frame) to both datasets so that
   row t carries data from bar t+1, matching the backtest execution paradigm.
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Optional

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from crypto_trader.data_loader import DataLoader
    from crypto_trader.features import FeatureEngineer
    from crypto_trader.models.signal_model import SignalPredictor
    from crypto_trader.backtest_moe import resolve_execution_frame
except ImportError:
    PARENT_DIR = os.path.dirname(SCRIPT_DIR)
    sys.path.insert(0, PARENT_DIR)
    from data_loader import DataLoader
    from features import FeatureEngineer
    from models.signal_model import SignalPredictor
    from backtest_moe import resolve_execution_frame


class DataPreparer:
    """Per-fold data preparation: XGBoost-only train → Signal_Proba → next-bar shift.

    Instantiate once per walk-forward fold.  Call ``.prepare()`` to get
    train/test DataFrames that are ready for RL training and backtesting.

    Parameters
    ----------
    config
        A config object with at minimum:
        - ``symbol`` or ``data.symbol``  (e.g. "ETH/USDT:USDT")
        - ``interval`` or ``data.interval``  (e.g. "1d")
        - ``execution_mode`` (optional, default "next_bar")
    fold_config
        A fold config with ``train_start``, ``train_end``, ``test_start``, ``test_end``
        string attributes (``FoldSpec`` or ``FoldConfig`` compatible).
    """

    def __init__(self, config, fold_config):
        self.config = config
        self.fold = fold_config

    @property
    def symbol(self) -> str:
        """Resolve symbol from either ``config.symbol`` or ``config.data.symbol``."""
        for attr in ("symbol", "data.symbol"):
            val = self._resolve_attr(attr)
            if val is not None:
                return val
        return "ETH/USDT:USDT"

    @property
    def interval(self) -> str:
        """Resolve interval from either ``config.interval`` or ``config.data.interval``."""
        for attr in ("interval", "data.interval"):
            val = self._resolve_attr(attr)
            if val is not None:
                return val
        return "1d"

    @property
    def execution_mode(self) -> str:
        """Resolve execution_mode, defaulting to "next_bar"."""
        val = self._resolve_attr("execution_mode")
        return val if val else "next_bar"

    def _resolve_attr(self, dotted: str) -> Optional[str]:
        """Walk a dotted path through the config object hierarchy."""
        parts = dotted.split(".")
        obj = self.config
        for p in parts:
            obj = getattr(obj, p, None)
            if obj is None:
                return None
        return obj

    def prepare(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch, engineer, and return (train, test) DataFrames.

        Technical indicators are calculated per-fold to avoid leakage, filtered
        exclusively on the fold's training window, and are shifted for next-bar execution.
        """
        loader = DataLoader()
        engineer = FeatureEngineer()

        # Add padding to ensure indicators (like SMA 200) have enough historical context
        padding_days = 250

        def _get_padded_start(start_str: str) -> str:
            from datetime import datetime, timedelta
            dt = datetime.strptime(start_str, "%Y-%m-%d")
            # OKX ETH/USDT Swap data starts around 2020-01-01
            floor_dt = datetime(2020, 1, 1)
            padded = max(floor_dt, dt - timedelta(days=padding_days))
            return padded.strftime("%Y-%m-%d")
        train_raw = loader.fetch_data(
            _get_padded_start(self.fold.train_start),
            self.fold.train_end,
            self.symbol,
            interval=self.interval,
        )
        test_raw = loader.fetch_data(
            _get_padded_start(self.fold.test_start),
            self.fold.test_end,
            self.symbol,
            interval=self.interval,
        )

        if isinstance(train_raw.columns, pd.MultiIndex):
            train_raw.columns = train_raw.columns.get_level_values(0)
        if isinstance(test_raw.columns, pd.MultiIndex):
            test_raw.columns = test_raw.columns.get_level_values(0)

        train_df_full = engineer.add_technical_indicators(train_raw, symbol=self.symbol)
        test_df_full = engineer.add_technical_indicators(test_raw, symbol=self.symbol)

        # Filter back to original requested windows
        train_df = train_df_full[train_df_full.index >= self.fold.train_start].copy()
        test_df = test_df_full[test_df_full.index >= self.fold.test_start].copy()


        if len(train_df) < 200:
            raise ValueError(
                f"Training data too short ({len(train_df)} rows) for fold "
                f"{getattr(self.fold, 'fold_id', '?')} "
                f"({self.fold.train_start} → {self.fold.train_end})"
            )
        if len(test_df) < 2:
            raise ValueError(
                f"Test data too short ({len(test_df)} rows) for fold "
                f"{getattr(self.fold, 'fold_id', '?')} "
                f"({self.fold.test_start} → {self.fold.test_end})"
            )

        predictor = SignalPredictor()
        predictor.train(train_df)

        train_df["Signal_Proba"] = predictor.predict_proba(train_df)
        test_df["Signal_Proba"] = predictor.predict_proba(test_df)

        train_shifted, _ = resolve_execution_frame(train_df, self.execution_mode)
        test_shifted, _ = resolve_execution_frame(test_df, self.execution_mode)

        self.save_fold_data(train_shifted, test_shifted)
        return train_shifted, test_shifted

    def save_fold_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Path]:
        """Persist per-fold prepared datasets for audit and resume diagnostics."""
        root = Path(getattr(self.config, "results_root", "crypto_trader/walk_forward/results/walk_forward_moe"))
        fold_id = getattr(self.fold, "fold_id", "unknown_fold")
        out_dir = root / fold_id / "data"
        out_dir.mkdir(parents=True, exist_ok=True)

        train_path = out_dir / "train_prepared.csv"
        test_path = out_dir / "test_prepared.csv"
        metadata_path = out_dir / "metadata.json"

        train_df.to_csv(train_path)
        test_df.to_csv(test_path)
        metadata = {
            "fold_id": fold_id,
            "train_start": getattr(self.fold, "train_start", ""),
            "train_end": getattr(self.fold, "train_end", ""),
            "test_start": getattr(self.fold, "test_start", ""),
            "test_end": getattr(self.fold, "test_end", ""),
            "execution_mode": self.execution_mode,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "columns": list(map(str, train_df.columns)),
            "signal_proba_present": "Signal_Proba" in train_df.columns and "Signal_Proba" in test_df.columns,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"train": train_path, "test": test_path, "metadata": metadata_path}
