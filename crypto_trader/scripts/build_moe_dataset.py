from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from crypto_trader.data_loader import DataLoader
    from crypto_trader.features import FeatureEngineer
    from crypto_trader.models.signal_model import SignalPredictor
except ImportError:
    from data_loader import DataLoader
    from features import FeatureEngineer
    from models.signal_model import SignalPredictor


def build_dataset(
    symbol: str,
    start: str,
    end: str,
    train_ratio: float,
    output_prefix: str,
    interval: str = "1d",
) -> None:
    loader = DataLoader()
    engineer = FeatureEngineer()

    raw_df = loader.fetch_data(start, end, symbol, interval=interval)
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)

    processed_df = engineer.add_technical_indicators(raw_df, symbol=symbol)
    split_idx = int(len(processed_df) * train_ratio)
    train_df = processed_df.iloc[:split_idx].copy()

    predictor = SignalPredictor()
    predictor.train(train_df)
    processed_df["Signal_Proba"] = predictor.predict_proba(processed_df)

    split_idx2 = int(len(processed_df) * train_ratio)
    train80 = processed_df.iloc[:split_idx2].copy()
    oos20 = processed_df.iloc[split_idx2:].copy()

    out_prefix = Path(output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    full_path = out_prefix.with_name(out_prefix.name + "_full.csv")
    train_path = out_prefix.with_name(out_prefix.name + "_train80.csv")
    oos_path = out_prefix.with_name(out_prefix.name + "_oos20.csv")

    processed_df.to_csv(full_path)
    train80.to_csv(train_path)
    oos20.to_csv(oos_path)

    print(f"symbol={symbol}")
    print(f"full_rows={len(processed_df)} full_start={processed_df.index.min().date()} full_end={processed_df.index.max().date()}")
    print(f"train_rows={len(train80)} train_start={train80.index.min().date()} train_end={train80.index.max().date()}")
    print(f"oos_rows={len(oos20)} oos_start={oos20.index.min().date()} oos_end={oos20.index.max().date()}")
    print(f"saved_full={full_path}")
    print(f"saved_train80={train_path}")
    print(f"saved_oos20={oos_path}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build MoE dataset with train80/oos20 split")
    p.add_argument("--symbol", type=str, required=True, help="CCXT symbol, e.g. ETH/USDT:USDT")
    p.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    p.add_argument("--interval", type=str, default="1d", help="Kline interval, e.g. 1d, 4h")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--output-prefix", type=str, required=True, help="Output prefix without suffix")
    return p


def main() -> None:
    args = _parser().parse_args()
    build_dataset(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        interval=args.interval,
        train_ratio=args.train_ratio,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()
