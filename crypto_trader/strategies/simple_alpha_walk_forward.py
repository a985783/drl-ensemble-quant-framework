from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from crypto_trader.models.signal_model import SignalPredictor
    from crypto_trader.strategies.simple_alpha import backtest_simple_alpha, choose_params, load_dataset
except ImportError:  # pragma: no cover
    from models.signal_model import SignalPredictor
    from strategies.simple_alpha import backtest_simple_alpha, choose_params, load_dataset


@dataclass(frozen=True)
class FoldSpec:
    name: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str


DEFAULT_FOLDS = [
    FoldSpec("fold1_test2022", "2020-07-19", "2021-12-31", "2022-01-01", "2022-12-31"),
    FoldSpec("fold2_test2023", "2020-07-19", "2022-12-31", "2023-01-01", "2023-12-31"),
    FoldSpec("fold3_test2024", "2020-07-19", "2023-12-31", "2024-01-01", "2024-12-31"),
    FoldSpec("fold4_test2025", "2020-07-19", "2024-12-31", "2025-01-01", "2025-12-31"),
]


def build_fold_dataframe(df: pd.DataFrame, fold: FoldSpec) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clean = df.copy()
    if "Signal_Proba" in clean.columns:
        clean = clean.drop(columns=["Signal_Proba"])
    train = clean.loc[pd.Timestamp(fold.train_start) : pd.Timestamp(fold.train_end)].copy()
    test = clean.loc[pd.Timestamp(fold.test_start) : pd.Timestamp(fold.test_end)].copy()
    if train.empty or test.empty:
        raise ValueError(f"Fold {fold.name} has empty train/test split")
    return train, test


def add_fold_signal(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    predictor = SignalPredictor()
    acc, _ = predictor.train(train_df)

    train_scored = train_df.copy()
    test_scored = test_df.copy()
    train_scored["Signal_Proba"] = predictor.predict_proba(train_scored)
    test_scored["Signal_Proba"] = predictor.predict_proba(test_scored)

    y_test = (test_scored["Close"].shift(-1) > test_scored["Close"]).astype(int).iloc[:-1]
    p_test = test_scored["Signal_Proba"].iloc[:-1]
    signal_acc = float(((p_test > 0.5).astype(int) == y_test).mean()) if len(y_test) else 0.0
    return train_scored, test_scored, {"train_internal_accuracy": float(acc), "test_signal_accuracy": signal_acc}


def run_fold(df: pd.DataFrame, fold: FoldSpec) -> Dict[str, object]:
    train_raw, test_raw = build_fold_dataframe(df, fold)
    train_df, test_df, signal_metrics = add_fold_signal(train_raw, test_raw)
    params, train_result = choose_params(train_df)
    test_result = backtest_simple_alpha(test_df, params)

    return {
        "fold": fold.name,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "total_return": float(test_result["total_return"]),
        "benchmark_return": float(test_result["benchmark_return"]),
        "alpha": float(test_result["alpha"]),
        "max_drawdown": float(test_result["max_drawdown"]),
        "sharpe": float(test_result["sharpe"]),
        "turnover": float(test_result["turnover"]),
        "trade_cost": float(test_result["trade_cost"]),
        "funding_cost": float(test_result["funding_cost"]),
        "train_selected_return": float(train_result["total_return"]),
        "train_selected_max_drawdown": float(train_result["max_drawdown"]),
        "train_internal_accuracy": float(signal_metrics["train_internal_accuracy"]),
        "test_signal_accuracy": float(signal_metrics["test_signal_accuracy"]),
        "params": asdict(params),
    }


def summarize_fold_rows(rows: Iterable[Dict[str, object]]) -> Dict[str, float]:
    rows = list(rows)
    cumulative = 1.0
    for row in rows:
        cumulative *= 1.0 + float(row["total_return"])
    return {
        "folds": float(len(rows)),
        "cumulative_return": float(cumulative - 1.0),
        "avg_return": float(np.mean([float(r["total_return"]) for r in rows])) if rows else 0.0,
        "avg_alpha": float(np.mean([float(r["alpha"]) for r in rows])) if rows else 0.0,
        "max_drawdown": float(max([float(r["max_drawdown"]) for r in rows], default=0.0)),
        "positive_return_fraction": float(np.mean([float(r["total_return"]) > 0 for r in rows])) if rows else 0.0,
        "positive_alpha_fraction": float(np.mean([float(r["alpha"]) > 0 for r in rows])) if rows else 0.0,
    }


def _write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [k for k in rows[0].keys() if k != "params"] + ["params_json"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {k: v for k, v in row.items() if k != "params"}
            out["params_json"] = json.dumps(row["params"], sort_keys=True)
            writer.writerow(out)


def run_walk_forward(data_path: str, output_dir: str) -> Dict[str, object]:
    df = load_dataset(data_path)
    rows = [run_fold(df, fold) for fold in DEFAULT_FOLDS]
    summary = summarize_fold_rows(rows)
    payload = {
        "strategy": "simple_alpha_v1",
        "protocol": "anchored_walk_forward_refit_xgb_select_params_on_train_only_next_bar",
        "data_path": data_path,
        "summary": summary,
        "folds": rows,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_rows(out / "walk_forward_metrics.csv", rows)
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Anchored walk-forward for SimpleAlpha.")
    parser.add_argument("--data-path", default="crypto_trader/data_moe_20200101_20260216_full.csv")
    parser.add_argument("--output-dir", default="results/simple_alpha_walk_forward")
    args = parser.parse_args()
    run_walk_forward(args.data_path, args.output_dir)


if __name__ == "__main__":
    main()
