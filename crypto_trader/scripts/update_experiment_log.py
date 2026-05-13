#!/usr/bin/env python3
"""Append experiment log entries from recent backtest artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

# Ensure crypto_trader is on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiment_log import append_experiment_log


DEFAULT_EXPERIMENT_LOG = Path("quant_docs/experiment_log.csv")
DEFAULT_DATA_VERSIONS = Path("quant_docs/data_versions.csv")


def _latest_data_version_hash(data_versions_path: Path) -> str:
    if not data_versions_path.exists():
        return ""
    df = pd.read_csv(data_versions_path)
    if df.empty or "hash_sha256" not in df.columns:
        return ""
    return str(df.iloc[-1]["hash_sha256"])


def _summarize_backtest_recent(path: Path) -> str:
    if not path.exists():
        return "backtest_recent.csv not found"
    df = pd.read_csv(path)
    if df.empty:
        return "backtest_recent.csv empty"
    start = df["Date"].iloc[0]
    end = df["Date"].iloc[-1]
    net_start = df["NetWorth"].iloc[0]
    net_end = df["NetWorth"].iloc[-1]
    ret = (net_end / net_start - 1) * 100
    max_dd = df["Drawdown"].max() * 100 if "Drawdown" in df.columns else 0.0
    return f"backtest_recent {start}..{end}: net {ret:.2f}% maxDD {max_dd:.2f}%"


def _summarize_walk_forward_metrics(path: Path) -> str:
    if not path.exists():
        return "walk_forward_metrics.csv not found"
    df = pd.read_csv(path)
    if df.empty:
        return "walk_forward_metrics.csv empty"
    # Summarize by average alpha and max drawdown
    avg_alpha = df["alpha"].mean() * 100
    max_dd = df["max_drawdown"].max() * 100
    return f"walk_forward folds={len(df)} avg_alpha={avg_alpha:.2f}% maxDD={max_dd:.2f}%"


def main():
    parser = argparse.ArgumentParser(description="Update quant_docs/experiment_log.csv")
    parser.add_argument("--owner", default="self", help="Owner name")
    parser.add_argument("--strategy", default="repo-local", help="Strategy version label")
    parser.add_argument("--data-versions", default=str(DEFAULT_DATA_VERSIONS))
    parser.add_argument("--experiment-log", default=str(DEFAULT_EXPERIMENT_LOG))
    parser.add_argument("--backtest-recent", default="results/backtest_recent.csv")
    parser.add_argument("--walk-forward-metrics", default="crypto_trader/walk_forward/results/walk_forward_metrics.csv")
    args = parser.parse_args()

    data_versions_path = Path(args.data_versions)
    experiment_log_path = Path(args.experiment_log)

    data_hash = _latest_data_version_hash(data_versions_path)
    backtest_summary = _summarize_backtest_recent(Path(args.backtest_recent))
    walk_forward_summary = _summarize_walk_forward_metrics(Path(args.walk_forward_metrics))

    summary = f"{backtest_summary}; {walk_forward_summary}"

    row = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "hypothesis": "Daily trend capture with PPO+XGB constraints",
        "data_version": data_hash,
        "strategy_version": args.strategy,
        "params": "ensemble=20; tau=0.25; delta=0.15; cooldown=3; vol_scale=0.05/ATR%",
        "results": summary,
        "risk_notes": "Costs/funding approximated; walk-forward metrics required",
        "decision": "review",
        "owner": args.owner,
    }

    append_experiment_log(row, experiment_log_path)
    print(f"Appended to {experiment_log_path}")


if __name__ == "__main__":
    main()
