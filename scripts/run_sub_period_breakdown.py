#!/usr/bin/env python3
"""Sub-period breakdown for F_model_top4 configuration."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from crypto_trader.backtest_moe import backtest_moe
from crypto_trader.validation.metrics import compute_equity_metrics

DATA_PATH = "crypto_trader/data_moe_20200101_20260216_oos20.csv"
MANIFEST_PATH = Path("crypto_trader/configs/moe_experts.yaml")
STAGE1_ROOT = "checkpoints/moe/stable/experts"
STAGE2_ROOT = "checkpoints/moe/stable/gate"
DISABLED_EXPERTS = [
    "E1_PPO_trend_return",
    "E3_PPO_range_calmar",
    "E6_SAC_tail_hedge",
    "E8_A2C_regime_switch",
]
OUTPUT_DIR = Path("results/candidates/revival_final")
OUTPUT_CSV = OUTPUT_DIR / "sub_period_metrics.csv"

SUB_PERIODS = [
    ("2025-Q1", "2025-01-01", "2025-03-31"),
    ("2025-Q2", "2025-04-01", "2025-06-30"),
    ("2025-Q3", "2025-07-01", "2025-09-30"),
    ("2025-Q4", "2025-10-01", "2025-12-31"),
    ("2026-Q1", "2026-01-01", "2026-02-28"),
    ("Full OOS", "2025-01-01", "2026-02-28"),
]

TMP_DIR = Path("/var/folders/h0/_kjdc9hd5j744_wj42tfx9900000gn/T/opencode")


def count_trades(positions: List[float]) -> int:
    trades = 0
    prev = None
    for p in positions:
        if prev is not None and abs(p - prev) > 1e-6:
            trades += 1
        prev = p
    return trades


def run_period(period_name: str, start: str, end: str) -> dict:
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    sub = df.loc[start:end]
    if len(sub) == 0:
        return {"period": period_name, "error": "no data"}

    tmp_path = TMP_DIR / f"sub_{period_name.replace(' ', '_').replace('-', '_')}.csv"
    sub.to_csv(tmp_path)

    result = backtest_moe(
        manifest_path=MANIFEST_PATH,
        stage1_root=STAGE1_ROOT,
        stage2_root=STAGE2_ROOT,
        data_path=str(tmp_path),
        gate_temperature=0.68,
        symbol="ETH/USDT:USDT",
        env_overrides={"tau": 0.12},
        gate_mode="model",
        disabled_experts=DISABLED_EXPERTS,
        execution_mode="next_bar",
        return_history=True,
    )

    tmp_path.unlink(missing_ok=True)

    if "error" in result:
        return {"period": period_name, "error": str(result["error"])}

    history = result.get("history", {})
    positions = history.get("positions", [])
    net_worth = history.get("net_worth", [])
    benchmark = history.get("benchmark_values", [])

    metrics = compute_equity_metrics(
        net_worth=net_worth,
        benchmark_values=benchmark,
        positions=positions,
        turnovers=history.get("turnovers", []),
        trade_costs=history.get("trade_costs", []),
        funding_costs=history.get("funding_costs", []),
    )

    return {
        "period": period_name,
        "total_return": float(metrics["total_return"]),
        "alpha": float(metrics["alpha"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "sharpe": float(metrics["sharpe"]),
        "sortino": float(metrics["sortino"]),
        "num_trades": count_trades(positions),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for period_name, start, end in SUB_PERIODS:
        print(f"Running {period_name} ({start} to {end})...")
        row = run_period(period_name, start, end)
        rows.append(row)
        print(f"  Result: {row}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "period", "total_return", "alpha", "max_drawdown",
            "sharpe", "sortino", "num_trades"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
