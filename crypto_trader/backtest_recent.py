from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from crypto_trader.backtest_moe import backtest_moe
except ImportError:
    from backtest_moe import backtest_moe


def backtest_recent() -> None:
    """
    Legacy entry kept for compatibility.
    Uses locked stable MoE on the current strict OOS slice.
    """
    data_path = Path("crypto_trader/data_moe_20200101_20260222_oos20.csv")
    result = backtest_moe(
        manifest_path=Path("crypto_trader/configs/moe_experts.yaml"),
        stage1_root="checkpoints/moe/stable/experts",
        stage2_root="checkpoints/moe/stable/gate",
        data_path=str(data_path),
        plot_path="results/moe_stable_locked_oos20.png",
        gate_temperature=0.68,
    )

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    summary = pd.DataFrame(
        [
            {
                "start": df.index.min().date().isoformat(),
                "end": df.index.max().date().isoformat(),
                "total_return_pct": result.get("total_return", 0.0) * 100.0,
                "benchmark_return_pct": result.get("benchmark_return", 0.0) * 100.0,
                "alpha_pct": result.get("alpha", 0.0) * 100.0,
                "max_dd_pct": result.get("max_dd", 0.0) * 100.0,
                "final_net_worth": result.get("final_net_worth", 0.0),
            }
        ]
    )
    summary.to_csv("results/backtest_recent.csv", index=False)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("saved_csv results/backtest_recent.csv")


if __name__ == "__main__":
    backtest_recent()
