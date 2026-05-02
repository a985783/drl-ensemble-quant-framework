from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

try:
    from crypto_trader.backtest_moe import backtest_moe
except ImportError:
    from backtest_moe import backtest_moe


def _default_manifest() -> Path:
    return Path("crypto_trader/configs/moe_experts.yaml")


def backtest_ensemble() -> None:
    """
    Legacy entry kept for compatibility.
    Now runs stable MoE backtest only.
    """
    result = backtest_moe(
        manifest_path=_default_manifest(),
        stage1_root="checkpoints/moe/stable/experts",
        stage2_root="checkpoints/moe/stable/gate",
        data_path="crypto_trader/data_moe_20200101_20260216_oos20.csv",
        plot_path="results/moe_stable_locked_oos20.png",
        gate_temperature=0.68,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def backtest_ensemble_with_config(config: "BaseConfig", max_steps: Optional[int] = None) -> Dict:
    """
    Compatibility API for sanity checks.
    """
    result = backtest_moe(
        manifest_path=_default_manifest(),
        stage1_root="checkpoints/moe/stable/experts",
        stage2_root="checkpoints/moe/stable/gate",
        data_path="crypto_trader/data_moe_20200101_20260216_oos20.csv",
        max_steps=max_steps,
        plot_path="results/moe_stable_locked_oos20.png",
        gate_temperature=0.68,
    )
    return {
        "total_return": result.get("total_return", 0.0),
        "max_dd": result.get("max_dd", 0.0),
        "first_actions": result.get("first_actions", []),
    }


if __name__ == "__main__":
    backtest_ensemble()
