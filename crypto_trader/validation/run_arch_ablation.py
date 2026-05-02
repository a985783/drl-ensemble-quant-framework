"""
Simplified Architecture Ablation Runner.

Tests simplified MoE architectures under next_bar execution using existing
stable checkpoints (no retraining).

Configurations:
  A: uniform gate + top 3 experts (E5, E2, E4)
  B: uniform gate + top 4 experts (E5, E2, E4, E7)
  C: average_experts gate + top 3 experts
  D: average_experts gate + top 4 experts
  E: average_experts gate + all 8 experts (control)
  F: model gate + top 4 experts
  G: uniform gate + top 3 experts, tau sweep [0.10, 0.15, 0.20, 0.25, 0.30]
  H: original stable baseline (control) – model gate, all 8, tau=0.25

Output:  results/candidates/arch_simplified/arch_comparison.csv

Usage:
    PYTHONPATH=. python -m crypto_trader.validation.run_arch_ablation
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from crypto_trader.backtest_moe import backtest_moe
    from crypto_trader.validation.metrics import compute_equity_metrics
except ImportError:
    from backtest_moe import backtest_moe
    from validation.metrics import compute_equity_metrics


OUTPUT_DIR = Path("results/candidates/arch_simplified")
MANIFEST = Path("crypto_trader/configs/moe_experts.yaml")
STAGE1_ROOT = "checkpoints/moe/stable/experts"
STAGE2_ROOT = "checkpoints/moe/stable/gate"
DATA_PATH = "crypto_trader/data_moe_20200101_20260216_oos20.csv"
SYMBOL = "ETH/USDT:USDT"

ALL_EXPERTS = [
    "E1_PPO_trend_return",
    "E2_PPO_bear_drawdown",
    "E3_PPO_range_calmar",
    "E4_PPO_highvol_risk",
    "E5_PPO_lowvol_carry",
    "E6_SAC_tail_hedge",
    "E7_SAC_fast_adapt",
    "E8_A2C_regime_switch",
]

# Expert selection based on next_bar audit: E5 +157%, E2 +150%, E4 +120%, E7 +99%
# are individually profitable; E1/E3/E6/E8 are not.
TOP_3 = ["E5_PPO_lowvol_carry", "E2_PPO_bear_drawdown", "E4_PPO_highvol_risk"]
TOP_4 = TOP_3 + ["E7_SAC_fast_adapt"]
BOTTOM_5 = ["E1_PPO_trend_return", "E3_PPO_range_calmar",
            "E6_SAC_tail_hedge", "E7_SAC_fast_adapt", "E8_A2C_regime_switch"]
BOTTOM_4 = ["E1_PPO_trend_return", "E3_PPO_range_calmar",
            "E6_SAC_tail_hedge", "E8_A2C_regime_switch"]

DELTA_MAX = 0.15
COOLDOWN_N = 3
K_SINGLE = 0.0008
FUNDING_DAILY = 0.0003
TEMPERATURE = 0.68

def _env_overrides_dict(tau: float) -> Dict[str, float]:
    return {
        "tau": tau,
        "delta_max": DELTA_MAX,
        "cooldown_n": COOLDOWN_N,
        "k_single": K_SINGLE,
        "funding_daily": FUNDING_DAILY,
    }


def _run_single(
    config_name: str,
    gate_mode: str,
    disabled_experts: List[str],
    tau: float,
    plot_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single backtest and return computed metrics + raw result."""
    env_overrides = _env_overrides_dict(tau)

    result = backtest_moe(
        manifest_path=MANIFEST,
        stage1_root=STAGE1_ROOT,
        stage2_root=STAGE2_ROOT,
        data_path=DATA_PATH,
        gate_temperature=TEMPERATURE,
        symbol=SYMBOL,
        env_overrides=env_overrides,
        gate_mode=gate_mode,
        disabled_experts=disabled_experts,
        execution_mode="next_bar",
        return_history=True,
        plot_path=plot_path or str(OUTPUT_DIR / f"{config_name}.png"),
    )

    if "error" in result:
        return {"config": config_name, "status": "error", "error": result.get("error", "unknown")}

    # Compute equity metrics from history
    history = result.get("history")
    if isinstance(history, dict) and history.get("net_worth"):
        computed = compute_equity_metrics(
            net_worth=history["net_worth"],
            benchmark_values=history.get("benchmark_values"),
            positions=history.get("positions"),
            turnovers=history.get("turnovers"),
            trade_costs=history.get("trade_costs"),
            funding_costs=history.get("funding_costs"),
        )
    else:
        computed = {
            "total_return": result.get("total_return", 0.0),
            "benchmark_return": result.get("benchmark_return", 0.0),
            "alpha": result.get("alpha", 0.0),
            "max_drawdown": result.get("max_dd", 0.0),
        }

    # Count trades: number of times position changes significantly
    positions = history.get("positions", []) if isinstance(history, dict) else []
    num_trades = _count_trades(positions)

    return {
        "config": config_name,
        "status": "ok",
        "total_return": computed.get("total_return", 0.0),
        "benchmark_return": computed.get("benchmark_return", 0.0),
        "alpha": computed.get("alpha", 0.0),
        "max_drawdown": computed.get("max_drawdown", 0.0),
        "sharpe": computed.get("sharpe", 0.0),
        "sortino": computed.get("sortino", 0.0),
        "turnover": computed.get("turnover", 0.0),
        "num_trades": num_trades,
        "final_net_worth": computed.get("final_net_worth", 0.0),
        # metadata for reproducibility
        "gate_mode": gate_mode,
        "disabled_experts": "|".join(disabled_experts) if disabled_experts else "",
        "tau": tau,
    }


def _count_trades(positions: List[float], min_change: float = 0.01) -> int:
    """Count number of significant position changes."""
    if not positions or len(positions) < 2:
        return 0
    changes = 0
    prev = positions[0]
    for p in positions[1:]:
        if abs(p - prev) > min_change:
            changes += 1
        prev = p
    return changes


# ---------------------------------------------------------------------------
# Main ablation run
# ---------------------------------------------------------------------------

def run_ablation() -> List[Dict[str, Any]]:
    """Execute all architecture configurations and return results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ── Config Definitions ────────────────────────────────────────────────
    configs: List[Tuple[str, str, List[str], float]] = [
        # (name, gate_mode, disabled_experts, tau)
        ("A_uniform_top3",   "uniform",         BOTTOM_5,  0.20),
        ("B_uniform_top4",   "uniform",         BOTTOM_4,  0.20),
        ("C_average_top3",   "average_experts", BOTTOM_5,  0.20),
        ("D_average_top4",   "average_experts", BOTTOM_4,  0.20),
        ("E_average_all8",   "average_experts", [],         0.20),
        ("F_model_top4",     "model",           BOTTOM_4,  0.20),
        ("H_stable_baseline","model",           [],         0.25),
    ]

    # Config G: tau sweep
    for tau_val in [0.10, 0.15, 0.20, 0.25, 0.30]:
        name = f"G_tau_{str(tau_val).replace('.', 'p')}"
        configs.append((name, "uniform", BOTTOM_5, tau_val))

    n_total = len(configs)
    print(f"Running {n_total} architecture ablation configurations...")
    print(f"Output dir: {OUTPUT_DIR}")
    print()

    for idx, (name, gate_mode, disabled, tau) in enumerate(configs, 1):
        print(f"[{idx}/{n_total}] {name}  gate={gate_mode}  "
              f"disabled={len(disabled)}  tau={tau} ... ", end="", flush=True)

        entry = _run_single(
            config_name=name,
            gate_mode=gate_mode,
            disabled_experts=disabled,
            tau=tau,
        )

        if entry["status"] == "error":
            print(f"ERROR: {entry.get('error', '?')}")
        else:
            ret = entry["total_return"]
            alpha = entry["alpha"]
            dd = entry["max_drawdown"]
            sr = entry["sharpe"]
            print(f"return={ret:+.2%}  alpha={alpha:+.2%}  max_dd={dd:.2%}  sharpe={sr:+.2f}")

        results.append(entry)

    # ── Write results CSV ─────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "arch_comparison.csv"
    _write_results_csv(results, csv_path)
    print(f"\nResults written to {csv_path}")

    # ── Print summary table ───────────────────────────────────────────────
    print()
    print("=" * 100)
    print(f"{'Config':<25} {'Return':>8} {'Alpha':>8} {'MaxDD':>8} "
          f"{'Sharpe':>8} {'Sortino':>8} {'Trades':>7} {'Turnover':>9}")
    print("-" * 100)
    for entry in results:
        if entry["status"] != "ok":
            print(f"{entry['config']:<25} {'ERROR':>8}")
            continue
        print(f"{entry['config']:<25} {entry['total_return']:>7.2%} "
              f"{entry['alpha']:>7.2%} {entry['max_drawdown']:>7.2%} "
              f"{entry['sharpe']:>7.2f} {entry['sortino']:>7.2f} "
              f"{entry['num_trades']:>7} {entry['turnover']:>8.2f}")
    print("-" * 100)

    # ── Write summary JSON for programmatic consumption ───────────────────
    summary_path = OUTPUT_DIR / f"arch_ablation_{timestamp}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def _write_results_csv(results: List[Dict[str, Any]], path: Path) -> None:
    """Write ablation results CSV."""
    fieldnames = [
        "config", "total_return", "alpha", "max_drawdown",
        "sharpe", "sortino", "turnover", "num_trades",
        "gate_mode", "disabled_experts", "tau",
        "benchmark_return", "final_net_worth", "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    run_ablation()


if __name__ == "__main__":
    main()
