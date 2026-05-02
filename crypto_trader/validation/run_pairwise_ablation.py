"""
Pairwise Expert Conflict Analysis.

Tests whether any specific pairs of the top-4 experts (E5, E2, E4, E7) have
destructive interactions that degrade performance when combined.

Steps:
  1. Run 4 individual expert backtests (only 1 expert active).
  2. Run 6 pairwise backtests (only 2 experts active).
  3. Run 1 all-top-4 backtest for reference.
  4. Compute degradation for each pair:
       degradation = max(single_A, single_B) - combined
       is_destructive = combined < max(single_A, single_B)

Output:  results/candidates/arch_simplified/pairwise_conflicts.csv

Usage:
    PYTHONPATH=. python -m crypto_trader.validation.run_pairwise_ablation
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

# Top-4 experts by individual next_bar return
TOP_4 = ["E5_PPO_lowvol_carry", "E2_PPO_bear_drawdown",
         "E4_PPO_highvol_risk", "E7_SAC_fast_adapt"]

# All possible pairs within top-4
TOP_4_PAIRS = [
    ("E5_PPO_lowvol_carry", "E2_PPO_bear_drawdown"),
    ("E5_PPO_lowvol_carry", "E4_PPO_highvol_risk"),
    ("E5_PPO_lowvol_carry", "E7_SAC_fast_adapt"),
    ("E2_PPO_bear_drawdown", "E4_PPO_highvol_risk"),
    ("E2_PPO_bear_drawdown", "E7_SAC_fast_adapt"),
    ("E4_PPO_highvol_risk", "E7_SAC_fast_adapt"),
]

# Short names for display
EXPERT_SHORT = {
    "E1_PPO_trend_return": "E1",
    "E2_PPO_bear_drawdown": "E2",
    "E3_PPO_range_calmar": "E3",
    "E4_PPO_highvol_risk": "E4",
    "E5_PPO_lowvol_carry": "E5",
    "E6_SAC_tail_hedge": "E6",
    "E7_SAC_fast_adapt": "E7",
    "E8_A2C_regime_switch": "E8",
}

DELTA_MAX = 0.15
COOLDOWN_N = 3
K_SINGLE = 0.0008
FUNDING_DAILY = 0.0003
TAU = 0.20
TEMPERATURE = 0.68


def _env_overrides_dict() -> Dict[str, float]:
    return {
        "tau": TAU,
        "delta_max": DELTA_MAX,
        "cooldown_n": COOLDOWN_N,
        "k_single": K_SINGLE,
        "funding_daily": FUNDING_DAILY,
    }


def _other_experts(keep: List[str]) -> List[str]:
    """Return all experts NOT in `keep`."""
    keep_set = set(keep)
    return [e for e in ALL_EXPERTS if e not in keep_set]


def _run_single(
    config_name: str,
    disabled_experts: List[str],
    plot_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single backtest with average_experts gate mode and return metrics."""
    env_overrides = _env_overrides_dict()

    result = backtest_moe(
        manifest_path=MANIFEST,
        stage1_root=STAGE1_ROOT,
        stage2_root=STAGE2_ROOT,
        data_path=DATA_PATH,
        gate_temperature=TEMPERATURE,
        symbol=SYMBOL,
        env_overrides=env_overrides,
        gate_mode="average_experts",
        disabled_experts=disabled_experts,
        execution_mode="next_bar",
        return_history=True,
        plot_path=plot_path or str(OUTPUT_DIR / f"pairwise_{config_name}.png"),
    )

    if "error" in result:
        return {"config": config_name, "status": "error",
                "error": result.get("error", "unknown")}

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
        "disabled_experts": "|".join(disabled_experts) if disabled_experts else "",
    }


def _count_trades(positions: List[float], min_change: float = 0.01) -> int:
    if not positions or len(positions) < 2:
        return 0
    changes = 0
    prev = positions[0]
    for p in positions[1:]:
        if abs(p - prev) > min_change:
            changes += 1
        prev = p
    return changes


def run_pairwise_ablation() -> List[Dict[str, Any]]:
    """Execute all pairwise configurations and return results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    single_results: Dict[str, Dict[str, Any]] = {}
    pair_results: List[Dict[str, Any]] = []

    # ── Phase 1: Individual Expert Backtests ─────────────────────────────
    print("=" * 80)
    print("PHASE 1: Individual Expert Backtests")
    print("=" * 80)
    n_individuals = len(TOP_4)
    for idx, expert in enumerate(TOP_4, 1):
        short = EXPERT_SHORT[expert]
        config_name = f"single_{short}"
        disabled = _other_experts([expert])
        print(f"[{idx}/{n_individuals}] {config_name} (disabled {len(disabled)} others) ... ",
              end="", flush=True)

        entry = _run_single(config_name=config_name, disabled_experts=disabled)
        if entry["status"] == "error":
            print(f"ERROR: {entry.get('error', '?')}")
        else:
            print(f"return={entry['total_return']:+.2%}  "
                  f"alpha={entry['alpha']:+.2%}  "
                  f"max_dd={entry['max_drawdown']:.2%}  "
                  f"sharpe={entry['sharpe']:+.2f}")
        single_results[expert] = entry
    print()

    # ── Phase 2: Pairwise Backtests ──────────────────────────────────────
    print("=" * 80)
    print("PHASE 2: Pairwise Expert Backtests (6 pairs)")
    print("=" * 80)
    n_pairs = len(TOP_4_PAIRS)
    for idx, (expert_a, expert_b) in enumerate(TOP_4_PAIRS, 1):
        short_a = EXPERT_SHORT[expert_a]
        short_b = EXPERT_SHORT[expert_b]
        config_name = f"pair_{short_a}_{short_b}"
        disabled = _other_experts([expert_a, expert_b])
        print(f"[{idx}/{n_pairs}] {config_name} (disabled {len(disabled)} others) ... ",
              end="", flush=True)

        entry = _run_single(config_name=config_name, disabled_experts=disabled)
        if entry["status"] == "error":
            print(f"ERROR: {entry.get('error', '?')}")
        else:
            print(f"return={entry['total_return']:+.2%}  "
                  f"alpha={entry['alpha']:+.2%}  "
                  f"max_dd={entry['max_drawdown']:.2%}  "
                  f"sharpe={entry['sharpe']:+.2f}")
        pair_results.append(entry)
    print()

    # ── Phase 3: All Top-4 Backtest ──────────────────────────────────────
    print("=" * 80)
    print("PHASE 3: All Top-4 Backtest")
    print("=" * 80)
    config_name = "all_top4"
    disabled = _other_experts(TOP_4)
    print(f"[1/1] {config_name} (disabled {len(disabled)} experts) ... ",
          end="", flush=True)
    entry_all4 = _run_single(config_name=config_name, disabled_experts=disabled)
    if entry_all4["status"] == "error":
        print(f"ERROR: {entry_all4.get('error', '?')}")
    else:
        print(f"return={entry_all4['total_return']:+.2%}  "
              f"alpha={entry_all4['alpha']:+.2%}  "
              f"max_dd={entry_all4['max_drawdown']:.2%}  "
              f"sharpe={entry_all4['sharpe']:+.2f}")
    print()

    # ── Phase 4: Compute Conflicts ───────────────────────────────────────
    print("=" * 80)
    print("PHASE 4: Conflict Analysis")
    print("=" * 80)

    conflict_rows: List[Dict[str, Any]] = []

    for (expert_a, expert_b), pair_entry in zip(TOP_4_PAIRS, pair_results):
        short_a = EXPERT_SHORT[expert_a]
        short_b = EXPERT_SHORT[expert_b]
        sa = single_results[expert_a]
        sb = single_results[expert_b]

        if sa["status"] != "ok" or sb["status"] != "ok" or pair_entry["status"] != "ok":
            conflict_rows.append({
                "pair": f"{short_a}+{short_b}",
                "single_A_return": sa.get("total_return", 0),
                "single_B_return": sb.get("total_return", 0),
                "combined_return": pair_entry.get("total_return", 0),
                "max_individual_return": 0,
                "degradation_percent": 0,
                "is_destructive": "skipped",
                "status": "error",
            })
            continue

        combined_return = pair_entry["total_return"]
        max_individual = max(sa["total_return"], sb["total_return"])
        degradation = max_individual - combined_return
        is_destructive = combined_return < max_individual

        row = {
            "pair": f"{short_a}+{short_b}",
            "single_A_id": short_a,
            "single_A_return": sa["total_return"],
            "single_A_alpha": sa["alpha"],
            "single_B_id": short_b,
            "single_B_return": sb["total_return"],
            "single_B_alpha": sb["alpha"],
            "combined_return": combined_return,
            "combined_alpha": pair_entry["alpha"],
            "combined_max_dd": pair_entry["max_drawdown"],
            "combined_sharpe": pair_entry["sharpe"],
            "max_individual_return": max_individual,
            "degradation_percent": degradation,
            "is_destructive": str(is_destructive).lower(),
            "status": "ok",
        }
        conflict_rows.append(row)

        symbol_mark = "⚠️  DESTRUCTIVE" if is_destructive else "✅  SYNERGISTIC"
        print(f"  {short_a}+{short_b}:  combined={combined_return:+.2%}  "
              f"best_solo={max_individual:+.2%}  "
              f"degradation={degradation:+.2%}  {symbol_mark}")

    # ── Phase 5: All-Top-4 Synergy ─────────────────────────────────────────
    print()
    if entry_all4["status"] == "ok":
        best_pair_return = max(r["combined_return"] for r in conflict_rows if r["status"] == "ok")
        synergy = entry_all4["total_return"] - best_pair_return
        synergy_mark = "✅  Positive Synergy" if synergy > 0 else "⚠️  No Additional Benefit"
        print(f"  all_top4:         return={entry_all4['total_return']:+.2%}")
        print(f"  best_pair:        return={best_pair_return:+.2%}")
        print(f"  synergy (all4 - best_pair): {synergy:+.2%}  {synergy_mark}")

        # Add synergy row
        conflict_rows.append({
            "pair": "ALL_TOP4",
            "single_A_id": "",
            "single_A_return": 0,
            "single_A_alpha": 0,
            "single_B_id": "",
            "single_B_return": 0,
            "single_B_alpha": 0,
            "combined_return": entry_all4["total_return"],
            "combined_alpha": entry_all4["alpha"],
            "combined_max_dd": entry_all4["max_drawdown"],
            "combined_sharpe": entry_all4["sharpe"],
            "max_individual_return": best_pair_return,
            "degradation_percent": -synergy,  # negative degradation = synergy
            "is_destructive": "false",
            "status": "ok",
        })

    # ── Write CSV ─────────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "pairwise_conflicts.csv"
    _write_conflict_csv(conflict_rows, csv_path)
    print(f"\nResults written to {csv_path}")

    # ── Write JSON ────────────────────────────────────────────────────────
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = OUTPUT_DIR / f"pairwise_conflicts_{timestamp}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "single_results": {k: v for k, v in single_results.items()},
            "pair_results": pair_results,
            "all_top4": entry_all4,
            "conflicts": conflict_rows,
        }, f, indent=2)

    # ── Recommendations ───────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    destructive_pairs = [r for r in conflict_rows
                         if r["is_destructive"] == "true" and r["status"] == "ok"]
    if destructive_pairs:
        print(f"⚠️  {len(destructive_pairs)} destructive pair(s) found:")
        for r in destructive_pairs:
            print(f"    {r['pair']}: degradation={r['degradation_percent']:+.2%}")
        print("   Recommended: Remove the weaker expert from destructive pairs")
    else:
        print("✅  No destructive pairs – all combinations are synergistic!")
        print("   Recommended: Use all top-4 experts together.")

    return conflict_rows


def _write_conflict_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "pair",
        "single_A_id", "single_A_return", "single_A_alpha",
        "single_B_id", "single_B_return", "single_B_alpha",
        "combined_return", "combined_alpha", "combined_max_dd", "combined_sharpe",
        "max_individual_return", "degradation_percent", "is_destructive",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    run_pairwise_ablation()


if __name__ == "__main__":
    main()
