"""
Evaluate MoE experts independently under next_bar (corrected) execution.

Each expert runs as a standalone agent: its model + VecNormalize + feature_mask.
No Gate, no ensemble.  Metrics: total_return, alpha, max_drawdown,
sharpe, sortino, turnover, num_trades.

Usage:
    PYTHONPATH=. python crypto_trader/scripts/eval_moe_experts.py \
        --execution-mode next_bar \
        --output results/candidates/expert_audit_nextbar/expert_metrics.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from crypto_trader.envs.trading_env import TradingEnv
from crypto_trader.train_moe_stage2_gate import (
    _env_kwargs_for_symbol,
    build_gate_artifacts,
)
from crypto_trader.backtest_moe import _mask_obs, resolve_execution_frame
from crypto_trader.config import get_default_config

ALGO_MAP = {
    "ppo": PPO,
    "sac": SAC,
    "a2c": A2C,
}


def compute_sharpe(returns: np.ndarray, annual_factor: int = 252) -> float:
    """Annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    if sigma == 0:
        return 0.0
    return mu / sigma * np.sqrt(annual_factor)


def compute_sortino(returns: np.ndarray, annual_factor: int = 252) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    if len(returns) < 2:
        return 0.0
    mu = float(np.mean(returns))
    negative = returns[returns < 0]
    if len(negative) < 2:
        return 0.0
    down_std = float(np.std(negative, ddof=1))
    if down_std == 0:
        return 0.0
    return mu / down_std * np.sqrt(annual_factor)


def evaluate_expert(
    artifact,
    df: pd.DataFrame,
    symbol: str,
    interval: str,
) -> Optional[Dict]:
    """Evaluate a single expert on the given dataframe. Returns metrics dict or None on failure."""
    expert_id = artifact.expert_id
    algo_name = artifact.algorithm.lower()

    if algo_name not in ALGO_MAP:
        print(f"  [{expert_id}]  SKIP: unsupported algo '{algo_name}'")
        return None

    # ── Load model ──────────────────────────────────────────────────────
    algo_cls = ALGO_MAP[algo_name]
    try:
        model = algo_cls.load(str(artifact.model_path))
    except Exception as e:
        print(f"  [{expert_id}]  FAIL: model load error: {e}")
        return None

    # ── Load VecNormalize ────────────────────────────────────────────────
    env_kwargs = _env_kwargs_for_symbol(symbol, interval=interval)

    def make_temp_env(mask=artifact.feature_mask):
        kwargs = dict(env_kwargs)
        return TradingEnv(df, **kwargs, feature_mask=mask)

    temp_vec = DummyVecEnv([make_temp_env])
    try:
        vecnorm = VecNormalize.load(str(artifact.vecnorm_path), temp_vec)
        vecnorm.training = False
        vecnorm.norm_reward = False
    except Exception as e:
        print(f"  [{expert_id}]  FAIL: VecNormalize load error: {e}")
        return None

    # ── Create fresh eval env (with next_bar data) ───────────────────────
    env_kwargs_for_env = dict(env_kwargs)
    env_kwargs_for_env.pop("symbol", None)  # avoid duplicate with explicit param
    env = TradingEnv(
        df,
        symbol=symbol,
        initial_balance=10000.0,
        enable_kill_switch=False,
        feature_mask=artifact.feature_mask,
        **env_kwargs_for_env,
    )

    obs, _ = env.reset()
    done = False

    # ── Accumulators ────────────────────────────────────────────────────
    max_nw = 10000.0
    max_dd = 0.0
    step_returns: List[float] = []
    total_turnover = 0.0
    num_trades = 0

    while not done:
        # obs is already masked by TradingEnv._get_observation()
        norm_obs = vecnorm.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(norm_obs, deterministic=True)
        action_1d = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = env.step(action_1d)
        done = terminated or truncated

        nw = float(info["net_worth"])
        if nw > max_nw:
            max_nw = nw
        dd = (max_nw - nw) / max_nw if max_nw > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

        step_returns.append(float(info["step_return"]))
        total_turnover += float(info["turnover"])
        if float(info["turnover"]) > 0.001:
            num_trades += 1

    final_nw = float(info.get("net_worth", 10000.0))
    total_ret = (final_nw - 10000.0) / 10000.0

    returns_arr = np.array(step_returns, dtype=np.float64)
    sharpe = compute_sharpe(returns_arr)
    sortino = compute_sortino(returns_arr)

    return {
        "expert_id": expert_id,
        "total_return": round(total_ret * 100.0, 2),
        "alpha": 0.0,  # filled later after benchmark is known
        "max_drawdown": round(max_dd * 100.0, 2),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "turnover": round(total_turnover, 4),
        "num_trades": num_trades,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MoE experts independently under next_bar execution"
    )
    parser.add_argument(
        "--execution-mode",
        default="next_bar",
        choices=["next_bar", "same_bar"],
        help="Execution mode: next_bar (shift OHLCV by 1) or same_bar (legacy)",
    )
    parser.add_argument(
        "--output",
        default="results/candidates/expert_audit_nextbar/expert_metrics.csv",
        help="Output CSV path for expert metrics",
    )
    parser.add_argument(
        "--data-path",
        default="crypto_trader/data_moe_20200101_20260216_oos20.csv",
        help="Path to OOS test data CSV",
    )
    parser.add_argument(
        "--manifest",
        default="crypto_trader/configs/moe_experts.yaml",
        help="Path to expert manifest YAML",
    )
    parser.add_argument(
        "--stage1-root",
        default="checkpoints/moe/stable/experts",
        help="Root directory for expert checkpoints",
    )
    parser.add_argument(
        "--symbol",
        default="ETH/USDT:USDT",
        help="Trading pair symbol for env configuration",
    )
    args = parser.parse_args()

    # ── Validate args ───────────────────────────────────────────────────
    exec_mode = args.execution_mode.strip().lower()
    data_path = Path(args.data_path)
    manifest_path = Path(args.manifest)
    stage1_root = Path(args.stage1_root)
    output_path = Path(args.output)

    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}")
        sys.exit(1)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}")
        sys.exit(1)
    if not stage1_root.exists():
        print(f"ERROR: stage1 root not found: {stage1_root}")
        sys.exit(1)

    # ── Load data ───────────────────────────────────────────────────────
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"  Raw rows: {len(df)}, columns: {list(df.columns)}")

    # ── Apply execution mode ────────────────────────────────────────────
    if exec_mode == "next_bar":
        df, meta = resolve_execution_frame(df, "next_bar")
        print(f"  next_bar mode: dropped {meta.get('dropped_rows', 0)} row(s), "
              f"remaining rows: {len(df)}")
    else:
        print("  same_bar mode (legacy)")

    # ── Compute ETH buy-and-hold benchmark (on the same shifted data) ───
    close_series = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close_series) < 2:
        print("ERROR: insufficient Close data for benchmark")
        sys.exit(1)
    bench_return = (close_series.iloc[-1] - close_series.iloc[0]) / close_series.iloc[0]
    print(f"  ETH Buy & Hold benchmark: {bench_return * 100:.2f}%")

    # ── Build expert artifacts ─────────────────────────────────────────
    print(f"\nBuilding expert artifacts from {manifest_path}...")
    artifacts = build_gate_artifacts(manifest_path, str(stage1_root))
    print(f"  Found {len(artifacts)} experts")

    # ── Evaluate each expert ───────────────────────────────────────────
    results: List[Dict] = []
    interval = "1d"

    for i, artifact in enumerate(artifacts, 1):
        print(f"\n[{i}/{len(artifacts)}] Evaluating {artifact.expert_id} "
              f"({artifact.algorithm.upper()}, mask={artifact.feature_mask})...")
        metrics = evaluate_expert(
            artifact,
            df,
            symbol=args.symbol,
            interval=interval,
        )
        if metrics is None:
            continue
        # Fill alpha after benchmark is known
        metrics["alpha"] = round(metrics["total_return"] - bench_return * 100.0, 2)
        results.append(metrics)
        print(f"  Return: {metrics['total_return']:+.2f}%  "
              f"Alpha: {metrics['alpha']:+.2f}%  "
              f"MaxDD: {metrics['max_drawdown']:.2f}%  "
              f"Sharpe: {metrics['sharpe']:.4f}  "
              f"Sortino: {metrics['sortino']:.4f}  "
              f"Trades: {metrics['num_trades']}")

    if not results:
        print("\nERROR: no experts were successfully evaluated")
        sys.exit(1)

    # ── Save results ───────────────────────────────────────────────────
    res_df = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # ── Print summary table ────────────────────────────────────────────
    res_df_sorted = res_df.sort_values("total_return", ascending=False)
    random_baseline = -5.40  # approximate random policy OOS

    print("\n" + "=" * 90)
    print(f" EXPERT AUDIT — Next-Bar Execution — OOS: {close_series.index[0].date()} to {close_series.index[-1].date()} ".center(90, "="))
    print("=" * 90)
    print(f"{'Rank':<5} {'Expert':<25} {'Return%':>8} {'Alpha%':>8} {'MaxDD%':>8} "
          f"{'Sharpe':>8} {'Sortino':>8} {'Turnover':>9} {'Trades':>7}  {'vs Random':>10}")
    print("-" * 90)

    for rank, (_, row) in enumerate(res_df_sorted.iterrows(), 1):
        beats_random = "✓ BEATS" if row["total_return"] > random_baseline else "✗ below"
        print(f"{rank:<5} {row['expert_id']:<25} "
              f"{row['total_return']:>+7.2f}% "
              f"{row['alpha']:>+7.2f}% "
              f"{row['max_drawdown']:>7.2f}% "
              f"{row['sharpe']:>8.4f} "
              f"{row['sortino']:>8.4f} "
              f"{row['turnover']:>9.4f} "
              f"{row['num_trades']:>7d}  "
              f"{beats_random:>10}")

    print("-" * 90)
    print(f"  Random baseline (approx): {random_baseline:+.2f}%")
    print(f"  ETH Buy & Hold:            {bench_return * 100:+.2f}%")

    beats_count = int((res_df_sorted["total_return"] > random_baseline).sum())
    print(f"  Experts beating random:    {beats_count}/{len(res_df_sorted)}")
    print("=" * 90)


if __name__ == "__main__":
    main()
