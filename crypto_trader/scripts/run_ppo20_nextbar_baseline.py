"""
run_ppo20_nextbar_baseline.py

Train a 20-seed PPO ensemble on next_bar execution data,
evaluate each seed on OOS, and report aggregate metrics.

This establishes the single-agent baseline against which the
MoE revival will be compared.

Output:
  - results/candidates/ppo20_nextbar_baseline/metrics.csv
  - results/candidates/ppo20_nextbar_baseline/report.json
"""
from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from crypto_trader.asset_profile import get_asset_profile
from crypto_trader.backtest_moe import _env_kwargs_for_symbol, _mask_obs, resolve_execution_frame
from crypto_trader.config import get_default_config
from crypto_trader.envs.trading_env import TradingEnv
from crypto_trader.risk_manager import RiskManager
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Seeds – exactly 20 as specified
# ---------------------------------------------------------------------------
SEEDS: List[int] = [
    42, 123, 456, 789, 1024, 2024, 2025, 3000, 4000, 5000,
    6000, 7000, 8000, 9000, 10000, 1111, 2222, 3333, 4444, 5555,
]

# Data paths
TRAIN_CSV = _PROJECT_ROOT / "data_moe_20200101_20260216_train80.csv"
OOS_CSV = _PROJECT_ROOT / "data_moe_20200101_20260216_oos20.csv"

# Output
OUTDIR = _PROJECT_ROOT.parent / "results" / "candidates" / "ppo20_nextbar_baseline"
METRICS_CSV = OUTDIR / "metrics.csv"
REPORT_JSON = OUTDIR / "report.json"

# Symbol / interval
SYMBOL = "ETH/USDT:USDT"
INTERVAL = "1d"
INITIAL_BALANCE = 10000.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_shift(csv_path: Path) -> pd.DataFrame:
    """Load CSV, set datetime index, apply next_bar OHLCV shift."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df, meta = resolve_execution_frame(df, execution_mode="next_bar")
    assert meta["execution_mode"] == "next_bar", "Execution mode mismatch"
    print(f"  Loaded {len(df)} rows from {csv_path.name} (shifted next_bar, dropped {meta['dropped_rows']})")
    return df


def make_env(df: pd.DataFrame, risk_manager: RiskManager, env_kwargs: dict):
    """Factory for DummyVecEnv."""
    return TradingEnv(
        df,
        risk_manager=risk_manager,
        **env_kwargs,
        enable_kill_switch=False,
    )


def compute_metrics(
    net_worths: np.ndarray,
    step_returns: np.ndarray,
    turnovers: List[float],
    bench_nw: np.ndarray,
    periods_per_year: int = 365,
) -> Dict[str, float]:
    """Compute all required performance metrics."""
    nw = np.asarray(net_worths, dtype=np.float64)
    rets = np.asarray(step_returns, dtype=np.float64)

    # Basic PnL
    total_return = float((nw[-1] - INITIAL_BALANCE) / INITIAL_BALANCE)
    bench_return = float((bench_nw[-1] - INITIAL_BALANCE) / INITIAL_BALANCE)
    alpha = total_return - bench_return

    # Drawdowns
    peak = np.maximum.accumulate(nw)
    drawdowns = (peak - nw) / peak
    max_drawdown = float(drawdowns.max())

    # Sharpe (annualised, assuming 365 trading days for crypto)
    mean_ret = float(np.mean(rets)) if len(rets) > 0 else 0.0
    std_ret = float(np.std(rets, ddof=1)) if len(rets) > 1 else 1e-8
    sharpe = (mean_ret / max(std_ret, 1e-8)) * np.sqrt(periods_per_year)

    # Sortino (annualised, downside deviation only)
    neg_rets = rets[rets < 0]
    if len(neg_rets) > 1:
        downside_std = float(np.std(neg_rets, ddof=1))
    else:
        downside_std = std_ret  # fallback if no negative returns
    sortino = (mean_ret / max(downside_std, 1e-8)) * np.sqrt(periods_per_year)

    # Calmar (annualised return / max drawdown)
    annual_return = (1.0 + total_return) ** (periods_per_year / len(rets)) - 1.0 if len(rets) > 0 else 0.0
    calmar = annual_return / max(max_drawdown, 1e-8)

    # Turnover (mean per step)
    turnover = float(np.mean(turnovers)) if turnovers else 0.0

    return {
        "total_return": total_return,
        "alpha": alpha,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "turnover": turnover,
    }


def train_one_seed(
    seed: int,
    train_df: pd.DataFrame,
    config,
    env_kwargs: dict,
) -> Tuple[PPO, VecNormalize]:
    """Train a single PPO model on next-bar-shifted training data, return model + vecnorm."""
    np.random.seed(seed)

    rm = RiskManager(
        max_drawdown_limit=config.risk.max_drawdown_limit,
        freeze_period_steps=config.risk.freeze_period_steps,
        tier1_drawdown=config.risk.tier1_drawdown,
        tier1_limit=config.risk.tier1_limit,
        tier2_drawdown=config.risk.tier2_drawdown,
        tier2_limit=config.risk.tier2_limit,
        survival_drawdown=config.risk.survival_drawdown,
        survival_limit=config.risk.survival_limit,
    )

    vec_env = DummyVecEnv([lambda: make_env(train_df, rm, env_kwargs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config.model.learning_rate,
        gamma=config.model.gamma,
        n_steps=config.model.n_steps,
        batch_size=config.model.batch_size,
        ent_coef=config.model.ent_coef,
        clip_range=config.model.clip_range,
        gae_lambda=config.model.gae_lambda,
        seed=seed,
        verbose=0,
    )
    model.learn(total_timesteps=config.model.total_timesteps)
    return model, vec_env


def evaluate_one_seed(
    model: PPO,
    vecnorm: VecNormalize,
    eval_df: pd.DataFrame,
    config,
    env_kwargs: dict,
) -> Dict[str, float]:
    """Evaluate a single trained PPO model on eval data, return metrics dict."""
    # Build a fresh env + VecNormalize in eval mode
    rm = RiskManager(
        max_drawdown_limit=config.risk.max_drawdown_limit,
        freeze_period_steps=config.risk.freeze_period_steps,
        tier1_drawdown=config.risk.tier1_drawdown,
        tier1_limit=config.risk.tier1_limit,
        tier2_drawdown=config.risk.tier2_drawdown,
        tier2_limit=config.risk.tier2_limit,
        survival_drawdown=config.risk.survival_drawdown,
        survival_limit=config.risk.survival_limit,
    )

    vec_env = DummyVecEnv([lambda: make_env(eval_df, rm, env_kwargs)])
    eval_venv = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
    # Copy normalization stats from training, switch to eval mode
    eval_venv.obs_rms = vecnorm.obs_rms
    eval_venv.training = False
    eval_venv.norm_reward = False

    obs = eval_venv.reset()

    net_worths: List[float] = [INITIAL_BALANCE]
    step_returns: List[float] = []
    turnovers: List[float] = []

    n_steps = len(eval_df) - 1
    for _ in range(max(0, n_steps)):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, info = eval_venv.step(action)
        info0 = info[0]
        net_worths.append(float(info0.get("net_worth", net_worths[-1])))
        step_returns.append(float(info0.get("step_return", 0.0)))
        turnovers.append(float(info0.get("turnover", 0.0)))

    # Benchmark: ETH buy-and-hold over same period
    nw_arr = np.asarray(net_worths, dtype=np.float64)
    bench_close = eval_df["Close"].values
    bench_nw = (bench_close / bench_close[0]) * INITIAL_BALANCE
    # Align lengths (net_worth has 1 extra for initial)
    bench_nw = bench_nw[: len(nw_arr)]

    metrics = compute_metrics(
        net_worths=nw_arr,
        step_returns=np.asarray(step_returns, dtype=np.float64),
        turnovers=turnovers,
        bench_nw=bench_nw,
    )
    metrics["final_net_worth"] = float(nw_arr[-1])
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PPO20 Next-Bar Baseline")
    print("=" * 70)

    # --- Load config and env kwargs ---
    config = get_default_config()
    config.data.symbol = SYMBOL
    env_kwargs = _env_kwargs_for_symbol(SYMBOL, interval=INTERVAL)

    # --- Load & shift data ---
    print("\n[1/3] Loading data with next_bar execution shift...")
    train_df = load_and_shift(TRAIN_CSV)
    oos_df = load_and_shift(OOS_CSV)
    print(f"  Train: {len(train_df)} rows, OOS: {len(oos_df)} rows")

    # --- Train 20 seeds + evaluate ---
    print(f"\n[2/3] Training & evaluating {len(SEEDS)} seeds...")
    print(f"  PPO config: lr={config.model.learning_rate}, gamma={config.model.gamma}, "
          f"n_steps={config.model.n_steps}, batch={config.model.batch_size}, "
          f"ent={config.model.ent_coef}, clip={config.model.clip_range}, "
          f"gae_lambda={config.model.gae_lambda}, total_timesteps={config.model.total_timesteps}")

    rows: List[Dict[str, float]] = []

    for idx, seed in enumerate(SEEDS):
        print(f"\n  [{idx+1}/{len(SEEDS)}] Seed={seed}")

        # Train
        model, vecnorm = train_one_seed(seed, train_df, config, env_kwargs)

        # Evaluate
        metrics = evaluate_one_seed(model, vecnorm, oos_df, config, env_kwargs)
        metrics["seed"] = seed
        rows.append(metrics)

        print(f"    Total Return={metrics['total_return']:.4f}  "
              f"Alpha={metrics['alpha']:.4f}  "
              f"MaxDD={metrics['max_drawdown']:.4f}  "
              f"Sharpe={metrics['sharpe']:.2f}  "
              f"Sortino={metrics['sortino']:.2f}  "
              f"Calmar={metrics['calmar']:.2f}  "
              f"Turnover={metrics['turnover']:.4f}")

    # --- Aggregate & save ---
    print(f"\n[3/3] Computing aggregate stats & saving...")
    df = pd.DataFrame(rows)
    df = df.sort_values("seed").reset_index(drop=True)

    # Reorder columns
    col_order = ["seed", "total_return", "alpha", "max_drawdown", "sharpe", "sortino", "calmar", "turnover"]
    df = df[col_order]
    df.to_csv(METRICS_CSV, index=False)
    print(f"  Saved metrics.csv → {METRICS_CSV}")

    # Build report.json
    def _agg(col: str):
        vals = df[col].values
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "median": float(np.median(vals)),
        }

    report = {
        "baseline": "ppo20_nextbar",
        "execution_mode": "next_bar",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "n_seeds": len(SEEDS),
        "seeds": SEEDS,
        "ppo_config": {
            "learning_rate": config.model.learning_rate,
            "gamma": config.model.gamma,
            "n_steps": config.model.n_steps,
            "batch_size": config.model.batch_size,
            "ent_coef": config.model.ent_coef,
            "clip_range": config.model.clip_range,
            "gae_lambda": config.model.gae_lambda,
            "total_timesteps": config.model.total_timesteps,
        },
        "train_data": {
            "path": str(TRAIN_CSV),
            "rows_after_shift": len(train_df),
        },
        "oos_data": {
            "path": str(OOS_CSV),
            "rows_after_shift": len(oos_df),
            "period": f"{oos_df.index.min().date()} → {oos_df.index.max().date()}",
        },
        "aggregate": {
            "total_return": _agg("total_return"),
            "alpha": _agg("alpha"),
            "max_drawdown": _agg("max_drawdown"),
            "sharpe": _agg("sharpe"),
            "sortino": _agg("sortino"),
            "calmar": _agg("calmar"),
            "turnover": _agg("turnover"),
        },
        "all_seeds": df.to_dict(orient="records"),
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Saved report.json → {REPORT_JSON}")

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY (mean ± std across 20 seeds)")
    print("=" * 70)
    for metric in ["total_return", "alpha", "max_drawdown", "sharpe", "sortino", "calmar", "turnover"]:
        a = report["aggregate"][metric]
        print(f"  {metric:>15s}: {a['mean']:+.4f} ± {a['std']:.4f}  "
              f"[{a['min']:+.4f}, {a['max']:+.4f}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
