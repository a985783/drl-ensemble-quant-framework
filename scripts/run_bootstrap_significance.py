#!/usr/bin/env python3
"""
Bootstrap significance test for top-5 parameter combinations.

Runs backtest_moe for each configuration, collects step-level returns,
and computes 95% bootstrap CI on both mean step return and compounded total return.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from crypto_trader.backtest_moe import backtest_moe
from crypto_trader.validation.metrics import bootstrap_confidence_interval

E1 = "E1_PPO_trend_return"
E2 = "E2_PPO_bear_drawdown"
E3 = "E3_PPO_range_calmar"
E4 = "E4_PPO_highvol_risk"
E5 = "E5_PPO_lowvol_carry"
E6 = "E6_SAC_tail_hedge"
E7 = "E7_SAC_fast_adapt"
E8 = "E8_A2C_regime_switch"

CONFIGS = [
    {"name": "F_model_top4", "tau": 0.20, "gate_mode": "model",
     "disabled_experts": [E1, E3, E6, E8]},
    {"name": "tau_0_12", "tau": 0.12, "gate_mode": "model",
     "disabled_experts": []},
    {"name": "tau_0_15", "tau": 0.15, "gate_mode": "model",
     "disabled_experts": []},
    {"name": "E_average_all8_tau_0_20", "tau": 0.20, "gate_mode": "average_experts",
     "disabled_experts": []},
    {"name": "H_stable_baseline", "tau": 0.25, "gate_mode": "model",
     "disabled_experts": []},
]

MANIFEST = "crypto_trader/configs/moe_experts.yaml"
DATA_PATH = "crypto_trader/data_moe_20200101_20260216_oos20.csv"
STAGE1_ROOT = "checkpoints/moe/stable/experts"
STAGE2_ROOT = "checkpoints/moe/stable/gate"
OUTPUT_PATH = "results/candidates/param_sweep/bootstrap_results.csv"
GATE_TEMPERATURE = 0.68  # stable model training temperature

N_BOOTSTRAP = 10000
CI_LEVEL = 0.95


def compute_bootstrap_on_total_return(step_returns, n_bootstrap=N_BOOTSTRAP,
                                      ci=CI_LEVEL, seed=42):
    rng = np.random.default_rng(seed)
    n_steps = len(step_returns)
    total_returns = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(step_returns, size=n_steps, replace=True)
        total_returns[i] = float(np.prod(1.0 + sample) - 1.0)
    alpha = 1.0 - ci
    lower = float(np.percentile(total_returns, 100.0 * alpha / 2.0))
    upper = float(np.percentile(total_returns, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper, total_returns


print("=" * 72)
print("Bootstrap Significance Test for Top-5 Parameter Combinations")
print("=" * 72)

all_rows = []

for config in CONFIGS:
    print(f"\n{'─' * 72}")
    print(f"Config: {config['name']}")
    print(f"  tau={config['tau']}, gate_mode={config['gate_mode']}, "
          f"disabled={config['disabled_experts'] or 'none'}")

    env_overrides = {"tau": config["tau"]}
    result = backtest_moe(
        manifest_path=Path(MANIFEST),
        stage1_root=STAGE1_ROOT,
        stage2_root=STAGE2_ROOT,
        data_path=DATA_PATH,
        gate_temperature=GATE_TEMPERATURE,
        symbol="ETH/USDT:USDT",
        env_overrides=env_overrides,
        gate_mode=config["gate_mode"],
        disabled_experts=config["disabled_experts"] or None,
        return_history=True,
    )

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        continue

    total_return = result.get("total_return", 0.0)
    history = result.get("history", {})
    nw = np.array(history.get("net_worth", [10000.0]), dtype=np.float64)
    step_returns = np.diff(nw) / np.maximum(nw[:-1], 1e-12)
    n_steps = len(step_returns)
    mean_step = float(np.mean(step_returns))
    std_step = float(np.std(step_returns))

    print(f"  total_return={total_return:+.6f}  ({total_return*100:+.2f}%)")
    print(f"  step_returns: n={n_steps}, mean={mean_step:.6f}, std={std_step:.6f}")

    ci_lower, ci_upper = bootstrap_confidence_interval(
        step_returns, n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL, seed=42
    )
    ci_width = ci_upper - ci_lower
    step_significant = ci_lower > 0

    tr_lower, tr_upper, _ = compute_bootstrap_on_total_return(
        step_returns, n_bootstrap=N_BOOTSTRAP, ci=CI_LEVEL, seed=42
    )
    tr_width = tr_upper - tr_lower
    tr_significant = tr_lower > 0

    print(f"  Step-mean CI95%: [{ci_lower:.6f}, {ci_upper:.6f}]  "
          f"width={ci_width:.6f}  significant={step_significant}")
    print(f"  Total-ret CI95%: [{tr_lower:.4f}, {tr_upper:.4f}]  "
          f"width={tr_width:.4f}  significant={tr_significant}")

    all_rows.append({
        "config": config["name"],
        "tau": config["tau"],
        "gate_mode": config["gate_mode"],
        "disabled_experts": "|".join(config["disabled_experts"]) if config["disabled_experts"] else "",
        "total_return": total_return,
        "mean_step_return": mean_step,
        "step_std": std_step,
        "step_ci_lower": ci_lower,
        "step_ci_upper": ci_upper,
        "step_ci_width": ci_width,
        "step_significant": step_significant,
        "total_return_ci_lower": tr_lower,
        "total_return_ci_upper": tr_upper,
        "total_return_ci_width": tr_width,
        "total_return_significant": tr_significant,
        "n_steps": n_steps,
    })

out_df = pd.DataFrame(all_rows)
out_path = Path(OUTPUT_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(out_path, index=False)
print(f"\n{'=' * 72}")
print(f"Results saved to {OUTPUT_PATH}")
print(f"{'=' * 72}")
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 120)
pd.set_option('display.float_format', '{:+.6f}'.format)
print(out_df.to_string(index=False))

