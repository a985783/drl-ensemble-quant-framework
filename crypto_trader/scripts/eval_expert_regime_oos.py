from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "crypto_trader") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "crypto_trader"))

try:
    from crypto_trader.asset_profile import infer_asset_key
    from crypto_trader.backtest_moe import _env_kwargs_for_symbol, _load_expert_runtimes, _mask_obs
    from crypto_trader.config import get_default_config
    from crypto_trader.envs.trading_env import TradingEnv
    from crypto_trader.moe.manifest import load_expert_manifest
    from crypto_trader.moe.regime import select_market_slice
    from crypto_trader.risk_manager import build_risk_manager_from_config
    from crypto_trader.train_moe_stage2_gate import build_gate_artifacts
except ImportError:
    from asset_profile import infer_asset_key
    from backtest_moe import _env_kwargs_for_symbol, _load_expert_runtimes, _mask_obs
    from config import get_default_config
    from envs.trading_env import TradingEnv
    from moe.manifest import load_expert_manifest
    from moe.regime import select_market_slice
    from risk_manager import build_risk_manager_from_config
    from train_moe_stage2_gate import build_gate_artifacts


def _evaluate_single_expert(runtime, eval_df: pd.DataFrame, env_kwargs: Dict[str, float], cfg, no_cost: bool) -> Dict[str, float]:
    local_df = eval_df.copy()
    kwargs = dict(env_kwargs)
    if no_cost:
        kwargs["k_single"] = 0.0
        kwargs["funding_daily"] = 0.0
        if "Funding_Rate" in local_df.columns:
            local_df["Funding_Rate"] = 0.0

    rm = build_risk_manager_from_config(cfg)

    def make_env():
        return TradingEnv(local_df, risk_manager=rm, enable_kill_switch=False, **kwargs)

    env = DummyVecEnv([make_env])
    obs = env.reset()

    net_worths: List[float] = [10000.0]
    step_returns: List[float] = []
    turnovers: List[float] = []
    trade_costs: List[float] = []
    funding_costs: List[float] = []

    for _ in range(max(0, len(local_df) - 1)):
        raw_obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)[0]
        masked_obs = _mask_obs(raw_obs, runtime.artifact.feature_mask)
        norm_obs = runtime.vecnorm.normalize_obs(masked_obs.reshape(1, -1))
        action, _ = runtime.model.predict(norm_obs, deterministic=True)
        action_scalar = float(np.asarray(action).reshape(-1)[0])

        obs, _, _, info = env.step(np.array([[action_scalar]], dtype=np.float32))
        info0 = info[0]

        net_worths.append(float(info0.get("net_worth", net_worths[-1])))
        step_returns.append(float(info0.get("step_return", 0.0)))
        turnovers.append(float(info0.get("turnover", 0.0)))
        trade_costs.append(float(info0.get("trade_cost", 0.0)))
        funding_costs.append(float(info0.get("funding_cost", 0.0)))

    nw = np.asarray(net_worths, dtype=np.float64)
    peak = np.maximum.accumulate(nw)
    max_dd = float(((peak - nw) / peak).max()) if nw.size else 0.0

    return {
        "rows": int(len(local_df)),
        "steps": int(max(0, len(local_df) - 1)),
        "start": str(local_df.index.min().date()) if len(local_df) else None,
        "end": str(local_df.index.max().date()) if len(local_df) else None,
        "total_return": float((nw[-1] - 10000.0) / 10000.0) if nw.size else 0.0,
        "max_dd": max_dd,
        "win_rate": float(np.mean(np.asarray(step_returns, dtype=np.float64) > 0.0)) if step_returns else 0.0,
        "avg_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "total_turnover": float(np.sum(turnovers)) if turnovers else 0.0,
        "trade_cost_total": float(np.sum(trade_costs)) if trade_costs else 0.0,
        "funding_cost_total": float(np.sum(funding_costs)) if funding_costs else 0.0,
        "final_net_worth": float(nw[-1]) if nw.size else 10000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate each expert only on its own regime slice (OOS).")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--stage1-root", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--expert-ids", type=str, default=None, help="Comma-separated expert ids.")
    parser.add_argument("--min-return-after-cost", type=float, default=0.0)
    parser.add_argument("--max-dd", type=float, default=0.30)
    parser.add_argument("--min-win-rate", type=float, default=0.0)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df_all = pd.read_csv(data_path, index_col=0, parse_dates=True)
    cfg = get_default_config()
    symbol = args.symbol or infer_asset_key(str(data_path), interval=cfg.data.interval)
    cfg.data.symbol = symbol
    env_kwargs = _env_kwargs_for_symbol(symbol, interval=cfg.data.interval)

    manifest = load_expert_manifest(args.manifest)
    selected = {x.strip() for x in args.expert_ids.split(",")} if args.expert_ids else None
    target_specs = [e for e in manifest.experts if selected is None or e.expert_id in selected]
    if not target_specs:
        raise ValueError("No experts selected for evaluation.")

    artifacts = build_gate_artifacts(Path(args.manifest), args.stage1_root)
    runtimes = _load_expert_runtimes(
        artifacts=artifacts,
        df=df_all,
        config=cfg,
        symbol=symbol,
        interval=cfg.data.interval,
        enable_kill_switch=False,
    )
    runtime_by_id = {r.artifact.expert_id: r for r in runtimes}

    results = []
    for spec in target_specs:
        if spec.expert_id not in runtime_by_id:
            continue
        rt = runtime_by_id[spec.expert_id]
        df_slice = select_market_slice(df_all, spec.data_slice)
        net = _evaluate_single_expert(rt, df_slice, env_kwargs=env_kwargs, cfg=cfg, no_cost=False)
        gross = _evaluate_single_expert(rt, df_slice, env_kwargs=env_kwargs, cfg=cfg, no_cost=True)

        pass_flag = (
            net["total_return"] >= args.min_return_after_cost
            and net["max_dd"] <= args.max_dd
            and net["win_rate"] >= args.min_win_rate
        )

        results.append(
            {
                "expert_id": spec.expert_id,
                "algorithm": spec.algorithm,
                "data_slice": spec.data_slice,
                "return_after_cost": net["total_return"],
                "return_gross_no_cost": gross["total_return"],
                "cost_drag_return": float(gross["total_return"] - net["total_return"]),
                "max_dd": net["max_dd"],
                "win_rate": net["win_rate"],
                "avg_turnover": net["avg_turnover"],
                "total_turnover": net["total_turnover"],
                "trade_cost_usd": net["trade_cost_total"],
                "funding_cost_usd": net["funding_cost_total"],
                "cost_total_usd": float(net["trade_cost_total"] + net["funding_cost_total"]),
                "slice_rows": net["rows"],
                "slice_steps": net["steps"],
                "slice_period": f"{net['start']}~{net['end']}",
                "pass": bool(pass_flag),
            }
        )

    summary = {
        "data_path": str(data_path.resolve()),
        "data_start": str(df_all.index.min().date()),
        "data_end": str(df_all.index.max().date()),
        "rows": int(len(df_all)),
        "criteria": {
            "min_return_after_cost": float(args.min_return_after_cost),
            "max_dd": float(args.max_dd),
            "min_win_rate": float(args.min_win_rate),
        },
        "pass_count": int(sum(1 for x in results if x["pass"])),
        "expert_count": int(len(results)),
        "results": results,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
