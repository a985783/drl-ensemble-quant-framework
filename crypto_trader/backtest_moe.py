from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    from crypto_trader.asset_profile import get_asset_profile, infer_asset_key
    from crypto_trader.config import get_default_config, load_config
    from crypto_trader.envs.trading_env import TradingEnv
    from crypto_trader.risk_manager import build_risk_manager_from_config
    from crypto_trader.train_moe_stage2_gate import (
        GateExpertArtifact,
        build_gate_artifacts,
        softmax_weights,
        validate_stage1_artifacts,
    )
except ImportError:
    from asset_profile import get_asset_profile, infer_asset_key
    from config import get_default_config, load_config
    from envs.trading_env import TradingEnv
    from risk_manager import build_risk_manager_from_config
    from train_moe_stage2_gate import (
        GateExpertArtifact,
        build_gate_artifacts,
        softmax_weights,
        validate_stage1_artifacts,
    )


@dataclass
class ExpertRuntime:
    artifact: GateExpertArtifact
    model: object
    vecnorm: object


def _env_kwargs_for_symbol(symbol: str, interval: str = "1d") -> Dict[str, float]:
    cfg = get_asset_profile(symbol, interval=interval).env
    return {
        "symbol": symbol,
        "atr_floor": cfg.atr_floor,
        "vol_scale_min": cfg.vol_scale_min,
        "vol_scale_max": cfg.vol_scale_max,
        "target_atr_pct": cfg.target_atr_pct,
        "tau": cfg.tau,
        "delta_max": cfg.delta_max,
        "cooldown_n": cfg.cooldown_n,
        "k_single": cfg.k_single,
        "funding_daily": cfg.funding_daily,
    }


def load_stage2_usage(stage2_dir: Path, expert_ids: List[str]) -> Dict[str, float]:
    metadata_path = Path(stage2_dir) / "metadata.json"
    if not metadata_path.exists():
        uniform = 1.0 / float(len(expert_ids))
        return {eid: uniform for eid in expert_ids}

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    ids = metadata.get("expert_ids", expert_ids)
    usage = metadata.get("usage_ema", [1.0 / float(len(expert_ids))] * len(expert_ids))

    out = {eid: 0.0 for eid in expert_ids}
    if isinstance(usage, dict):
        for eid, value in usage.items():
            if eid in out:
                out[eid] = float(value)
    else:
        for eid, value in zip(ids, usage):
            if eid in out:
                out[eid] = float(value)

    total = float(sum(out.values()))
    if total <= 0:
        uniform = 1.0 / float(len(expert_ids))
        return {eid: uniform for eid in expert_ids}

    return {k: float(v / total) for k, v in out.items()}


def aggregate_usage(weights_history: np.ndarray, expert_ids: List[str]) -> Dict[str, float]:
    if weights_history.ndim != 2:
        raise ValueError("weights_history must be 2D")
    if weights_history.shape[1] != len(expert_ids):
        raise ValueError("weights_history shape mismatch with expert_ids")

    mean_weights = np.mean(weights_history, axis=0)
    total = float(np.sum(mean_weights))
    if total <= 0:
        mean_weights = np.ones_like(mean_weights) / float(len(mean_weights))
    else:
        mean_weights = mean_weights / total
    return {eid: float(w) for eid, w in zip(expert_ids, mean_weights)}


def estimate_expert_contribution(
    weights_history: np.ndarray,
    expert_actions_history: np.ndarray,
    step_returns: np.ndarray,
    expert_ids: List[str],
) -> Dict[str, float]:
    if weights_history.shape != expert_actions_history.shape:
        raise ValueError("weights_history and expert_actions_history must have same shape")
    if weights_history.shape[0] != len(step_returns):
        raise ValueError("step_returns length must equal number of timesteps")
    if weights_history.shape[1] != len(expert_ids):
        raise ValueError("expert_ids length mismatch")

    # Signed contribution proxy: weight * expert_action * realized step return
    contrib_matrix = weights_history * expert_actions_history * step_returns.reshape(-1, 1)
    contrib = np.sum(contrib_matrix, axis=0)
    return {eid: float(v) for eid, v in zip(expert_ids, contrib)}


def _mask_obs(obs: np.ndarray, feature_mask: Optional[List[int]]) -> np.ndarray:
    if feature_mask is None:
        return obs
    out = np.zeros_like(obs)
    out[feature_mask] = obs[feature_mask]
    return out


def apply_data_transform(df: pd.DataFrame, data_transform: Optional[str]) -> pd.DataFrame:
    if not data_transform:
        return df
    out = df.copy()
    if data_transform == "signal_delay_1d":
        if "Signal_Proba" not in out.columns:
            raise ValueError("Signal_Proba column is required for signal_delay_1d")
        out["Signal_Proba"] = out["Signal_Proba"].shift(1).fillna(out["Signal_Proba"])
        return out
    if data_transform == "signal_neutral_0_5":
        if "Signal_Proba" not in out.columns:
            raise ValueError("Signal_Proba column is required for signal_neutral_0_5")
        out["Signal_Proba"] = 0.5
        return out
    raise ValueError(f"Unsupported data_transform: {data_transform}")


def resolve_execution_frame(df: pd.DataFrame, execution_mode: str = "next_bar") -> tuple[pd.DataFrame, Dict[str, object]]:
    mode = (execution_mode or "next_bar").strip().lower()
    if mode in {"same_bar", "legacy_same_bar"}:
        return df.copy(), {"execution_mode": mode, "dropped_rows": 0}
    if mode != "next_bar":
        raise ValueError(f"Unsupported execution_mode: {execution_mode}")
    if len(df) < 2:
        raise ValueError("next_bar execution requires at least 2 rows")

    shifted = df.copy()
    for col in shifted.columns:
        shifted[col] = shifted[col].shift(-1)
    shifted = shifted.iloc[:-1].copy()
    return shifted, {"execution_mode": "next_bar", "dropped_rows": 1}


def resolve_gate_weights(
    logits: np.ndarray,
    expert_actions: np.ndarray,
    gate_mode: str = "model",
    temperature: float = 1.0,
) -> np.ndarray:
    mode = (gate_mode or "model").strip().lower()
    if mode == "model":
        return softmax_weights(logits, temperature=temperature)
    if mode in {"uniform", "average_experts"}:
        if len(expert_actions) == 0:
            return np.asarray([], dtype=np.float32)
        return np.ones(len(expert_actions), dtype=np.float32) / float(len(expert_actions))
    raise ValueError(f"Unsupported gate_mode: {gate_mode}")


def _apply_disabled_experts(
    weights: np.ndarray,
    expert_ids: List[str],
    disabled_experts: Optional[List[str]],
) -> np.ndarray:
    if not disabled_experts:
        return weights
    disabled = {str(eid) for eid in disabled_experts}
    adjusted = np.asarray(weights, dtype=np.float32).copy()
    for idx, expert_id in enumerate(expert_ids):
        if expert_id in disabled:
            adjusted[idx] = 0.0
    total = float(np.sum(adjusted))
    if total <= 0:
        return np.ones(len(expert_ids), dtype=np.float32) / float(len(expert_ids))
    return adjusted / total


def _load_expert_runtimes(
    artifacts: List[GateExpertArtifact],
    df: pd.DataFrame,
    config,
    symbol: str,
    interval: str,
    enable_kill_switch: bool = False,
    env_overrides: Optional[Dict[str, float]] = None,
) -> List[ExpertRuntime]:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    try:
        from crypto_trader.train_moe_stage1 import get_algo_registry
    except ImportError:
        from train_moe_stage1 import get_algo_registry

    runtimes: List[ExpertRuntime] = []
    algo_registry = get_algo_registry(load_classes=True)

    for artifact in artifacts:
        algo_cls = algo_registry[artifact.algorithm]
        model = algo_cls.load(str(artifact.model_path))
        env_kwargs = _env_kwargs_for_symbol(symbol, interval=interval)

        rm = build_risk_manager_from_config(config)

        def make_env(a: GateExpertArtifact = artifact):
            kwargs = dict(env_kwargs)
            if env_overrides:
                kwargs.update(env_overrides)
            return TradingEnv(
                df,
                risk_manager=rm,
                **kwargs,
                feature_mask=a.feature_mask,
                reward_profile=a.reward_profile,
                enable_kill_switch=enable_kill_switch,
            )

        temp_vec = DummyVecEnv([make_env])
        vecnorm = VecNormalize.load(str(artifact.vecnorm_path), temp_vec)
        vecnorm.training = False
        vecnorm.norm_reward = False

        runtimes.append(ExpertRuntime(artifact=artifact, model=model, vecnorm=vecnorm))

    return runtimes


def backtest_moe(
    manifest_path: Path,
    stage1_root: str = "checkpoints/moe/stable/experts",
    stage2_root: str = "checkpoints/moe/stable/gate",
    data_path: str = "crypto_trader/data_moe_20200101_20260216_oos20.csv",
    max_steps: Optional[int] = None,
    plot_path: str = "results/net_worth_ETH_moe.png",
    gate_temperature: float = 1.0,
    symbol: Optional[str] = None,
    config_path: Optional[str] = None,
    enable_kill_switch: bool = False,  # 日内止损：策略不使用，保持关闭
    env_overrides: Optional[Dict[str, float]] = None,
    gate_mode: str = "model",
    disabled_experts: Optional[List[str]] = None,
    data_transform: Optional[str] = None,
    return_history: bool = False,
    execution_mode: str = "next_bar",
    active_expert_ids: Optional[List[str]] = None,
) -> Dict[str, object]:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    test_path = Path(data_path)
    if not test_path.exists():
        return {"error": f"test data not found: {data_path}"}

    test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)
    try:
        test_df = apply_data_transform(test_df, data_transform)
        test_df, execution_metadata = resolve_execution_frame(test_df, execution_mode)
    except ValueError as exc:
        return {"error": str(exc)}
    artifacts = build_gate_artifacts(
        manifest_path=manifest_path,
        stage1_root=stage1_root,
        active_expert_ids=active_expert_ids,
    )

    missing = validate_stage1_artifacts(artifacts)
    stage2_dir = Path(stage2_root)
    gate_model_path = stage2_dir / "gate_model.zip"
    gate_vecnorm_path = stage2_dir / "gate_vec_normalize.pkl"

    if not gate_model_path.exists():
        missing.append(str(gate_model_path))
    if not gate_vecnorm_path.exists():
        missing.append(str(gate_vecnorm_path))

    if missing:
        return {"error": "missing artifacts", "missing": missing}

    config = load_config(config_path) if config_path else get_default_config()
    symbol_to_use = symbol or infer_asset_key(data_path, interval=config.data.interval)
    config.data.symbol = symbol_to_use
    env_kwargs = _env_kwargs_for_symbol(symbol_to_use, interval=config.data.interval)
    runtimes = _load_expert_runtimes(
        artifacts,
        test_df,
        config,
        symbol=symbol_to_use,
        interval=config.data.interval,
        enable_kill_switch=enable_kill_switch,
        env_overrides=env_overrides,
    )
    expert_ids = [r.artifact.expert_id for r in runtimes]

    gate_model = PPO.load(str(gate_model_path))

    rm_gate = build_risk_manager_from_config(config)

    def make_gate_env():
        kwargs = dict(env_kwargs)
        if env_overrides:
            kwargs.update(env_overrides)
        return TradingEnv(
            test_df,
            risk_manager=rm_gate,
            **kwargs,
            enable_kill_switch=enable_kill_switch,
        )

    gate_temp_vec = DummyVecEnv([make_gate_env])
    gate_vecnorm = VecNormalize.load(str(gate_vecnorm_path), gate_temp_vec)
    gate_vecnorm.training = False
    gate_vecnorm.norm_reward = False

    rm_main = build_risk_manager_from_config(config)

    def make_main_env():
        kwargs = dict(env_kwargs)
        if env_overrides:
            kwargs.update(env_overrides)
        return TradingEnv(
            test_df,
            risk_manager=rm_main,
            **kwargs,
            enable_kill_switch=enable_kill_switch,
        )

    main_env = DummyVecEnv([make_main_env])
    obs = main_env.reset()

    n_steps = min(len(test_df) - 1, max_steps) if max_steps else (len(test_df) - 1)

    net_worths = [10000.0]
    mixed_actions: List[float] = []
    weights_hist: List[np.ndarray] = []
    expert_actions_hist: List[np.ndarray] = []
    step_returns: List[float] = []
    positions: List[float] = []
    turnovers: List[float] = []
    trade_costs: List[float] = []
    funding_costs: List[float] = []

    for _ in range(n_steps):
        gate_obs = gate_vecnorm.normalize_obs(obs)
        logits, _ = gate_model.predict(gate_obs, deterministic=True)
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)

        if logits.size != len(runtimes):
            return {
                "error": f"gate output dim mismatch: expected {len(runtimes)}, got {logits.size}",
            }

        raw_obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)[0]
        expert_actions = []
        for runtime in runtimes:
            masked = _mask_obs(raw_obs, runtime.artifact.feature_mask)
            norm_obs = runtime.vecnorm.normalize_obs(masked.reshape(1, -1))
            action, _ = runtime.model.predict(norm_obs, deterministic=True)
            expert_actions.append(float(np.asarray(action).reshape(-1)[0]))

        expert_actions_arr = np.asarray(expert_actions, dtype=np.float32)
        try:
            weights = resolve_gate_weights(
                logits=logits,
                expert_actions=expert_actions_arr,
                gate_mode=gate_mode,
                temperature=gate_temperature,
            )
        except ValueError as exc:
            return {"error": str(exc)}
        weights = _apply_disabled_experts(weights, expert_ids, disabled_experts)
        mixed_action = float(np.clip(np.dot(weights, expert_actions_arr), -1.0, 1.0))

        obs, _, _, info = main_env.step(np.array([[mixed_action]], dtype=np.float32))
        info0 = info[0]

        net_worths.append(float(info0["net_worth"]))
        mixed_actions.append(mixed_action)
        weights_hist.append(weights)
        expert_actions_hist.append(expert_actions_arr)
        step_returns.append(float(info0.get("step_return", 0.0)))
        positions.append(float(info0.get("position", 0.0)))
        turnovers.append(float(info0.get("turnover", 0.0)))
        trade_costs.append(float(info0.get("trade_cost", 0.0)))
        funding_costs.append(float(info0.get("funding_cost", 0.0)))

    net_worth_arr = np.asarray(net_worths, dtype=np.float64)
    peak = np.maximum.accumulate(net_worth_arr)
    max_dd = float(((peak - net_worth_arr) / peak).max())
    total_return = float((net_worth_arr[-1] - 10000.0) / 10000.0)

    bench = (test_df["Close"] / test_df["Close"].iloc[0]) * 10000.0
    bench_ret = float((bench.iloc[min(len(bench) - 1, len(net_worth_arr) - 1)] - 10000.0) / 10000.0)

    if weights_hist:
        weights_np = np.vstack(weights_hist)
        actions_np = np.vstack(expert_actions_hist)
        returns_np = np.asarray(step_returns, dtype=np.float32)
        gate_usage = aggregate_usage(weights_np, expert_ids)
        contributions = estimate_expert_contribution(weights_np, actions_np, returns_np, expert_ids)
    else:
        gate_usage = {eid: 0.0 for eid in expert_ids}
        contributions = {eid: 0.0 for eid in expert_ids}

    if plt is not None:
        out_plot = Path(plot_path)
        out_plot.parent.mkdir(parents=True, exist_ok=True)
        x = test_df.index[: len(net_worth_arr)]
        plt.figure(figsize=(14, 8))
        plt.plot(x, net_worth_arr, label=f"MoE (+{total_return*100:.1f}%)", color="teal")
        plt.plot(test_df.index, bench.values, label="Benchmark", color="gray", linestyle="--")
        plt.title("MoE Backtest")
        plt.ylabel("Value (USD)")
        plt.legend()
        plt.grid(True)
        plt.savefig(out_plot)
        plt.close()

    stage2_prior_usage = load_stage2_usage(Path(stage2_root), expert_ids)

    result = {
        "total_return": total_return,
        "benchmark_return": bench_ret,
        "alpha": total_return - bench_ret,
        "max_dd": max_dd,
        "final_net_worth": float(net_worth_arr[-1]),
        "first_actions": [float(a) for a in mixed_actions[:10]],
        "gate_usage": gate_usage,
        "stage2_prior_usage": stage2_prior_usage,
        "expert_contribution": contributions,
        "plot_path": plot_path,
        "symbol": symbol_to_use,
        "interval": config.data.interval,
        "execution": execution_metadata,
    }
    if return_history:
        result["history"] = {
            "net_worth": [float(v) for v in net_worth_arr],
            "benchmark_values": [float(v) for v in bench.values[: len(net_worth_arr)]],
            "positions": positions,
            "turnovers": turnovers,
            "trade_costs": trade_costs,
            "funding_costs": funding_costs,
            "mixed_actions": [float(a) for a in mixed_actions],
            "weights": weights_np.tolist() if weights_hist else [],
            "expert_actions": actions_np.tolist() if weights_hist else [],
        }
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest MoE gate + experts")
    parser.add_argument("--manifest", type=str, default="crypto_trader/configs/moe_experts.yaml")
    parser.add_argument("--stage1-root", type=str, default="checkpoints/moe/stable/experts")
    parser.add_argument("--stage2-root", type=str, default="checkpoints/moe/stable/gate")
    parser.add_argument("--data-path", type=str, default="crypto_trader/data_moe_20200101_20260216_oos20.csv")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--plot-path", type=str, default="results/net_worth_ETH_moe.png")
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--symbol", type=str, default=None, help="Trading symbol for asset profile, e.g. ETH/USDT:USDT")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    parser.add_argument("--no-kill-switch", action="store_true",
                        help="禁用日内止损 kill switch（默认开启以与实盘保持一致）")
    parser.add_argument("--gate-mode", type=str, default="model", choices=["model", "uniform", "average_experts"])
    parser.add_argument("--data-transform", type=str, default=None, choices=["signal_delay_1d", "signal_neutral_0_5"])
    parser.add_argument("--execution-mode", type=str, default="next_bar", choices=["next_bar", "same_bar"])
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = backtest_moe(
        manifest_path=Path(args.manifest),
        stage1_root=args.stage1_root,
        stage2_root=args.stage2_root,
        data_path=args.data_path,
        max_steps=args.max_steps,
        plot_path=args.plot_path,
        gate_temperature=args.gate_temperature,
        symbol=args.symbol,
        config_path=args.config,
        enable_kill_switch=not args.no_kill_switch,
        gate_mode=args.gate_mode,
        data_transform=args.data_transform,
        execution_mode=args.execution_mode,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
