from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from crypto_trader.asset_profile import get_asset_profile
    from crypto_trader.config import BaseConfig, get_default_config, load_config
    from crypto_trader.moe.manifest import load_expert_manifest
    from crypto_trader.moe.regime import select_market_slice
except ImportError:
    from asset_profile import get_asset_profile
    from config import BaseConfig, get_default_config, load_config
    from moe.manifest import load_expert_manifest
    from moe.regime import select_market_slice


@dataclass
class TrainingSpec:
    expert_id: str
    algorithm: str
    seed: int
    data_slice: str
    feature_mask: Optional[List[int]]
    reward_profile: Dict[str, float]
    timesteps: int
    output_dir: str


def get_algo_registry(load_classes: bool = False):
    registry = {
        "ppo": None,
        "a2c": None,
        "sac": None,
    }
    if not load_classes:
        return registry

    from stable_baselines3 import A2C, PPO, SAC

    return {
        "ppo": PPO,
        "a2c": A2C,
        "sac": SAC,
    }


def build_training_specs(manifest_path: Path, output_root: str = "checkpoints/moe/stage1") -> List[TrainingSpec]:
    manifest = load_expert_manifest(manifest_path)
    specs: List[TrainingSpec] = []
    for e in manifest.experts:
        specs.append(
            TrainingSpec(
                expert_id=e.expert_id,
                algorithm=e.algorithm,
                seed=e.seed,
                data_slice=e.data_slice,
                feature_mask=e.feature_mask,
                reward_profile=e.reward_profile or {},
                timesteps=e.timesteps,
                output_dir=str(Path(output_root) / e.expert_id),
            )
        )
    return specs


def _prepare_training_dataframe(
    config: BaseConfig,
    train_data_path: Optional[str] = None,
    symbol_override: Optional[str] = None,
) -> pd.DataFrame:
    if train_data_path:
        local_path = Path(train_data_path)
        if not local_path.exists():
            raise FileNotFoundError(f"train_data_path not found: {train_data_path}")
        df = pd.read_csv(local_path)
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            df = df.set_index("Timestamp")
        elif "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.set_index("Date")

        if "Signal_Proba" not in df.columns:
            try:
                from crypto_trader.models.signal_model import SignalPredictor
            except ImportError:
                from models.signal_model import SignalPredictor
            predictor = SignalPredictor()
            predictor.train(df)
            df["Signal_Proba"] = predictor.predict_proba(df)

        split_idx = int(len(df) * config.data.train_split_ratio)
        return df.iloc[:split_idx].copy()

    try:
        from crypto_trader.data_loader import DataLoader
        from crypto_trader.features import FeatureEngineer
        from crypto_trader.models.signal_model import SignalPredictor
    except ImportError:
        from data_loader import DataLoader
        from features import FeatureEngineer
        from models.signal_model import SignalPredictor

    loader = DataLoader()
    engineer = FeatureEngineer()

    symbol = symbol_override or config.data.symbol
    start_date = config.data.train_start
    end_date = config.data.train_end

    raw_df = loader.fetch_data(start_date, end_date, symbol, interval=config.data.interval)
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)

    processed_df = engineer.add_technical_indicators(raw_df, symbol=symbol)
    split_idx = int(len(processed_df) * config.data.train_split_ratio)
    train_df = processed_df.iloc[:split_idx].copy()

    predictor = SignalPredictor()
    predictor.train(train_df)
    train_df["Signal_Proba"] = predictor.predict_proba(train_df)
    return train_df


def _algo_kwargs(algorithm: str, config: BaseConfig, seed: int):
    if algorithm == "ppo":
        return {
            "learning_rate": config.model.learning_rate,
            "gamma": config.model.gamma,
            "n_steps": config.model.n_steps,
            "batch_size": config.model.batch_size,
            "ent_coef": config.model.ent_coef,
            "clip_range": config.model.clip_range,
            "gae_lambda": config.model.gae_lambda,
            "seed": seed,
        }

    if algorithm == "a2c":
        return {
            "learning_rate": config.model.learning_rate,
            "gamma": config.model.gamma,
            "n_steps": min(64, max(8, config.model.n_steps // 16)),
            "gae_lambda": config.model.gae_lambda,
            "ent_coef": config.model.ent_coef,
            "seed": seed,
        }

    if algorithm == "sac":
        return {
            "learning_rate": config.model.learning_rate,
            "gamma": config.model.gamma,
            "batch_size": config.model.batch_size,
            "seed": seed,
            "train_freq": (1, "step"),
            "gradient_steps": 1,
        }

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _build_regime_mask(df: pd.DataFrame, slice_name: str) -> np.ndarray:
    if (slice_name or "full").strip().lower() == "full":
        return np.ones(len(df), dtype=np.float32)

    sliced = select_market_slice(df, slice_name)
    mask = df.index.isin(sliced.index).astype(np.float32)
    if float(mask.sum()) <= 0:
        return np.ones(len(df), dtype=np.float32)
    return mask


def _training_controls_for_expert(spec: TrainingSpec) -> Dict[str, float]:
    # Global training regularizers (applies to all experts):
    # 1) Full-sequence training with regime-weighted reward.
    # 2) Saturation and directional-bias penalties to avoid one-sided collapse.
    controls: Dict[str, float] = {
        "regime_main_reward_weight": 1.0,
        "regime_off_reward_weight": 0.12,
        "sat_penalty_coef": 0.08,
        "sat_threshold": 0.70,
        "directional_bias_coef": 0.03,
        "directional_bias_alpha": 0.02,
        "action_cap": 0.85,
        "funding_cost_multiplier": 1.0,
        "short_squeeze_threshold": -1.0,  # sentinel: disabled
        "short_squeeze_max_short": 0.25,
    }

    eid = spec.expert_id
    if eid == "E3_PPO_range_calmar":
        controls.update(
            {
                "action_cap": 0.50,
                "sat_penalty_coef": 0.14,
                "directional_bias_coef": 0.06,
                "regime_off_reward_weight": 0.08,
            }
        )
    elif eid == "E5_PPO_lowvol_carry":
        controls.update(
            {
                "action_cap": 0.35,
                "sat_penalty_coef": 0.10,
                "directional_bias_coef": 0.05,
                "funding_cost_multiplier": 2.5,
                "regime_off_reward_weight": 0.06,
            }
        )
    elif eid == "E2_PPO_bear_drawdown":
        controls.update(
            {
                "action_cap": 0.70,
                "sat_penalty_coef": 0.09,
                "directional_bias_coef": 0.05,
                "short_squeeze_threshold": 0.06,
                "short_squeeze_max_short": 0.25,
            }
        )

    return controls


def _train_one_expert(spec: TrainingSpec, base_df: pd.DataFrame, config: BaseConfig) -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    try:
        from crypto_trader.envs.trading_env import TradingEnv
        from crypto_trader.risk_manager import RiskManager
    except ImportError:
        from envs.trading_env import TradingEnv
        from risk_manager import RiskManager

    algo_cls = get_algo_registry(load_classes=True)[spec.algorithm]
    env_cfg = get_asset_profile(config.data.symbol, interval=config.data.interval).env

    if len(base_df) < 64:
        raise ValueError(f"Training dataframe too small: {len(base_df)}")

    regime_mask = _build_regime_mask(base_df, spec.data_slice)
    controls = _training_controls_for_expert(spec)
    short_squeeze_threshold = controls["short_squeeze_threshold"]
    if short_squeeze_threshold < 0:
        short_squeeze_threshold = None

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

    def make_env():
        return TradingEnv(
            base_df,
            risk_manager=rm,
            atr_floor=env_cfg.atr_floor,
            vol_scale_min=env_cfg.vol_scale_min,
            vol_scale_max=env_cfg.vol_scale_max,
            symbol=config.data.symbol,
            target_atr_pct=env_cfg.target_atr_pct,
            tau=env_cfg.tau,
            delta_max=env_cfg.delta_max,
            cooldown_n=env_cfg.cooldown_n,
            k_single=env_cfg.k_single,
            funding_daily=env_cfg.funding_daily,
            feature_mask=spec.feature_mask,
            reward_profile=spec.reward_profile,
            regime_mask=regime_mask,
            regime_main_reward_weight=controls["regime_main_reward_weight"],
            regime_off_reward_weight=controls["regime_off_reward_weight"],
            sat_penalty_coef=controls["sat_penalty_coef"],
            sat_threshold=controls["sat_threshold"],
            directional_bias_coef=controls["directional_bias_coef"],
            directional_bias_alpha=controls["directional_bias_alpha"],
            action_cap=controls["action_cap"],
            funding_cost_multiplier=controls["funding_cost_multiplier"],
            short_squeeze_threshold=short_squeeze_threshold,
            short_squeeze_max_short=controls["short_squeeze_max_short"],
        )

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    model = algo_cls("MlpPolicy", vec_env, verbose=0, **_algo_kwargs(spec.algorithm, config, spec.seed))
    model.learn(total_timesteps=spec.timesteps)

    out_dir = Path(spec.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir / "model"))
    vec_env.save(str(out_dir / "vec_normalize.pkl"))

    with open(out_dir / "spec.json", "w", encoding="utf-8") as f:
        payload = asdict(spec)
        payload["training_controls"] = controls
        payload["regime_coverage"] = float(np.mean(regime_mask > 0.5))
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_stage1(
    manifest_path: Path,
    dry_run: bool = False,
    timesteps_override: Optional[int] = None,
    output_root: str = "checkpoints/moe/stage1",
    train_data_path: Optional[str] = None,
    symbol: Optional[str] = None,
    config_path: Optional[str] = None,
) -> List[TrainingSpec]:
    config = load_config(config_path) if config_path else get_default_config()
    if symbol:
        config.data.symbol = symbol
    specs = build_training_specs(manifest_path, output_root=output_root)

    if timesteps_override is not None:
        for spec in specs:
            spec.timesteps = int(timesteps_override)

    if dry_run:
        print("MoE Stage1 dry-run plan:")
        for spec in specs:
            print(json.dumps(asdict(spec), ensure_ascii=False))
        return specs

    base_df = _prepare_training_dataframe(
        config,
        train_data_path=train_data_path,
        symbol_override=symbol,
    )

    for spec in specs:
        np.random.seed(spec.seed)
        print(f"[Stage1] Training {spec.expert_id} ({spec.algorithm}, slice={spec.data_slice})")
        _train_one_expert(spec, base_df, config)

    return specs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train diverse experts for MoE stage1")
    parser.add_argument(
        "--manifest",
        type=str,
        default="crypto_trader/configs/moe_experts.yaml",
        help="Path to expert manifest YAML",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print resolved specs")
    parser.add_argument("--timesteps-override", type=int, default=None, help="Override all expert timesteps")
    parser.add_argument("--output-root", type=str, default="checkpoints/moe/stage1", help="Output root directory")
    parser.add_argument("--train-data-path", type=str, default=None, help="Local CSV path for training data")
    parser.add_argument("--symbol", type=str, default=None, help="Trading symbol for asset profile, e.g. ETH/USDT:USDT")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    run_stage1(
        manifest_path=Path(args.manifest),
        dry_run=args.dry_run,
        timesteps_override=args.timesteps_override,
        output_root=args.output_root,
        train_data_path=args.train_data_path,
        symbol=args.symbol,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
