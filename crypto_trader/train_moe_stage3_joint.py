from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from crypto_trader.asset_profile import get_asset_profile
    from crypto_trader.config import get_default_config, load_config
    from crypto_trader.moe.regime import select_market_slice
    from crypto_trader.train_moe_stage2_gate import (
        GateExpertArtifact,
        build_gate_artifacts,
        run_stage2,
        validate_stage1_artifacts,
    )
except ImportError:
    from asset_profile import get_asset_profile
    from config import get_default_config, load_config
    from moe.regime import select_market_slice
    from train_moe_stage2_gate import (
        GateExpertArtifact,
        build_gate_artifacts,
        run_stage2,
        validate_stage1_artifacts,
    )


def allocate_expert_timesteps(
    usage: Dict[str, float],
    base_timesteps: int,
    min_ratio: float = 0.5,
    max_ratio: float = 1.5,
) -> Dict[str, int]:
    if base_timesteps <= 0:
        raise ValueError("base_timesteps must be positive")
    if not usage:
        raise ValueError("usage must not be empty")

    keys = list(usage.keys())
    values = np.asarray([max(float(usage[k]), 0.0) for k in keys], dtype=np.float32)
    total = float(values.sum())
    if total <= 0:
        values = np.ones_like(values) / float(len(values))
    else:
        values = values / total

    mean_usage = float(1.0 / len(keys))
    out: Dict[str, int] = {}

    for key, u in zip(keys, values):
        ratio = 1.0 + 0.8 * ((float(u) - mean_usage) / max(mean_usage, 1e-8))
        ratio = float(np.clip(ratio, min_ratio, max_ratio))
        out[key] = int(round(base_timesteps * ratio))

    return out


def _load_usage_from_stage2(stage2_dir: Path, expert_ids: List[str]) -> Dict[str, float]:
    metadata_path = stage2_dir / "metadata.json"
    if not metadata_path.exists():
        uniform = 1.0 / float(len(expert_ids))
        return {eid: uniform for eid in expert_ids}

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    ids = metadata.get("expert_ids", expert_ids)
    usage = metadata.get("usage_ema", [1.0 / float(len(expert_ids))] * len(expert_ids))

    out: Dict[str, float] = {eid: 0.0 for eid in expert_ids}
    for eid, w in zip(ids, usage):
        if eid in out:
            out[eid] = float(w)

    if sum(out.values()) <= 0:
        uniform = 1.0 / float(len(expert_ids))
        out = {eid: uniform for eid in expert_ids}

    return out


def _finetune_one_expert(
    artifact: GateExpertArtifact,
    base_df,
    timesteps: int,
    output_root: Path,
    symbol: str,
    config,
) -> GateExpertArtifact:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    try:
        from crypto_trader.envs.trading_env import TradingEnv
        from crypto_trader.risk_manager import RiskManager
        from crypto_trader.train_moe_stage1 import get_algo_registry
    except ImportError:
        from envs.trading_env import TradingEnv
        from risk_manager import RiskManager
        from train_moe_stage1 import get_algo_registry

    env_cfg = get_asset_profile(symbol, interval=config.data.interval).env
    algo_cls = get_algo_registry(load_classes=True)[artifact.algorithm]
    expert_df = select_market_slice(base_df, artifact.data_slice)

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

    def make_env(a: GateExpertArtifact = artifact):
        return TradingEnv(
            expert_df,
            risk_manager=rm,
            symbol=symbol,
            atr_floor=env_cfg.atr_floor,
            vol_scale_min=env_cfg.vol_scale_min,
            vol_scale_max=env_cfg.vol_scale_max,
            target_atr_pct=env_cfg.target_atr_pct,
            tau=env_cfg.tau,
            delta_max=env_cfg.delta_max,
            cooldown_n=env_cfg.cooldown_n,
            k_single=env_cfg.k_single,
            funding_daily=env_cfg.funding_daily,
            feature_mask=a.feature_mask,
            reward_profile=a.reward_profile,
        )

    temp_vec = DummyVecEnv([make_env])
    vecnorm = VecNormalize.load(str(artifact.vecnorm_path), temp_vec)
    vecnorm.training = True
    vecnorm.norm_reward = True

    model = algo_cls.load(str(artifact.model_path), env=vecnorm)
    model.learn(total_timesteps=int(timesteps), reset_num_timesteps=False)

    out_dir = output_root / artifact.expert_id
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(out_dir / "model"))
    vecnorm.save(str(out_dir / "vec_normalize.pkl"))

    return GateExpertArtifact(
        expert_id=artifact.expert_id,
        algorithm=artifact.algorithm,
        seed=artifact.seed,
        data_slice=artifact.data_slice,
        feature_mask=artifact.feature_mask,
        reward_profile=artifact.reward_profile,
        model_path=out_dir / "model.zip",
        vecnorm_path=out_dir / "vec_normalize.pkl",
    )


def run_stage3(
    manifest_path: Path,
    stage1_root: str = "checkpoints/moe/stage1",
    stage2_root: str = "checkpoints/moe/stage2",
    output_root: str = "checkpoints/moe/stage3",
    rounds: int = 2,
    base_expert_timesteps: int = 60_000,
    gate_timesteps: int = 80_000,
    load_balance_coef: float = 0.02,
    diversity_coef: float = 0.01,
    gate_temperature: float = 1.0,
    dry_run: bool = False,
    train_data_path: Optional[str] = None,
    symbol: Optional[str] = None,
    config_path: Optional[str] = None,
) -> None:
    try:
        from crypto_trader.train_moe_stage1 import _prepare_training_dataframe
    except ImportError:
        from train_moe_stage1 import _prepare_training_dataframe

    if rounds <= 0:
        raise ValueError("rounds must be positive")

    artifacts = build_gate_artifacts(manifest_path=manifest_path, stage1_root=stage1_root)
    missing = validate_stage1_artifacts(artifacts)

    if dry_run:
        print("MoE Stage3 dry-run plan:")
        print(
            "rounds="
            f"{rounds}, base_expert_timesteps={base_expert_timesteps}, gate_timesteps={gate_timesteps}, "
            f"load_balance_coef={load_balance_coef}, diversity_coef={diversity_coef}, "
            f"gate_temperature={gate_temperature}"
        )
        for a in artifacts:
            print(a.expert_id, a.algorithm, a.model_path, a.vecnorm_path)
        if missing:
            print("Missing artifacts:")
            for m in missing:
                print(m)
        return

    if missing:
        raise FileNotFoundError("Stage1 artifacts missing:\n" + "\n".join(missing))

    config = load_config(config_path) if config_path else get_default_config()
    if symbol:
        config.data.symbol = symbol
    symbol_to_use = config.data.symbol
    base_df = _prepare_training_dataframe(
        config,
        train_data_path=train_data_path,
        symbol_override=symbol_to_use,
    )

    current_artifacts = artifacts
    current_stage2_dir = Path(stage2_root)
    out_root = Path(output_root)

    for r in range(1, rounds + 1):
        round_dir = out_root / f"round{r}"
        experts_out = round_dir / "experts"
        gate_out = round_dir / "gate"
        experts_out.mkdir(parents=True, exist_ok=True)

        usage = _load_usage_from_stage2(current_stage2_dir, [a.expert_id for a in current_artifacts])
        allocation = allocate_expert_timesteps(usage, base_timesteps=base_expert_timesteps)

        updated: List[GateExpertArtifact] = []
        for artifact in current_artifacts:
            ts = allocation.get(artifact.expert_id, base_expert_timesteps)
            print(f"[Stage3][Round {r}] Fine-tune {artifact.expert_id} for {ts} steps")
            updated_artifact = _finetune_one_expert(
                artifact,
                base_df,
                ts,
                experts_out,
                symbol=symbol_to_use,
                config=config,
            )
            updated.append(updated_artifact)

        # Re-train gate on updated experts (frozen-expert routing stage)
        run_stage2(
            manifest_path=manifest_path,
            stage1_root=str(experts_out),
            output_dir=str(gate_out),
            dry_run=False,
            total_timesteps=gate_timesteps,
            load_balance_coef=load_balance_coef,
            diversity_coef=diversity_coef,
            gate_temperature=gate_temperature,
            train_data_path=train_data_path,
            symbol=symbol_to_use,
            config_path=config_path,
        )

        with open(round_dir / "joint_metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "round": r,
                    "usage": usage,
                    "allocation": allocation,
                    "experts_root": str(experts_out),
                    "gate_root": str(gate_out),
                    "load_balance_coef": load_balance_coef,
                    "diversity_coef": diversity_coef,
                    "gate_temperature": gate_temperature,
                    "symbol": symbol_to_use,
                    "interval": config.data.interval,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        current_artifacts = updated
        current_stage2_dir = gate_out


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage3 alternating joint fine-tuning for MoE")
    parser.add_argument("--manifest", type=str, default="crypto_trader/configs/moe_experts.yaml")
    parser.add_argument("--stage1-root", type=str, default="checkpoints/moe/stage1")
    parser.add_argument("--stage2-root", type=str, default="checkpoints/moe/stage2")
    parser.add_argument("--output-root", type=str, default="checkpoints/moe/stage3")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--base-expert-timesteps", type=int, default=60000)
    parser.add_argument("--gate-timesteps", type=int, default=80000)
    parser.add_argument("--load-balance-coef", type=float, default=0.02)
    parser.add_argument("--diversity-coef", type=float, default=0.01)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train-data-path", type=str, default=None)
    parser.add_argument("--symbol", type=str, default=None, help="Trading symbol for asset profile, e.g. ETH/USDT:USDT")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_stage3(
        manifest_path=Path(args.manifest),
        stage1_root=args.stage1_root,
        stage2_root=args.stage2_root,
        output_root=args.output_root,
        rounds=args.rounds,
        base_expert_timesteps=args.base_expert_timesteps,
        gate_timesteps=args.gate_timesteps,
        load_balance_coef=args.load_balance_coef,
        diversity_coef=args.diversity_coef,
        gate_temperature=args.gate_temperature,
        dry_run=args.dry_run,
        train_data_path=args.train_data_path,
        symbol=args.symbol,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
