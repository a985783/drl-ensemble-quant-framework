from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np

try:
    from crypto_trader.asset_profile import get_asset_profile
    from crypto_trader.config import BaseConfig, get_default_config, load_config
    from crypto_trader.moe.manifest import load_expert_manifest
except ImportError:
    from asset_profile import get_asset_profile
    from config import BaseConfig, get_default_config, load_config
    from moe.manifest import load_expert_manifest


@dataclass
class GateExpertArtifact:
    expert_id: str
    algorithm: str
    seed: int
    data_slice: str
    feature_mask: Optional[List[int]]
    reward_profile: Dict[str, float]
    model_path: Path
    vecnorm_path: Path


@dataclass
class LoadedExpert:
    artifact: GateExpertArtifact
    model: object
    vecnorm: object


def softmax_weights(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    x = np.asarray(logits, dtype=np.float32).reshape(-1)
    if x.size == 0:
        raise ValueError("logits must be non-empty")
    z = (x - np.max(x)) / float(temperature)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


def build_gate_artifacts(
    manifest_path: Path,
    stage1_root: str = "checkpoints/moe/stage1",
    active_expert_ids: Optional[List[str]] = None,
) -> List[GateExpertArtifact]:
    manifest = load_expert_manifest(manifest_path)
    root = Path(stage1_root)
    artifacts: List[GateExpertArtifact] = []
    active = set(active_expert_ids or [])

    for expert in manifest.experts:
        if active and expert.expert_id not in active:
            continue
        artifacts.append(
            GateExpertArtifact(
                expert_id=expert.expert_id,
                algorithm=expert.algorithm,
                seed=expert.seed,
                data_slice=expert.data_slice,
                feature_mask=expert.feature_mask,
                reward_profile=expert.reward_profile or {},
                model_path=root / expert.expert_id / "model.zip",
                vecnorm_path=root / expert.expert_id / "vec_normalize.pkl",
            )
        )
    return artifacts


def validate_stage1_artifacts(artifacts: List[GateExpertArtifact]) -> List[str]:
    missing: List[str] = []
    for a in artifacts:
        if not a.model_path.exists():
            missing.append(str(a.model_path))
        if not a.vecnorm_path.exists():
            missing.append(str(a.vecnorm_path))
    return missing


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


class GateRoutingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        experts: List[LoadedExpert],
        config: BaseConfig,
        load_balance_coef: float = 0.02,
        diversity_coef: float = 0.01,
        gate_temperature: float = 1.0,
    ):
        from gymnasium import spaces

        try:
            from crypto_trader.envs.trading_env import TradingEnv
            from crypto_trader.risk_manager import RiskManager
        except ImportError:
            from envs.trading_env import TradingEnv
            from risk_manager import RiskManager

        super().__init__()
        self.experts = experts
        self.k = len(experts)
        if self.k == 0:
            raise ValueError("No experts provided to GateRoutingEnv")

        self.load_balance_coef = float(load_balance_coef)
        self.diversity_coef = float(diversity_coef)
        self.gate_temperature = float(gate_temperature)

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
        env_kwargs = _env_kwargs_for_symbol(config.data.symbol, interval=config.data.interval)
        self.base_env = TradingEnv(
            df,
            risk_manager=rm,
            **env_kwargs,
        )

        self.observation_space = self.base_env.observation_space
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.k,), dtype=np.float32)

        self.usage_ema = np.ones(self.k, dtype=np.float32) / float(self.k)
        self._last_obs = None

    def _mask_obs(self, obs: np.ndarray, feature_mask: Optional[List[int]]) -> np.ndarray:
        if feature_mask is None:
            return obs
        out = np.zeros_like(obs)
        out[feature_mask] = obs[feature_mask]
        return out

    def _expert_action(self, expert: LoadedExpert, obs: np.ndarray) -> float:
        obs_i = self._mask_obs(obs, expert.artifact.feature_mask)
        norm_obs = expert.vecnorm.normalize_obs(obs_i.reshape(1, -1))
        action, _ = expert.model.predict(norm_obs, deterministic=True)
        return float(np.asarray(action).reshape(-1)[0])

    def reset(self, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        self.usage_ema = np.ones(self.k, dtype=np.float32) / float(self.k)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        return obs, info

    def step(self, action):
        logits = np.asarray(action, dtype=np.float32).reshape(-1)
        if logits.size != self.k:
            raise ValueError(f"Expected {self.k} gate logits, got {logits.size}")

        weights = softmax_weights(logits, temperature=self.gate_temperature)
        current_obs = self._last_obs if self._last_obs is not None else np.asarray(self.base_env._get_observation(), dtype=np.float32)
        expert_actions = np.asarray([self._expert_action(e, current_obs) for e in self.experts], dtype=np.float32)
        mixed_action = float(np.clip(np.dot(weights, expert_actions), -1.0, 1.0))

        obs, reward, terminated, truncated, info = self.base_env.step(np.array([mixed_action], dtype=np.float32))
        self._last_obs = np.asarray(obs, dtype=np.float32)

        self.usage_ema = 0.99 * self.usage_ema + 0.01 * weights
        target = 1.0 / float(self.k)
        balance_penalty = float(np.mean((self.usage_ema - target) ** 2))
        diversity_bonus = float(np.std(expert_actions))

        routed_reward = float(reward - self.load_balance_coef * balance_penalty + self.diversity_coef * diversity_bonus)

        info = dict(info)
        info.update(
            {
                "gate_weights": weights.tolist(),
                "expert_actions": expert_actions.tolist(),
                "mixed_action": mixed_action,
                "balance_penalty": balance_penalty,
                "diversity_bonus": diversity_bonus,
            }
        )
        return obs, routed_reward, terminated, truncated, info


def _load_experts(artifacts: List[GateExpertArtifact], df, config: BaseConfig) -> List[LoadedExpert]:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    try:
        from crypto_trader.envs.trading_env import TradingEnv
        from crypto_trader.risk_manager import RiskManager
        from crypto_trader.train_moe_stage1 import get_algo_registry
    except ImportError:
        from envs.trading_env import TradingEnv
        from risk_manager import RiskManager
        from train_moe_stage1 import get_algo_registry

    algo_registry = get_algo_registry(load_classes=True)
    experts: List[LoadedExpert] = []

    for artifact in artifacts:
        algo_cls = algo_registry[artifact.algorithm]
        model = algo_cls.load(str(artifact.model_path))
        env_kwargs = _env_kwargs_for_symbol(config.data.symbol, interval=config.data.interval)

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
                df,
                risk_manager=rm,
                **env_kwargs,
                feature_mask=a.feature_mask,
                reward_profile=a.reward_profile,
            )

        temp_vec = DummyVecEnv([make_env])
        vecnorm = VecNormalize.load(str(artifact.vecnorm_path), temp_vec)
        vecnorm.training = False
        vecnorm.norm_reward = False

        experts.append(LoadedExpert(artifact=artifact, model=model, vecnorm=vecnorm))

    return experts


def run_stage2(
    manifest_path: Path,
    stage1_root: str = "checkpoints/moe/stage1",
    output_dir: str = "checkpoints/moe/stage2",
    dry_run: bool = False,
    total_timesteps: int = 120_000,
    load_balance_coef: float = 0.02,
    diversity_coef: float = 0.01,
    gate_temperature: float = 1.0,
    train_data_path: Optional[str] = None,
    symbol: Optional[str] = None,
    config_path: Optional[str] = None,
):
    try:
        from crypto_trader.train_moe_stage1 import _prepare_training_dataframe
    except ImportError:
        from train_moe_stage1 import _prepare_training_dataframe

    artifacts = build_gate_artifacts(manifest_path=manifest_path, stage1_root=stage1_root)
    missing = validate_stage1_artifacts(artifacts)

    if dry_run:
        print("MoE Stage2 dry-run plan:")
        print(json.dumps([asdict(a) for a in artifacts], default=str, ensure_ascii=False, indent=2))
        if missing:
            print("Missing artifacts:")
            for p in missing:
                print(p)
        return artifacts

    if missing:
        raise FileNotFoundError("Stage1 expert artifacts missing:\n" + "\n".join(missing))

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    config = load_config(config_path) if config_path else get_default_config()
    if symbol:
        config.data.symbol = symbol
    symbol_to_use = symbol or config.data.symbol
    base_df = _prepare_training_dataframe(
        config,
        train_data_path=train_data_path,
        symbol_override=symbol_to_use,
    )
    experts = _load_experts(artifacts, base_df, config)

    def make_env():
        return GateRoutingEnv(
            df=base_df,
            experts=experts,
            config=config,
            load_balance_coef=load_balance_coef,
            diversity_coef=diversity_coef,
            gate_temperature=gate_temperature,
        )

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=config.model.learning_rate,
        gamma=config.model.gamma,
        n_steps=min(config.model.n_steps, 1024),
        batch_size=min(config.model.batch_size, 128),
        ent_coef=max(config.model.ent_coef, 0.01),
        clip_range=config.model.clip_range,
        gae_lambda=config.model.gae_lambda,
        seed=config.seed.global_seed,
    )

    model.learn(total_timesteps=total_timesteps)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out / "gate_model"))
    vec_env.save(str(out / "gate_vec_normalize.pkl"))

    raw_env = getattr(getattr(vec_env, "venv", vec_env), "envs", [None])[0]
    usage = raw_env.usage_ema.tolist() if raw_env is not None else [1.0 / len(artifacts)] * len(artifacts)
    metadata = {
        "manifest_path": str(manifest_path),
        "stage1_root": stage1_root,
        "load_balance_coef": load_balance_coef,
        "diversity_coef": diversity_coef,
        "gate_temperature": gate_temperature,
        "total_timesteps": int(total_timesteps),
        "usage_ema": usage,
        "expert_ids": [a.expert_id for a in artifacts],
        "symbol": config.data.symbol,
        "interval": config.data.interval,
    }

    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return artifacts


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MoE stage2 gate with frozen experts")
    parser.add_argument("--manifest", type=str, default="crypto_trader/configs/moe_experts.yaml")
    parser.add_argument("--stage1-root", type=str, default="checkpoints/moe/stage1")
    parser.add_argument("--output-dir", type=str, default="checkpoints/moe/stage2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=120000)
    parser.add_argument("--load-balance-coef", type=float, default=0.02)
    parser.add_argument("--diversity-coef", type=float, default=0.01)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--train-data-path", type=str, default=None)
    parser.add_argument("--symbol", type=str, default=None, help="Trading symbol for asset profile, e.g. ETH/USDT:USDT")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_stage2(
        manifest_path=Path(args.manifest),
        stage1_root=args.stage1_root,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        total_timesteps=args.total_timesteps,
        load_balance_coef=args.load_balance_coef,
        diversity_coef=args.diversity_coef,
        gate_temperature=args.gate_temperature,
        train_data_path=args.train_data_path,
        symbol=args.symbol,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
